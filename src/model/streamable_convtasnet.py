from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.norm import GlobalLayerNorm, NormType, fold_conv_bn
from model.convtasnet import ConvTasNet, ConvolutionalBlock1D, TCN


# Stateless block + folded batch norm
class StreamingConvBlock1D(nn.Module):
    """Chunked streaming view of a trained, causal ConvolutionalBlock1D (eval mode, BN norm)."""

    @torch.no_grad()
    def __init__(self, src: ConvolutionalBlock1D):
        super().__init__()
        if src.normalization1.training or src.normalization2.training:
            raise ValueError("Call model.eval() first: BN must use running stats.")
        if not isinstance(src.normalization1, nn.BatchNorm1d):
            raise ValueError("Only per-frame norms (BatchNorm1d) are streamable here.")
        self.src = src  # reuse weights by reference
        self.K, self.D = src.K, src.D
        self.buf = (self.K - 1) * self.D  # past frames to keep
        self.hidden = src.dconv.in_channels

        self.conv1x1 = fold_conv_bn(self.src.conv1d, self.src.normalization2)
        src_dconv = self.src.dconv
        sdconv = nn.Conv1d(
            in_channels=src_dconv.in_channels,
            out_channels=src_dconv.in_channels,
            kernel_size=src_dconv.kernel_size,
            stride=1,
            padding=0,
            dilation=self.D,
            groups=src_dconv.groups,
        )
        sdconv.weight.copy_(src_dconv.weight)
        sdconv.bias.copy_(src_dconv.bias)
        self.dconv = fold_conv_bn(sdconv, self.src.normalization1)

    def init_state(self, batch_size, device, dtype=torch.float32) -> torch.Tensor:
        # zeros == the F.pad zeros used in the offline forward
        return torch.zeros(
            batch_size, self.hidden, self.buf, device=device, dtype=dtype
        )

    def forward(self, x, state):
        s = self.src
        y = s.conv1d(x)
        y = s.nonlinearity1(y)
        y = s.normalization1(y)                  # post-norm1 frames go into the state

        window = torch.cat([state, y], dim=-1)   # state == the offline zero pad
        z = s.dconv(window)                      # padding=0, so output has T frames
        new_state = window[..., -self.buf:]

        z = s.nonlinearity2(z)
        z = s.normalization2(z)                  # on z, not y

        skip = s.skip_conv1d(z)
        out = x + s.out_conv1d(z)
        return out, skip, new_state


class StreamableTCN(nn.Module):
    """
    Streamable Temporal Convolutional Network (TCN)
    Uses packed states consisting of bufferized conv1x1 outputs
    Return masks of shape [Batch, n_src, N, L]

    Args:
        base: The trained TCN
    """

    def __init__(self, base: TCN):
        super().__init__()
        # (asserts on eval mode / non-global norm as before)
        self.base = base
        self.layers = nn.ModuleList(StreamingConvBlock1D(l) for l in base.layers)
        self.sizes = [l.buf for l in self.layers]  # e.g. [2, 4, ..., 256, 2, ...]
        self.hidden = self.layers[0].hidden

    def init_states(self, batch_size, device="cpu", dtype=torch.float32):
        return torch.zeros(
            batch_size, self.hidden, sum(self.sizes), device=device, dtype=dtype
        )

    def forward(self, x, packed):
        # x: [B, N, T]   packed: [B, H, sum(sizes)]
        b = self.base
        B, T = x.size(0), x.size(-1)
        states = packed.split(self.sizes, dim=-1)  # exports as a single ONNX Split

        x = b.bottleneck_layer(b.layer_norm(x))
        skip_sum, new_states = 0, []
        for layer, s in zip(self.layers, states):
            x, skip, s = layer(x, s)
            skip_sum = skip_sum + skip
            new_states.append(s)

        out = b.mask_activation(b.conv1d_out(b.non_linearity_out(skip_sum)))
        masks = out.reshape(B, b.n_src, b.input_dim, T)
        return masks, torch.cat(new_states, dim=-1)  # single Concat


class StreamableConvTasNet(nn.Module):
    def __init__(self, base: ConvTasNet, chunk: int = 128):
        super().__init__()
        self.base = base.eval()
        self.tcn = StreamableTCN(base.tcn)
        self.overlap = base.encoder_kernel_size - base.stride  # 32
        self.chunk = chunk
        self.T = (
            self.overlap + chunk - base.encoder_kernel_size
        ) // base.stride + 1  # 16

    def init_states(self, B: int = 1, device="cpu", dtype=torch.float32):
        audio_buffer = torch.zeros(B, 1, self.overlap, device=device, dtype=dtype)
        ola_carry = torch.zeros(
            B, self.base.n_src, self.overlap, device=device, dtype=dtype
        )
        tcn_state = self.tcn.init_states(B, device, dtype)
        return audio_buffer, tcn_state, ola_carry

    def forward(self, x, audio_buffer, tcn_state, ola_carry):
        b = self.base
        x = x.unsqueeze(1)  # [B, 1, chunk]
        B = x.size(0)

        x_with_overlap = torch.cat([audio_buffer, x], dim=-1)  # [B, 1, overlap+chunk]
        new_audio_buffer = x_with_overlap[..., -self.overlap :]

        w = b.encoder(x_with_overlap)  # [B, N, T]
        masks, tcn_state = self.tcn(w, tcn_state)
        masked = (masks * w.unsqueeze(1)).reshape(B * b.n_src, b.enc_dim, self.T)

        dec = b.decoder(masked).reshape(B, b.n_src, self.chunk + self.overlap)

        # overlap-add: finish the head with the previous chunk's tail
        head = dec[..., : self.overlap] + ola_carry
        body = dec[..., self.overlap : self.chunk]
        out = torch.cat([head, body], dim=-1)  # [B, n_src, chunk]
        new_ola_carry = dec[..., self.chunk :]  # [B, n_src, overlap]

        return out, new_audio_buffer, tcn_state, new_ola_carry
