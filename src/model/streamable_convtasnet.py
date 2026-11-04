from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.norm import GlobalLayerNorm, NormType, fold_conv_bn, cLN
from model.convtasnet import ConvTasNet, ConvolutionalBlock1D, TCN

class StreamingcLN(nn.Module):
    """Chunked cLN. state: [B, 3] = (cum_sum, cum_pow_sum, n_entries), zeros at stream start."""

    def __init__(self, src: cLN):
        super().__init__()
        self.src = src                                   # reuse gain / bias / eps

    @staticmethod
    def init_state(batch_size, device="cpu", dtype=torch.float32):
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def forward(self, x, state):                         # x: [B, C, T]
        s = self.src
        C, T = x.size(1), x.size(2)
        cs0, cp0, n0 = state[:, 0:1], state[:, 1:2], state[:, 2:3]    # each [B, 1]

        # offline cumsum, continued from the previous chunk's totals
        cum_sum = cs0 + torch.cumsum(x.sum(1), dim=1)                 # [B, T]
        cum_pow = cp0 + torch.cumsum(x.pow(2).sum(1), dim=1)          # [B, T]
        steps = torch.arange(1, T + 1, dtype=x.dtype, device=x.device).view(1, -1)
        cnt = n0 + C * steps                                          # [B, T]

        mean = cum_sum / cnt
        var = (cum_pow - 2 * mean * cum_sum) / cnt + mean.pow(2)      # same formula as offline
        std = (var.clamp_min(0) + s.eps).sqrt()

        y = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
        y = y * s.gain + s.bias                                       # [1, C, 1] broadcasts
        new_state = torch.stack([cum_sum[:, -1], cum_pow[:, -1], cnt[:, -1]], dim=1)
        return y, new_state

# Stateless block + folded batch norm
class StreamingConvBlock1DcLN(nn.Module):
    """Chunked streaming view of a trained, causal ConvolutionalBlock1D (eval mode, BN norm)."""

    @torch.no_grad()
    def __init__(self, src: ConvolutionalBlock1D):
        super().__init__()
        if src.normalization1.training or src.normalization2.training:
            raise ValueError("Call model.eval() first: BN must use running stats.")
        if isinstance(src.normalization1, cLN):
            self.norm_type = NormType.channel_layer_norm
        else:
            raise ValueError("Only cLN is streamable here.")
        self.src = src  # reuse weights by reference
        self.K, self.D = src.K, src.D
        self.buf = (self.K - 1) * self.D  # past frames to keep
        self.hidden = src.dconv.in_channels
        self.normalization1 = StreamingcLN(self.src.normalization1)
        self.normalization2 = StreamingcLN(self.src.normalization2)

    def init_state(self, batch_size, device, dtype=torch.float32) -> torch.Tensor:
        # zeros == the F.pad zeros used in the offline forward
        return (
            torch.zeros(batch_size, self.hidden, self.buf, device=device, dtype=dtype),
            StreamingcLN(batch_size, device, dtype),
            StreamingcLN(batch_size, device, dtype),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor, cln_states: torch.Tensor):
        cln1_state, cln2_state = cln_states[..., :3], cln_states[..., 3:]
        s = self.src
        y = s.conv1d(x)
        # y = self.conv1x1(x)
        y, cln1_state = self.normalization1(y, cln1_state)  # post-norm1 frames go into the state
        y = s.nonlinearity1(y)

        window = torch.cat([state, y], dim=-1)  # state == the offline zero pad
        z = s.dconv(window)  # padding=0, so output has T frames
        # z = self.dconv(window)                      # padding=0, so output has T frames
        new_state = window[..., -self.buf :]

        z, cln2_state = self.normalization2(z, cln2_state)  # on z, not y
        z = s.nonlinearity2(z)

        skip = s.skip_conv1d(z)
        out = x + s.out_conv1d(z)
        return out, skip, new_state, torch.cat([cln1_state, cln2_state], dim=-1)


class StreamableTCNcLN(nn.Module):
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
        self.layers = nn.ModuleList(StreamingConvBlock1DcLN(l) for l in base.layers)
        self.sizes = [l.buf for l in self.layers]  # e.g. [2, 4, ..., 256, 2, ...]
        self.hidden = self.layers[0].hidden
        self.layer_norm = StreamingcLN(self.base.layer_norm)

    def init_states(self, batch_size, device="cpu", dtype=torch.float32):
        return torch.zeros(
            batch_size, self.hidden, sum(self.sizes), device=device, dtype=dtype
        ), torch.zeros(batch_size, 3 * (2*len(self.layers)+1), device=device, dtype=dtype)

    def forward(self, x : torch.Tensor, packed:torch.Tensor, packed_cln_states: torch.Tensor):
        # x: [B, N, T]   packed: [B, H, sum(sizes)]
        b = self.base
        B, T = x.size(0), x.size(-1)
        states = packed.split(self.sizes, dim=-1)  # exports as a single ONNX Split
        cln_states = packed_cln_states[...,3:].split(6, dim=-1)

        skip_sum, new_states, new_cln_states = 0, [], []
        z, cs = self.layer_norm(x, packed_cln_states[..., :3])
        new_cln_states.append(cs)
        x = b.bottleneck_layer(z)
        for layer, s, cln_state in zip(self.layers, states, cln_states):
            x, skip, s, cs = layer(x, s, cln_state)
            skip_sum = skip_sum + skip
            new_states.append(s)
            new_cln_states.append(cs)

        out = b.mask_activation(b.conv1d_out(b.non_linearity_out(skip_sum)))
        masks = out.reshape(B, b.n_src, b.input_dim, T)
        return masks, torch.cat(new_states, dim=-1), torch.cat(new_cln_states, dim=-1)  # single Concat


class StreamableConvTasNetcLN(nn.Module):
    def __init__(self, base: ConvTasNet, chunk: int = 128):
        super().__init__()
        self.base = base.eval()
        self.tcn = StreamableTCNcLN(base.tcn)
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
        tcn_state, cln_states = self.tcn.init_states(B, device, dtype)
        return audio_buffer, tcn_state, cln_states, ola_carry

    def forward(self, x, audio_buffer, tcn_state, cln_states, ola_carry):
        b = self.base
        x = x.unsqueeze(1)  # [B, 1, chunk]
        B = x.size(0)

        x_with_overlap = torch.cat([audio_buffer, x], dim=-1)  # [B, 1, overlap+chunk]
        new_audio_buffer = x_with_overlap[..., -self.overlap :]

        w = b.encoder(x_with_overlap)  # [B, N, T]
        masks, tcn_state, cln_states = self.tcn(w, tcn_state, cln_states)
        masked = (masks * w.unsqueeze(1)).reshape(B * b.n_src, b.enc_dim, self.T)

        dec = b.decoder(masked).reshape(B, b.n_src, self.chunk + self.overlap)

        # overlap-add: finish the head with the previous chunk's tail
        head = dec[..., : self.overlap] + ola_carry
        body = dec[..., self.overlap : self.chunk]
        out = torch.cat([head, body], dim=-1)  # [B, n_src, chunk]
        new_ola_carry = dec[..., self.chunk :]  # [B, n_src, overlap]

        return out, new_audio_buffer, tcn_state, cln_states, new_ola_carry

##############################
###### Batch Norm model ######
##############################
class StreamingConvBlock1D(nn.Module):
    """Chunked streaming view of a trained, causal ConvolutionalBlock1D (eval mode, BN norm)."""

    @torch.no_grad()
    def __init__(self, src: ConvolutionalBlock1D):
        super().__init__()
        if src.normalization1.training or src.normalization2.training:
            raise ValueError("Call model.eval() first: BN must use running stats.")
        if isinstance(src.normalization1, nn.BatchNorm1d):
            self.norm_type = NormType.batch_norm
        else:
            raise ValueError(
                "Only per-frame norms (BatchNorm1d and cLN) are streamable here."
            )
        self.src = src  # reuse weights by reference
        self.K, self.D = src.K, src.D
        self.buf = (self.K - 1) * self.D  # past frames to keep
        self.hidden = src.dconv.in_channels

        if self.norm_type == NormType.batch_norm:
            self.conv1x1 = fold_conv_bn(self.src.conv1d, self.src.normalization1)
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
            self.dconv = fold_conv_bn(sdconv, self.src.normalization2)

    def init_state(self, batch_size, device, dtype=torch.float32) -> torch.Tensor:
        # zeros == the F.pad zeros used in the offline forward
        return torch.zeros(
            batch_size, self.hidden, self.buf, device=device, dtype=dtype
        )

    def forward(self, x, state):
        s = self.src
        y = s.conv1d(x)
        # y = self.conv1x1(x)
        y = s.normalization1(y)  # post-norm1 frames go into the state
        y = s.nonlinearity1(y)

        window = torch.cat([state, y], dim=-1)  # state == the offline zero pad
        z = s.dconv(window)  # padding=0, so output has T frames
        # z = self.dconv(window)                      # padding=0, so output has T frames
        new_state = window[..., -self.buf :]

        z = s.normalization2(z)  # on z, not y
        z = s.nonlinearity2(z)

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
