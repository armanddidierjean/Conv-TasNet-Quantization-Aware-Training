from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.norm import GlobalLayerNorm, NormType, cLN
from utils.conf import Conf

"""
We have T segments (k in [1, T])

L: length of each segment
N-> enc_dim: probably number of filters in autoencoder?

TODO: pass params as params
"""


class Encoder(nn.Module):
    """
    An input segment $x_k$ (dim [1, L]) is transformed into a
    N-dimensional representation, $w$ (dim [1, N])
    by a 1-D convolution operation $w = H(xU)$

    > H(·) is an optional nonlinear function.
    > In [21], [26], H(·) was the rectified linear unit (ReLU)
    > to ensure that the representation is non-negative

    Args:
        enc_dim: `N`, number of filters in autoencoder
        kernel_size: size of the convolutional kernel
        stride: stride of the convolution, usually set to kernel_size // 2 to have 50% overlap between segments

    L: length of each segment
    Batch: batch size

    x ([Batch, 1, L]) -> w ([Batch, N, L'])
    Length of the output segments will depend on the kernel size and stride of the convolution
    """

    def __init__(
        self,
        enc_dim: int = 512,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()

        self.stride = stride
        self.kernel_size = kernel_size

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=enc_dim,
            kernel_size=self.kernel_size,
            bias=False,
            stride=self.stride,
            padding=(0),
            dilation=1,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: a [Batch, 1, L] or [Batch, L] tensor containing the input segments

        Returns:
            A [Batch, N, L'] tensor
        """

        return self.conv1d(input)

    def get_loggable_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield "encoder_conv1d.weight", self.conv1d.weight


class Decoder(nn.Module):
    """
    An input segment $x_k$ (dim [1, L]) is transformed into a
    N-dimensional representation, $w$ (dim [1, N])
    by a 1-D convolution operation $x̂ = wV$

    > H(·) is an optional nonlinear function.
    > In [21], [26], H(·) was the rectified linear unit (ReLU)
    > to ensure that the representation is non-negative

    Args:
        enc_dim: N, number of filters in autoencoder
        kernel_size: size of the convolutional kernel
        stride: stride of the convolution, usually set to kernel_size // 2 to have 50% overlap between segments

    L: length of each segment
    Batch: batch size

    w ([Batch, N, L']) -> x̂ ([Batch, 1, L])
    Length of the output segments will depend on the kernel size and stride of the convolution
    """

    def __init__(
        self,
        enc_dim: int = 512,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()

        self.stride = stride
        self.kernel_size = kernel_size

        # TODO: is the output really 2?
        self.convTranspose1d = nn.ConvTranspose1d(
            in_channels=enc_dim,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            stride=stride,
            padding=0,
            dilation=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: a [Batch, N, L'] tensor

        Returns:
            A [Batch, 1, L] tensor containing separated sources
        """

        return self.convTranspose1d(input)

    def get_loggable_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield "decoder_convtranspose1d.weight", self.convTranspose1d.weight


class ConvolutionalBlock1D(nn.Module):
    """
    [Batch, B, L] -> [Batch, H, L] -> output: [Batch, B, L]
                                      skip:   [Batch, Sc, L]

    Args:
        in_channels: `B`, number of input channels
        hidden_channels: number of channels in the hidden layers of this convolutional block
        skip_channels: `Sc`, number of channels in the skip connection output
        kernel_size: kernel size of the depth-wise convolution
        padding: padding of the depth-wise convolution
        dilation: dilation of the depth-wise convolution

    B correspond to the number of channels in bottleneck and the residual paths
    """

    # TODO: should we use a S-conv?

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: NormType,
    ):
        super().__init__()
        # TODO: causal padding
        # TODO: norm for causal

        self.dilation = dilation

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.nonlinearity1 = nn.PReLU()
        if norm_type == NormType.channel_layer_norm:
            self.normalization1 = cLN(hidden_channels)
        elif norm_type == NormType.global_layer_norm:
            self.normalization1 = GlobalLayerNorm(channel_size=hidden_channels)
        elif norm_type == NormType.batch_norm:
            self.normalization1 = nn.BatchNorm1d(hidden_channels)

        self.dconv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            padding=padding,
        )

        self.nonlinearity2 = nn.PReLU()
        if norm_type == NormType.channel_layer_norm:
            self.normalization2 = cLN(hidden_channels)
        elif norm_type == NormType.global_layer_norm:
            self.normalization2 = GlobalLayerNorm(channel_size=hidden_channels)
        elif norm_type == NormType.batch_norm:
            self.normalization2 = nn.BatchNorm1d(hidden_channels)

        self.skip_conv1d = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=skip_channels,
            kernel_size=1,
        )
        self.out_conv1d = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

        self.K = kernel_size
        self.D = dilation
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: a [Batch, B, L] tensor

        Returns:
            A tuple containing the output tensor and the skip connection tensor
            output: [Batch, B, L]
            skip:   [Batch, Sc, L]
        """
        output = self.conv1d(input)
        output = self.nonlinearity1(output)
        output = self.normalization1(output)

        output = F.pad(output, ((self.K - 1) * self.D, 0))
        output = self.dconv(output)

        output = self.nonlinearity2(output)
        output = self.normalization2(output)

        skip = self.skip_conv1d(output)
        residual = self.out_conv1d(output)

        # TODO: we put this here instead of in the TCN which may be unconventional
        output = input + residual
        return output, skip

    def get_loggable_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield "ConvolutionalBlock1D_conv1d.weight", self.conv1d.weight
        yield "ConvolutionalBlock1D_dconv.weight", self.dconv.weight
        yield "ConvolutionalBlock1D_skip_conv1d.weight", self.skip_conv1d.weight
        yield "ConvolutionalBlock1D_out_conv1d.weight", self.out_conv1d.weight


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    Fully-convolutional separation module that consists of stacked 1-D dilated convolutional blocks
    Dilation factors increase exponentially: M convolutional blocks with dilation factors 1,2,4,...,2^(M-1) are repeated R times

    Return masks of shape [Batch, n_src, N, L]

    bottleneck_dim: `B`, dimension of the bottleneck layer

    [Batch, N, L] -> [Batch, n_src, N, L]

    Args:
        input_dim: `N`, dimension of the input layer, correspond to the number of filters in autoencoder
        n_src: number of speakers/sources to separate
        bottleneck_dim: `B`, dimension of the bottleneck layer
        hidden_dim: `H`, dimension of the hidden layers
        skip_dim: `Sc`, dimension of the skip connection layer
        kernel_size: `P`, kernel size of the depth-wise convolution
        nb_layers: `R`, number of layers
        nb_conv_blocks_per_layer: `X` (`M` in equations), number of convolutional blocks per layer
        norm_type: type of normalization to use in the convolutional blocks
            global layer norm or channel-wise layer norm
            gln usually works better for non-causal configuration.

    """

    def __init__(
        self,
        input_dim: int,
        n_src: int,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        skip_dim: int = 128,
        kernel_size: int = 3,
        nb_layers: int = 3,
        nb_conv_blocks_per_layer: int = 8,
        norm_type: NormType = NormType.global_layer_norm,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_src = n_src
        self.skip_dim = skip_dim
        self.nb_layers = nb_layers
        self.nb_conv_blocks_per_layer = nb_conv_blocks_per_layer

        # TODO: support multiple type of normalization
        if norm_type == NormType.channel_layer_norm:
            self.layer_norm = cLN(input_dim)
        elif norm_type == NormType.global_layer_norm:
            self.layer_norm = GlobalLayerNorm(channel_size=input_dim)
        elif norm_type == NormType.batch_norm:
            self.layer_norm = torch.nn.BatchNorm1d(input_dim)

        self.bottleneck_layer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=bottleneck_dim,
            kernel_size=1,
        )

        self.receptive_field = 0

        self.layers: nn.ModuleList[ConvolutionalBlock1D] = nn.ModuleList(modules=[])
        for r in range(nb_layers):
            for m in range(nb_conv_blocks_per_layer):
                dilation = 2**m
                # todo support causal
                conv_block = ConvolutionalBlock1D(
                    in_channels=bottleneck_dim,
                    hidden_channels=hidden_dim,
                    skip_channels=skip_dim,
                    kernel_size=kernel_size,
                    # TODO: we may need to compute a different padding depending on the stride
                    padding=0,
                    dilation=dilation,
                    norm_type=norm_type,
                )

                self.layers.append(conv_block)

                if r == 0 and m == 0:
                    self.receptive_field += kernel_size
                else:
                    # We only implement dilated layers
                    # if self.dilated:
                    self.receptive_field += (kernel_size - 1) * 2**m
                    # else:
                    #     self.receptive_field += kernel_size - 1

        self.non_linearity_out = nn.PReLU()
        self.conv1d_out = nn.Conv1d(
            in_channels=skip_dim,
            out_channels=input_dim * n_src,
            kernel_size=1,
        )
        # We can use Sigmoid or Softmax activation
        # It is observed that with a Linear Encoder, Sigmoid performs better
        self.mask_activation = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, length_L = input.size()
        # input shape: [Batch, N, L]
        normed_input = self.layer_norm(input)
        # assert normed_input.shape == input.shape
        # normed_input shape: [Batch, N, L]
        bn_input = self.bottleneck_layer(normed_input)
        # bn_input shape: [Batch, B, L]
        conv_block_input = bn_input

        skip_connection_sum = torch.zeros(
            (conv_block_input.size(0), self.skip_dim, conv_block_input.size(2)),
            device=conv_block_input.device,
        )
        # skip_connection_sum shape: [Batch, Sc, L]

        for conv_block_layer in self.layers:
            conv_block_output, skip = conv_block_layer(conv_block_input)
            # conv_block_output shape: [Batch, B, L]
            # skip shape: [Batch, Sc, L]
            conv_block_input = conv_block_output
            skip_connection_sum = skip_connection_sum + skip

        output = self.non_linearity_out(skip_connection_sum)
        # output: [Batch, Sc, L]
        output = self.conv1d_out(output)
        # output: [Batch, N * n_src, L]
        output = self.mask_activation(output)
        # output: [Batch, N * n_src, L]
        masks = output.view(batch_size, self.n_src, self.input_dim, length_L)
        # masks: [Batch, n_src, N, L]
        return masks

    def get_loggable_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield "TCN_bottleneck_layer.weight", self.bottleneck_layer.weight

        for r in range(self.nb_layers):
            for m in range(self.nb_conv_blocks_per_layer):
                layer = self.layers[r * self.nb_conv_blocks_per_layer + m]
                for name, param in layer.get_loggable_weights():
                    yield f"TCN_layer_{r}_{m}.{name}", param

        yield "TCN_conv1d_out.weight", self.conv1d_out.weight


class ConvTasNet(nn.Module):
    """
    M convolutional blocks with dilation factors
    1,2,4,...,2M-1 are repeated R times

    [Batch, L] or [Batch, 1, L] ->

    Args:
        Encoder/Decoder config:
            enc_dim: `N`, number of filters in autoencoder
            n_src: number of speakers/sources to separate
            encoder_kernel_size: kernel size of the encoder convolution
            encoder_stride: stride of the encoder convolution
                If not specified, it will be set to encoder_kernel_size // 2
        Masker network config:
            bottleneck_dim: `B`, dimension of the bottleneck layer
            hidden_dim: `H`, dimension of the hidden layers
            skip_dim: `Sc`, dimension of the skip connection layer
            kernel_size: `P`, kernel size of the depth-wise convolution
            nb_layers: `R`, number of layers
            nb_conv_blocks_per_layer: `X` (`M` in equations), number of convolutional blocks per layer
            norm_type: Type of normalization to use in the convolutional blocks of the TCN
                global layer norm or channel-wise layer norm
                gln usually works better for non-causal configuration
    """

    def __init__(
        self,
        #
        enc_dim: int = 512,
        n_src: int = 2,
        encoder_kernel_size: int = 16,
        encoder_stride: int | None = None,
        #
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        skip_dim: int = 128,
        kernel_size: int = 3,
        nb_layers: int = 3,
        nb_conv_blocks_per_layer: int = 8,
        norm_type: NormType = NormType.global_layer_norm,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.n_src = n_src

        self.stride = encoder_stride or encoder_kernel_size // 2
        self.encoder_kernel_size = encoder_kernel_size

        self.encoder = Encoder(
            enc_dim=enc_dim,
            kernel_size=encoder_kernel_size,
            stride=self.stride,
        )

        # TODO: we may want to allow for causal config and norms
        # TODO: we may want to try different types of normalization/nonlin

        # The number of input channel of the TCN should be
        # the number of filters in the autoencoder, which is the output dimension of the encoder
        self.tcn = TCN(
            input_dim=enc_dim,
            n_src=n_src,
            bottleneck_dim=bottleneck_dim,
            skip_dim=skip_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            nb_layers=nb_layers,
            nb_conv_blocks_per_layer=nb_conv_blocks_per_layer,
            norm_type=norm_type,
        )

        self.decoder = Decoder(
            enc_dim=enc_dim,
            kernel_size=encoder_kernel_size,
            stride=self.stride,
        )

    @property
    def receptive_field(self) -> int:
        """
        Returns the receptive field of the TCN in number of samples
        """
        return self.tcn.receptive_field

    def pad_signal(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        n = x.size(-1)
        P = self.encoder_kernel_size - self.stride          # 32 left pad = streaming buffer size
        R0 = (-n) % self.stride                             # align length to the stride
        x = F.pad(x, (P, P + R0))                           # (P, 32 + R0)
        return x, R0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input: [Batch, L] or [Batch, 1, L]
        P = self.encoder_kernel_size - self.stride
        n = input.size(-1)
        # First we pad
        padded_input, rest = self.pad_signal(input)

        enc_output_mixture = self.encoder(padded_input)
        # enc_output_mixture shape: [Batch, N, L']

        tcn_output_masks = self.tcn(enc_output_mixture)
        # tcn_output_masks shape: [Batch, n_src, N, L']

        # We apply the masks
        # enc_output_mixture.unsqueeze(1) shape: [Batch, 1, N, L']
        masked_output = tcn_output_masks * enc_output_mixture.unsqueeze(1)
        # masked_output shape: [Batch, n_src, N, L']

        # We need to reshape the masked output to apply the decoder on each source separately
        batch_size, n_src, enc_dim, length_L = masked_output.size()
        masked_output = masked_output.view(batch_size * n_src, enc_dim, length_L)
        # masked_output shape: [Batch * n_src, N, L']
        # we treat each source as a separate sample in the batch for the decoder

        dec_output = self.decoder(masked_output)
        # dec_output shape: [Batch * n_src, 1, L]

        # We remove extra
        # See the paper implementation
        output = dec_output[:, :, P : P + n].contiguous()
        output = output.view(batch_size, n_src, -1)
        # output shape: [Batch, n_src, L]
        return output

    def get_loggable_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, param in self.encoder.get_loggable_weights():
            yield f"convtasnet.{name}", param

        for name, param in self.tcn.get_loggable_weights():
            yield f"convtasnet.{name}", param

        for name, param in self.decoder.get_loggable_weights():
            yield f"convtasnet.{name}", param


def test_encoder():
    print("Testing Encoder...")
    enc_dim = 512
    batch, length_L = 2, 32000

    x = torch.rand(batch, length_L)

    encoder = Encoder(enc_dim=enc_dim)
    enc_output = encoder(x)

    print("Output shape:", enc_output.shape)
    assert (enc_output.size(0), enc_output.size(1)) == (x.size(0), enc_dim), (
        f"Encoder output shape {enc_output.shape} does not match expected shape ({x.size(0)}, {enc_dim}, _)"
    )


def test_decoder():
    print("Testing Decoder...")
    enc_dim = 512
    batch, length_L = 2, 32000

    x = torch.rand(batch, enc_dim, length_L)

    decoder = Decoder()
    dec_output = decoder(x)

    print("Output shape:", dec_output.shape)
    assert (dec_output.size(0), dec_output.size(1)) == (x.size(0), 1), (
        f"Decoder output shape {dec_output.shape} does not match expected shape ({x.size(0)}, 1, _)"
    )


def test_ConvolutionalBlock1D():
    for dilation_factor in range(10):
        dilation = 2**dilation_factor
        print(f"Testing ConvolutionalBlock1D with dilation {dilation}...")
        x = torch.rand(2, 512, 100)
        conv_block = ConvolutionalBlock1D(
            in_channels=512,
            hidden_channels=512,
            skip_channels=128,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            norm_type=NormType.global_layer_norm,
        )
        output, skip = conv_block(x)
        print("output.shape:", output.shape)
        print("skip.shape:", skip.shape)
        assert output.shape == (x.size(0), x.size(1), x.size(2)), (
            f"Output shape {output.shape} does not match input shape {x.shape}"
        )
        assert skip.shape == (x.size(0), 128, x.size(2)), (
            f"Skip connection shape {skip.shape} does not match input shape {x.shape}"
        )


def test_TCN():
    batch, N, L = 2, 512, 100
    n_src = 2
    x = torch.rand(batch, N, L)
    tcn = TCN(
        input_dim=512,
        n_src=n_src,
        bottleneck_dim=512,
        skip_dim=128,
        hidden_dim=512,
        nb_layers=8,
        nb_conv_blocks_per_layer=4,
    )
    output = tcn(x)
    print("output.shape:", output.shape)
    assert output.shape == (batch, n_src, N, L), (
        f"Output shape {output.shape} does not match input shape {x.shape}"
    )


def test_ConvTasNet():
    print("Testing ConvTasNet...")

    x = torch.rand(6, 100)

    with open("conf.local.yml") as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    print("conf.masknet.kernel_size", conf.masknet.kernel_size)

    convTasNet = ConvTasNet(
        enc_dim=conf.encoder.n_filters,
        n_src=conf.data.n_src,
        encoder_kernel_size=conf.encoder.encoder_kernel_size,
        encoder_stride=conf.encoder.encoder_stride,
        bottleneck_dim=conf.masknet.bottleneck_dim,
        hidden_dim=conf.masknet.hidden_dim,
        skip_dim=conf.masknet.skip_dim,
        # kernel_size=conf.masknet.kernel_size,
        nb_layers=conf.masknet.nb_layers,
        nb_conv_blocks_per_layer=conf.masknet.nb_conv_blocks_per_layer,
        norm_type=conf.masknet.norm_type,
    )
    # convTasNet = ConvTasNet()
    enc_output = convTasNet(x)
    print("Output shape:", enc_output.shape)
    assert enc_output.shape == (x.size(0), 2, x.size(1)), (
        f"Output shape {enc_output.shape} does not match expected shape ({x.size(0)}, 2, {x.size(1)})"
    )


if __name__ == "__main__":
    test_encoder()
    test_decoder()
    test_ConvolutionalBlock1D()
    test_TCN()
    test_ConvTasNet()
