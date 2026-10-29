from enum import Enum

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class NormType(Enum):
    global_layer_norm = "global_layer_norm"
    channel_layer_norm = "channel_layer_norm"
    batch_norm = "batch_norm"

def global_layer_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global Layer Normalization (gLN)

    Attributes:
        input (torch.Tensor): the input tensor of shape [batch, channel_size, length]
        gamma (torch.Tensor): the scale parameter of shape [1, channel_size, 1]
        beta (torch.Tensor): the shift parameter of shape [1, channel_size, 1]
        eps (float): a small value to avoid division by zero
    """
    mean = input.mean(dim=[1, 2], keepdim=True)
    # mean is of shape [batch, 1, 1]
    var = input.var(dim=[1, 2], keepdim=True, unbiased=False)
    # var is of shape [batch, 1, 1]

    value = (input - mean) / torch.sqrt(var + eps)
    return value * gamma + beta


def global_layer_norm_using_torch_layer_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    channel_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global Layer Normalization (gLN)

    This method is equivalent to the forward method but use `nn.LayerNorm` instead of the custom implementation.
    This requires to reshape and `repeat` the weights to match the expected shape of `nn.LayerNorm`, introducing a computational overhead.

    The method can be useful to replace the custom implementation by an optimized LayerNorm during ONNX export or inference.

    Note: this method requires to compute the length of the input tensor to repeat the weights, which may not be compatible
    with dynamic input shapes.

    Attributes:
        input (torch.Tensor): the input tensor of shape [batch, channel_size, length]
        gamma (torch.Tensor): the scale parameter of shape [1, channel_size, 1]
        beta (torch.Tensor): the shift parameter of shape [1, channel_size, 1]
        eps (float): a small value to avoid division by zero
    """
    length = input.size(-1)
    weight = gamma.squeeze(0).repeat(1, length)
    bias = beta.squeeze(0).repeat(1, length)
    return nn.functional.layer_norm(input, [channel_size, length], weight, bias, eps)


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN)

    GlobalLayerNorm compute

    $ "gLN"(F) = (F - E[F]) / sqrt("Var"[F] + epsilon) gamma + beta $

    where $gamma, beta in RR^(N times 1)$. It intends to normalize across both channel and time dimensions sharing the same weights for the time dimension.

    It differ slightly from `nn.LayerNorm` which weights have the same shape as `normalized_shape` and compute the esperance and variance over the last $D$ dimensions, where D is the dimension of `normalized_shape`.

    Attributes:
        channel_size (int): the number of channels in the input tensor

    The input tensor $x$ should have shape [batch, channel_size, length] where length correspond to the time dimension.
    The output tensor will have the same shape as the input tensor.
    """

    def __init__(self, channel_size: int, eps: float = 1e-8):
        super(GlobalLayerNorm, self).__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return global_layer_norm(x, self.gamma, self.beta, self.eps)

    def forward_torch_layer_norm(self, x: torch.Tensor):
        """
        This method is equivalent to the forward method but use `nn.LayerNorm` instead of the custom implementation.
        This requires to reshape and `repeat` the weights to match the expected shape of `nn.LayerNorm`, introducing a computational overhead.

        The method can be useful to replace the custom implementation by an optimized LayerNorm during ONNX export or inference.
        """
        return global_layer_norm_using_torch_layer_norm(
            x, self.gamma, self.beta, self.channel_size, self.eps
        )


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


@torch.no_grad()
def fold_conv_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d) -> torch.nn.Conv1d:
    """Folds a batch norm into a preceding convolution (conv 1x1 or depthwise convolution).
    Should be used only for evaluation models.

    Args:
        conv (torch.nn.Conv1d): the convolution preceding the norm
        bn (torch.nn.BatchNorm1d): the batch norm to fold

    Returns:
        torch.nn.Conv1d: the resulting convolution containing the folded batch norm
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # [C_out]
    fused = torch.nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    fused.weight.copy_(conv.weight * scale.view(-1, 1, 1))
    conv_bias = (
        conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    )
    fused.bias.copy_((conv_bias - bn.running_mean) * scale + bn.bias)
    return fused
