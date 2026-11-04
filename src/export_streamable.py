import torch
import torch.nn as nn
from model.norm import NormType, cLN
from model.convtasnet import ConvTasNet
from model.streamable_convtasnet import StreamableConvTasNet, StreamableConvTasNetcLN
from argparse import ArgumentParser, BooleanOptionalAction
from utils.utils import construct_model, load_trained_model
from utils.conf import Conf
import yaml

parser = ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("off_onnx", type=str)
parser.add_argument("stream_onnx", type=str)
parser.add_argument("--stream_pth", type=str, default=None)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--ckpt", action=BooleanOptionalAction, default=False)
args = parser.parse_args()

with open(args.config) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)
model = load_trained_model(
    conf, eval=False, compile=args.ckpt, config_name=None, trained_model=args.model
)
model.eval()

norm_type = None
if isinstance(model.tcn.layer_norm, nn.BatchNorm1d):
    streamable = StreamableConvTasNet(model)
    norm_type = NormType.batch_norm
elif isinstance(model.tcn.layer_norm, cLN):
    streamable = StreamableConvTasNetcLN(model)
    norm_type = NormType.channel_layer_norm
else:
    raise Exception("Non streamable layer norm in TCN")
streamable.eval()

if args.stream_pth is not None:
    torch.save(streamable, args.stream_pth)

import torch.onnx

x_ns = torch.zeros(1, 16000)
torch.onnx.export(
    model,
    (x_ns),
    args.off_onnx,
    input_names=["x"],
    output_names=["y"],
    dynamo=True,
    external_data=False,
)
if norm_type == NormType.batch_norm:
    x, (audio_states, tcn_states, tail_states) = (
        torch.zeros(1, 128),
        streamable.init_states(),
    )
elif norm_type == NormType.channel_layer_norm:
    x, (audio_states, tcn_states, cln_states, tail_states) = (
        torch.zeros(1, 128),
        streamable.init_states(),
    )

input_names = ["x", "audio_states", "tcn_states", "tail_states"]
output_names = ["y", "next_audio_states", "next_tcn_states", "next_tail_states"]

if norm_type == NormType.channel_layer_norm:
    input_names.insert(-1, "cln_states")
    output_names.insert(-1, "next_cln_states")

dummy_input = (
    (x, audio_states, tcn_states, tail_states)
    if norm_type == NormType.batch_norm
    else (x, audio_states, tcn_states, cln_states, tail_states)
)
torch.onnx.export(
    streamable,
    dummy_input,
    args.stream_onnx,
    input_names=input_names,
    output_names=output_names,
    dynamo=True,
    external_data=False,
)
