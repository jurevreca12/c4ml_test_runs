import brevitas
import brevitas.nn as qnn
import numpy as np
import torch
from torch.nn import Module
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from quantizers import CommonWeightQuant
from quantizers import IntBiasQuant
from quantizers import IntActQuant

def get_conv_layer_model(
    input_size, input_ch, output_ch, kernel_size, iq, wq, bq, oq, depthwise=False
):
    class ConvLayerModel(Module):
        def __init__(self):
            super(ConvLayerModel, self).__init__()
            if depthwise:
                self.ishape = (1, output_ch) + input_size
            else:
                self.ishape = (1, input_ch) + input_size
            self.conv = qnn.QuantConv2d(
                in_channels=output_ch if depthwise else input_ch,
                out_channels=output_ch,
                groups=output_ch if depthwise else 1,
                kernel_size=kernel_size,
                stride=1,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=wq,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if wq == 1 else 2 ** (wq - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=bq,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bq - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=iq,
                input_scaling_impl_type="const",
                input_scaling_init=1 if iq == 1 else 2 ** (iq - 1),
                output_quant=IntActQuant,
                output_bit_width=oq,
                output_scaling_impl_type="const",
                output_scaling_init=1 if oq == 1 else 2 ** (oq - 1) - 1,
            )

        def forward(self, x):
            return self.conv(x)

    model = ConvLayerModel()
    wshape = (output_ch, 1 if depthwise else input_ch, kernel_size[0], kernel_size[1])
    bshape = (output_ch,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.conv.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.conv.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (32,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data
