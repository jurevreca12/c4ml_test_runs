import brevitas.nn as qnn
import numpy as np
import torch
from torch.nn import Module
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from models.quantizers import CommonWeightQuant
from models.quantizers import IntBiasQuant
from models.quantizers import IntActQuant


def get_linear_layer_model(in_features, out_features, bias, iq, wq, bq, oq):
    class LinearLayerModel(Module):
        def __init__(self):
            super(LinearLayerModel, self).__init__()
            self.ishape = (1, in_features)
            self.linear = qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
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
            return self.linear(x)

    model = LinearLayerModel()
    wshape = (out_features, in_features)
    bshape = (out_features,)
    # set seed for repeatability
    np.random.seed(42)
    wq_type = DataType[f"INT{wq}"] if wq > 1 else DataType["BIPOLAR"]
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    weights = gen_finn_dt_tensor(wq_type, wshape)
    bias = gen_finn_dt_tensor(DataType[f"INT{bq}"], bshape)
    model.linear.weight = torch.nn.Parameter(torch.from_numpy(weights).float())
    model.linear.bias = torch.nn.Parameter(torch.from_numpy(bias).float())
    ishape = (32,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data
