import brevitas.nn as qnn
import numpy as np
import torch
from torch.nn import Module
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from models.quantizers import IntActQuant


def get_maxpool_layer_model(channels, input_size, kernel_size, iq):
    class MaxPoolLayerModel(Module):
        def __init__(self, input_size):
            super(MaxPoolLayerModel, self).__init__()
            self.ishape = (1, channels) + input_size
            self.in_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1),
            )
            self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size)
            self.out_quant = qnn.QuantIdentity(
                act_quant=IntActQuant,
                bit_width=iq,
                scaling_impl_type="const",
                scaling_init=1 if iq == 1 else 2 ** (iq - 1),
            )

        def forward(self, x):
            tmp = self.in_quant(x)
            tmp = self.maxpool(tmp)
            return self.out_quant(tmp)

    model = MaxPoolLayerModel(input_size)
    # set seed for repeatability
    np.random.seed(42)
    iq_type = DataType[f"INT{iq}"] if iq > 1 else DataType["BIPOLAR"]
    ishape = (8,) + model.ishape[1:]
    input_data = gen_finn_dt_tensor(iq_type, ishape)
    return model, input_data, None
