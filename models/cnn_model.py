import brevitas.nn as qnn
import torch
from torch.nn import Module
from models.quantizers import WeightPerTensorQuant
from models.quantizers import IntBiasQuant


def get_cnn_model(bitwidth, bias_bitwidth=8, input_bitwidth=8, use_bn=True):
    class ConvBlock(Module):
        def __init__(self, input_ch, output_ch, conv_kernel_size=3, maxpool_kernel_size=2):
            super(ConvBlock, self).__init__()
            self.conv = qnn.QuantConv2d(
                in_channels=input_ch,
                out_channels=output_ch,
                groups=input_ch,
                kernel_size=conv_kernel_size,
                stride=1,
                bias=True,
                weight_quant=WeightPerTensorQuant,
                weight_bit_width=bitwidth,
                bias_quant=IntBiasQuant,
                bias_bit_width=bias_bitwidth,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bias_bitwidth - 1) - 1,
            )
            if use_bn:
                self.bn = torch.nn.BatchNorm2d(output_ch)
            self.relu = qnn.QuantReLU(bit_width=bitwidth, scaling_impl_type='const', scaling_init=(2**bitwidth) - 1)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size)
            self.quant = qnn.QuantIdentity(
                bit_width=bitwidth,
                scaling_impl_type='const',
                scaling_init=2**(bitwidth) - 1,
                signed=False  # ReLU
            )

        def forward(self, x):
            x = self.conv(x)
            if use_bn:
                x = self.bn(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.quant(x)
            return x

    class DenseBlock(Module):
        def __init__(self, in_features, out_features):
            super(DenseBlock, self).__init__()
            self.dense = qnn.QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                weight_quant=WeightPerTensorQuant,
                weight_bit_width=bitwidth,
                bias_quant=IntBiasQuant,
                bias_bit_width=bias_bitwidth,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (bias_bitwidth - 1) - 1,
            )
            if use_bn:
                self.bn = torch.nn.BatchNorm1d(out_features)
            self.relu = qnn.QuantReLU(bit_width=bitwidth, scaling_impl_type='const', scaling_init=(2**bitwidth) - 1)

        def forward(self, x):
            x = self.dense(x)
            if use_bn:
                x = self.bn(x)
            x = self.relu(x)
            return x

    class CNNModel(Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.ishape = (1, 1, 28, 28)
            self.quant_inp = qnn.QuantIdentity(
                bit_width=input_bitwidth,
                scaling_impl_type='const',
                scaling_init=2**(input_bitwidth) - 1,
                signed=False
            )
            self.conv0 = ConvBlock(input_ch=1, output_ch=8)  # 1 * 28 * 28
            self.conv1 = ConvBlock(input_ch=8, output_ch=8)  # 8 * 13 * 13
            self.dense0 = DenseBlock(in_features=8 * 5 * 5, out_features=256)  # 16 * 5 * 5
            self.dense1 = DenseBlock(in_features=256, out_features=10)

        def forward(self, x):
            x = self.quant_inp(x)
            x = self.conv0(x)
            x = self.conv1(x)
            x = torch.flatten(x, start_dim=1)
            x = self.dense0(x)
            x = self.dense1(x)
            return x

    model = CNNModel()
    return model
