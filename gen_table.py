import pandas as pd
import numpy as np
import math

EXPERIMENTS = {
	"linear_layer_var_in_features_exp": "Single fully connected layer experiment in which we vary the number of input features. The number of output features equals 32. The input, output and weights are quantized to 4 bits and the bias is quantized to 8 bits.",
	"linear_layer_var_out_features_exp": "Single fully connected layer experiment in which we vary the number of output features. The number of input features equals 16. The input, output and weights are quantized to 4 bits and the bias is quantized to 8 bits. ",
	"linear_layer_var_iq_exp": "Single fully connected layer experiment in which we vary the number of bits used to quantize the input features. The number of input and output features equals 32. The weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
	"linear_layer_var_wq_exp": "Single fully connected layer experiment in which we vary the number of bits used to quantize the weight parameters. The number of input and output features equals 32. The inputs and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",

	"conv_layer_var_input_ch_exp": "Single convolution layer experiment in which we vary the number of input channels of the convolution layer. The input size is set to an 8 by 8 square and the kernel size is set to 3 by 3. The number of output channels is set to 1. The inputs, weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
	"conv_layer_var_output_ch_exp": "Single convolution layer experiment in which we vary the number of output channels of the convolution layer. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input channels is set to 1. The inputs, weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
	"conv_layer_var_iq_exp": "Single convolution layer experiment in which we vary the number of bits used to quantize the input. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input and output channels is set to 1. The weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
	"conv_layer_var_wq_exp": "Single convolution layer experiment in which we vary the number of bits used to quantize the weight parameters. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input and output channels is set to 1. The inputs and outputs are quantized to 4 bits, and the bias is quantized to  8 bits.",
	"maxpool_layer_var_input_size_exp": "Single maximum pooling layer experiment in which we vary the input size whilst keeping it square. The size of the kernel is 3 by 3 and the number of channels is set to 3. The inputs to the layer are quantized to 4 bits.",
	"maxpool_layer_var_channels_exp": "Single maximum pooling layer experiment in which we vary the number of channels of the input tensor. The input size is set to an 8 by 8 square and the kernel size is set to 3 by 3. The inputs to the layer are quantized to 4 bits.",
	"maxpool_layer_var_kernel_size_exp": "Single maximum pooling layer experiment in which we vary the size of the square kernel. The number of channels is set to 3 and input size to an 8 by 8 square. The inputs to the layer are quantized to 4 bits.",
	"maxpool_layer_var_iq_exp": "Single maximum pooling layer experiment in which we vary the number of bits used to quantize the input. The number of channels is set to 3 and the input size to an 8 by 8 square. The kernel size is set to 2 by 2.",


	"lhc_model_var_bitwidth_exp" : "The multilayer perceptron experiment in which we vary the number of bits used to quantize the weights and activations. The inputs to the MLP are quantized to 8 bits. The global pruning rate is set to 0.5.",
	"lhc_model_var_prune_rate_exp" : "The multilayer perceptron experiment in which we vary the global pruning rate. The weights and activations are quantized to 4 bits. The inputs to the MLP are quantized to constant 8 bits.",

	"cnn_mnist_model_var_bitwidth_exp": "The convolutional neural network experiment in which we vary the number of bits used to quantize the weights and activations. The inputs to the CNN are quantized to constant 8 bits. The global pruning rate is set to 0.5.",
	"cnn_mnist_model_var_prune_rate_exp": "The convoultional neural network experiment in which we vary the global pruning rate. The weights and activations are quantized to 3 bits. The inputs to the CNN are quantized to constant 8 bits.",
}

def stringfy_int(x: np.float64) -> str:
    if math.isnan(x):
        return 'NaN'
    else:
        return str(int(x))

INT_COLUMNS = (
    'Look-Up Tables',
    'Flip-Flops',
    'Block RAM Tile',
    'Block RAM 36kb',
    'Block RAM 16kb',
    'UltraRAM',
    'DSP Block',
    'Initiation Interval',
    'Latency Cycles',
    'Throughput [Hz]'
)
COLUMN_RENAME_DICT = {
    'Bitwidth': 'BW',
    'tool': 'Tool',
    'Generation Time [hours]': 'Time',
    'Look-Up Tables': 'LUT',
    'Flip-Flops': 'FF',
    'Block RAM Tile': 'BRAM',
    'Block RAM 36kb': 'BRAM36',
    'Block RAM 16kb': 'BRAM16',
    'UltraRAM': 'URAM',
    'DSP Block': 'DSP',
    'Path Delay [ns]': 'PD',
    'Peak Memory [MiB]': 'PM',
    'Initiation Interval': 'II',
    'Throughput [Hz]': 'TP',
    'Latency Cycles': 'LC',
    'Total Latency [ns]': 'TL',
    'Prune Rate': 'Pruning Rate',
}

def traverse_exp(experiments):
    for key, val in experiments.items():
        gen_tex_table(key, val)


def gen_tex_table(exp, desc):
    df = pd.read_csv(f'plots/{exp}/{exp}.csv').iloc[:,1:]
    for col in INT_COLUMNS:
        df[col] = df[col].map(stringfy_int)

    df = df.rename(columns=COLUMN_RENAME_DICT)
    tex = df.to_latex(
        index=False,
        float_format="%.2f",
    )
    print(r"\begin{landscape}")
    print(r"\begin{table}[H]")
    print(r"\footnotesize")
    print(tex)
    print(r"\caption{" + desc + r"}")
    print(r"\end{table}")
    print(r"\end{landscape}")

if __name__ == '__main__':
    traverse_exp(EXPERIMENTS)


