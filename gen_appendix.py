ROOT = "figures/plots/"
EXPERIMENTS = {
    r"\subsection{Fully Connected Layer - Input Features}":  {
        "text": "Single fully connected layer experiment in which we vary the number of input features. The number of output features equals 32. The input, output and weights are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "linear_layer_var_in_features_exp",
    },
    r"\subsection{Fully Connected Layer - Output Features}": {
        "text": "Single fully connected layer experiment in which we vary the number of output features. The number of input features equals 16. The input, output and weights are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "linear_layer_var_out_features_exp",
    },
    r"\subsection{Fully Connected Layer - Input Bitwidth}":  {
        "text": "Single fully connected layer experiment in which we vary the number of bits used to quantize the input features. The number of input and output features equals 32. The weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "linear_layer_var_iq_exp",
    },
    r"\subsection{Fully Connected Layer - Weight Bitwidth}": {
        "text": "Single fully connected layer experiment in which we vary the number of bits used to quantize the weight parameters. The number of input and output features equals 32. The inputs and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "linear_layer_var_wq_exp",
    },
    r"\subsection{Convolution Layer - Input Channels}":  {
        "text": "Single convolution layer experiment in which we vary the number of input channels of the convolution layer. The input size is set to an 8 by 8 square and the kernel size is set to 3 by 3. The number of output channels is set to 1. The inputs, weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "conv_layer_var_input_ch_exp",
    },
    r"\subsection{Convolution Layer - Output Channels}": {
        "text": "Single convolution layer experiment in which we vary the number of output channels of the convolution layer. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input channels is set to 1. The inputs, weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "conv_layer_var_output_ch_exp",
    },
    r"\subsection{Convolution Layer - Input Bitwidth}": {
        "text": "Single convolution layer experiment in which we vary the number of bits used to quantize the input. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input and output channels is set to 1. The weights and outputs are quantized to 4 bits and the bias is quantized to 8 bits.",
        "plot": "conv_layer_var_iq_exp",
    },
    r"\subsection{Convolution Layer - Weight Bitwidth}": {
        "text": "Single convolution layer experiment in which we vary the number of bits used to quantize the weight parameters. The input size is set to an 16 by 16 square and the kernel size is set to 3 by 3. The number of input and output channels is set to 1. The inputs and outputs are quantized to 4 bits, and the bias is quantized to  8 bits.",
        "plot": "conv_layer_var_wq_exp",
    },
    r"\subsection{Maximum Pooling Layer - Input Window Size}": {
        "text": "Single maximum pooling layer experiment in which we vary the input size whilst keeping it square. The size of the kernel is 3 by 3 and the number of channels is set to 3. The inputs to the layer are quantized to 4 bits.",
        "plot": "maxpool_layer_var_input_size_exp",
    },
    r"\subsection{Maximum Pooling Layer - Channels}": {
        "text": "Single maximum pooling layer experiment in which we vary the number of channels of the input tensor. The input size is set to an 8 by 8 square and the kernel size is set to 3 by 3. The inputs to the layer are quantized to 4 bits.",
        "plot": "maxpool_layer_var_channels_exp",
    },
    r"\subsection{Maximum Pooling Layer - Kernel Size}": {
        "text": "Single maximum pooling layer experiment in which we vary the size of the square kernel. The number of channels is set to 3 and input size to an 8 by 8 square. The inputs to the layer are quantized to 4 bits.",
        "plot": "maxpool_layer_var_kernel_size_exp",
    },
    r"\subsection{Maximum Pooling Layer - Input Bitwidth}": {
        "text": "Single maximum pooling layer experiment in which we vary the number of bits used to quantize the input. The number of channels is set to 3 and the input size to an 8 by 8 square. The kernel size is set to 2 by 2.",
        "plot": "maxpool_layer_var_iq_exp"
    },
    r"\subsection{Jet Tagging MLP - Bitwidth}": {
        "text": "The multilayer perceptron experiment in which we vary the number of bits used to quantize the weights and activations. The inputs to the MLP are quantized to 8 bits. The global pruning rate is set to 0.5.",
        "plot": "lhc_model_var_bitwidth_exp",
    },
    r"\subsection{Jet Tagging MLP - Prune Rate}": {
        "text": "The multilayer perceptron experiment in which we vary the global pruning rate. The weights and activations are quantized to 4 bits. The inputs to the MLP are quantized to constant 8 bits.",
        "plot": "lhc_model_var_prune_rate_exp",
    },
    r"\subsection{CNN - Bitwidth}":{
        "text": "The convolutional neural network experiment in which we vary the number of bits used to quantize the weights and activations. The inputs to the CNN are quantized to constant 8 bits. The global pruning rate is set to 0.5.",
        "plot": "cnn_mnist_model_var_bitwidth_exp",
    },
    r"\subsection{CNN - Prune Rate}": {
        "text": "The convoultional neural network experiment in which we vary the global pruning rate. The weights and activations are quantized to 3 bits. The inputs to the CNN are quantized to constant 8 bits.",
        "plot": "cnn_mnist_model_var_prune_rate_exp",
    },
}
# 
# IMAGES = ("bram_b18_plot.pdf", "bram_b36_plot.pdf", "bram_tile_plot.pdf",
#IMAGES = ("bram_tile_plot.pdf",
#          "dsp_plot.pdf", "ff_plot.pdf", "init_interval_plot.pdf",
#          "latency_cycles_plot.pdf", "lut_plot.pdf", "path_delay_plot.pdf",
#          "peak_mem_usage_plot.pdf", "syn_time_plot.pdf", "throughput_plot.pdf",
#          "total_latency_plot.pdf", "uram_plot.pdf")
IMAGES = (
    "lut_plot.pdf",
    "ff_plot.pdf",
    "dsp_plot.pdf",
    "bram_tile_plot.pdf",
    "uram_plot.pdf",

    "init_interval_plot.pdf",
    "latency_cycles_plot.pdf",
    "path_delay_plot.pdf",
    "total_latency_plot.pdf",
    "throughput_plot.pdf",

    "peak_mem_usage_plot.pdf", 
    "syn_time_plot.pdf",
)

def handle_experiments(experiments):
    for key, val in experiments.items():
        if key == "text":
            print(val)
            continue
        if isinstance(val, dict):
            print(key)
            handle_experiments(val)
        else:
            generate_table(val)

def generate_table(val):
    print(r"\begin{table}[H]")
    print(r"\begin{tabular}{lll}")
    print(r"\centering")
    for idx, img in enumerate(IMAGES):
        sep = r"\\" if (idx + 1) % 3 == 0 else r"&"
        print(r"\includegraphics[width=.33\linewidth]{" + ROOT + val + r"/" + img + r"}" + sep)
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
        handle_experiments(EXPERIMENTS)
