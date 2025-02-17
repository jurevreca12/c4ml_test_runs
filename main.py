import os
import itertools
import functools
from multiprocessing.pool import ThreadPool
from models.linear_model import get_linear_layer_model
from models.conv_model import get_conv_layer_model
from models.maxpool_model import get_maxpool_layer_model
from models.train import train_quantized_mnist_model
from test_model import test_model
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

##############################
# LINEAR LAYER EXPERIMENTS   #
##############################
linear_layer_var_in_features_exp = {
    "in_features": (16, 32, 64, 128, 256, 512),
    "out_features": (32,),
    "bias": (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
linear_layer_var_out_features_exp = {
    "in_features": (16,),
    "out_features": (16, 32, 64, 128, 256, 512),
    "bias": (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
linear_layer_var_iq_exp = {
    "in_features": (32,),
    "out_features": (32,),
    "bias": (True,),
    "iq": (2, 3, 4, 5, 6, 7),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
linear_layer_var_wq_exp = {
    "in_features": (32,),
    "out_features": (32,),
    "bias": (True,),
    "iq": (4,),
    "wq": (2, 3, 4, 5, 6, 7),
    "bq": (8,),
    "oq": (4,)
}


##############################
# CONV LAYER EXPERIMENTS     #
##############################
conv_layer_var_input_ch_exp = {
    "input_size": ((8, 8),),
    "input_ch": (1, 2, 4, 8, 16),
    "output_ch": (1,),
    "kernel_size": ((3, 3),),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
conv_layer_var_output_ch_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1, 2, 4, 8, 16),
    "kernel_size": ((3, 3),),
    "iq": (4,), 
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
conv_layer_var_iq_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1,),
    "kernel_size": ((3, 3),),
    "iq": (2, 3, 4, 5, 6, 7),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}
conv_layer_var_wq_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1,),
    "kernel_size": ((3, 3),),
    "iq": (4,),
    "wq": (2, 3, 4, 5, 6, 7),
    "bq": (8,),
    "oq": (4,)
}


##############################
#  MAXPOOL LAYER EXPERIMENTS #
##############################
maxpool_layer_var_input_size_exp = {
    "channels": (3,),
    "input_size": ((4, 4), (8, 8), (12, 12), (16, 16)),
    "kernel_size": ((3, 3),),
    "iq": (4,)
}
maxpool_layer_var_channels_exp = {
    "channels": (1, 2, 4, 8, 16),
    "input_size": ((8, 8),),
    "kernel_size": ((3, 3),),
    "iq": (4,)
}
maxpool_layer_var_kernel_size_exp = {
    "channels": (3,),
    "input_size": ((8, 8),),
    "kernel_size": ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)),
    "iq": (4,)
}
maxpool_layer_var_iq_exp = {
    "channels": (3,),
    "input_size": ((8, 8),),
    "kernel_size": ((2, 2),),
    "iq": (2, 3, 4, 5, 6, 7)
}

##############################
#  CNN MODEL EXPERIMENTS     #
##############################
cnn_mnist_model_var_bitwidth_exp = {
    "bitwidth": (2, 3, 4, 5, 6, 7),
    "prune_rate": (0.5,)
}

cnn_mnist_model_var_prune_rate_exp = {
    "bitwidth": (4,),
    "prune_rate": (0.5, 0.8, 0.85, 0.88, 0.9)
}

EXPERIMENTS = (
    (linear_layer_var_in_features_exp, get_linear_layer_model, "linear_layer_var_in_features_exp", "NeuronProcessingUnit"),  # 0
    (linear_layer_var_out_features_exp, get_linear_layer_model, "linear_layer_var_out_features_exp", "NeuronProcessingUnit"),  # 1
    (linear_layer_var_iq_exp, get_linear_layer_model, "linear_layer_var_iq_exp", "NeuronProcessingUnit"),  # 2
    (linear_layer_var_wq_exp, get_linear_layer_model, "linear_layer_var_wq_exp", "NeuronProcessingUnit"),  # 3

    (conv_layer_var_input_ch_exp, get_conv_layer_model, "conv_layer_var_input_ch_exp", "NeuronProcessingUnit"),  # 4
    (conv_layer_var_output_ch_exp, get_conv_layer_model, "conv_layer_var_output_ch_exp", "NeuronProcessingUnit"),  # 5
    (conv_layer_var_iq_exp, get_conv_layer_model, "conv_layer_var_iq_exp", "NeuronProcessingUnit"),  # 6
    (conv_layer_var_wq_exp, get_conv_layer_model, "conv_layer_var_wq_exp", "NeuronProcessingUnit"),  # 7

    (maxpool_layer_var_input_size_exp, get_maxpool_layer_model, "maxpool_layer_var_input_size_exp", "OrderProcessingUnit"),  # 8
    (maxpool_layer_var_channels_exp, get_maxpool_layer_model, "maxpool_layer_var_channels_exp", "OrderProcessingUnit"),  # 9
    (maxpool_layer_var_kernel_size_exp, get_maxpool_layer_model, "maxpool_layer_var_kernel_size_exp", "OrderProcessingUnit"),  # 10
    (maxpool_layer_var_iq_exp, get_maxpool_layer_model, "maxpool_layer_var_iq_exp", "OrderProcessingUnit"),  # 11

    (cnn_mnist_model_var_bitwidth_exp, train_quantized_mnist_model, "cnn_mnist_model_var_bitwidth_exp", "ProcessingPipeline"),  # 12
    (cnn_mnist_model_var_prune_rate_exp, train_quantized_mnist_model, "cnn_mnist_model_var_prune_rate_exp", "ProcessingPipeline")  # 13
)
current_exp = 0


def get_exp_by_name(name):
    for ind, exp in enumerate(EXPERIMENTS):
        if name == exp[2]:
            return EXPERIMENTS[ind]
    raise ValueError(f"Key:{name} not found in experiments list.")


def get_work_dir(keys, values, base):
    def to_string(kv):
        key, vals = kv
        if isinstance(vals, (tuple, list)):
            val_str = ""
            for val in vals:
                val_str += str(val) + "_"
            val_str = val_str[:-1]
            return f"{key}{val_str}"
        else:
            return f"{key}{vals}"
    str_list = list(map(lambda kv: to_string(kv), zip(keys, values)))
    specific_dir = functools.reduce(lambda a, b: a + "_" + b, str_list)
    return SCRIPT_DIR + base + specific_dir


def run_test(*args):
    global current_exp
    exp_dict = EXPERIMENTS[current_exp][0]
    model_gen = EXPERIMENTS[current_exp][1]
    exp_name = EXPERIMENTS[current_exp][2]
    top_name = EXPERIMENTS[current_exp][3]
    work_dir = get_work_dir(exp_dict.keys(), args[0], base=f"/circuits/{exp_name}/")
    print(f"Starting work in: {work_dir}")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    hls4ml_exists = os.path.exists(f"{work_dir}/c4ml/utilization.rpt") 
    c4ml_exists = os.path.exists(f"{work_dir}/hls4ml/utilization.rpt")
    if hls4ml_exists and c4ml_exists:
        return
    brevitas_model, data, acc = model_gen(*args[0])
    if acc is not None:
        with open(f"{work_dir}/acc.log", 'w') as f:
            f.write(f"Final accuracy-{args[0]}:\n")
            f.write(f"{str(acc)}\n")
    return test_model(brevitas_model, data, work_dir, SCRIPT_DIR, top_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='c4ml_test_runs')
    parser.add_argument(
        '--num_workers',
        '-n',
        type=int,
        default=2,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--experiment_num',
        '-exp',
        type=int,
        default=-1,
        help='If set, run only the given experiment (else run all).'
    )
    parser.add_argument(
        '--experiment_name',
        '-name',
        default="",
        help='Name of the experiment to run.'
    )
    args = parser.parse_args()
    if args.experiment_name != "":
        exp = get_exp_by_name(args.experiment_name)
        EXPERIMENTS = (exp,)
    if args.experiment_num >= 0:
        EXPERIMENTS = (EXPERIMENTS[args.experiment_num],)

    if not os.path.exists(f'{SCRIPT_DIR}/results'):
        os.makedirs(f'{SCRIPT_DIR}/results')
    for exp in EXPERIMENTS:
        print(f"Running {exp[2]}")
        feat_list = list(itertools.product(*exp[0].values()))
        with ThreadPool(args.num_workers) as pool:
            pool.map(run_test, feat_list)
        current_exp += 1
        print(f"Finnished {exp[2]}")
