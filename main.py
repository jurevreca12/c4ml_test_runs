import os
import itertools
import functools
from multiprocessing.pool import ThreadPool
import torch
import numpy as np
import chisel4ml
from chisel4ml import transform
from chisel4ml import generate
from linear_model import get_linear_layer_model
from conv_model import get_conv_layer_model
from maxpool_model import get_maxpool_layer_model
from server import get_server, create_server
from test_model import test_model
import argparse
import json


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


##############################
## LINEAR LAYER EXPERIMENTS ##
##############################
linear_layer_var_in_features_exp = {
    "in_features": (16, 32, 64, 128, 256),
    "out_features": (32,),
    "bias":  (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

linear_layer_var_out_features_exp = {
    "in_features": (16,),
    "out_features": (16, 32, 64, 128, 256),
    "bias":  (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

linear_layer_var_iq_exp = {
    "in_features": (32,),
    "out_features": (32,),
    "bias":  (True,),
    "iq": (2, 3, 4, 5, 6, 7),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

linear_layer_var_wq_exp = {
    "in_features": (32,),
    "out_features": (32,),
    "bias":  (True,),
    "iq": (4,),
    "wq": (2, 3, 4, 5, 6, 7),
    "bq": (8,),
    "oq": (4,)
}


##############################
## CONV LAYER EXPERIMENTS   ##
##############################
conv_layer_var_input_ch_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1, 2, 4, 8, 16),
    "output_ch": (1,),
    "kernel_size": ((3,3),),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

conv_layer_var_output_ch_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1, 2, 4, 8, 16),
    "kernel_size": ((3,3),),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

dw_conv_layer_var_output_ch_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1, 2, 4, 8, 16),
    "kernel_size": ((3,3),),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,),
    "depthwise": (True,)
}

conv_layer_var_iq_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1,),
    "kernel_size": ((3,3),),
    "iq": (2,3,4,5,6,7),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

conv_layer_var_wq_exp = {
    "input_size": ((16, 16),),
    "input_ch": (1,),
    "output_ch": (1,),
    "kernel_size": ((3,3),),
    "iq": (4,),
    "wq": (2,3,4,5,6,7),
    "bq": (8,),
    "oq": (4,)
}


##############################
## MAXPOOL LAYER EXPERIMENTS #
##############################
maxpool_layer_var_input_size = {
    "channels": (3,),
    "input_size": ((4,4), (8,8), (12,12), (16,16)),
    "kernel_size": ((3, 3),),
    "iq": (4,)
}

maxpool_layer_var_channels = {
    "channels": (1, 2, 4, 8, 16),
    "input_size": ((8,8),),
    "kernel_size": ((3, 3),),
    "iq": (4,)
}

maxpool_layer_var_kernel_size = {
    "channels": (3,),
    "input_size": ((8,8),),
    "kernel_size": ((2, 2), (3, 3), (4,4), (5,5)),
    "iq": (4,)
}

experiments = (
    (linear_layer_var_in_features_exp, get_linear_layer_model, "linear_layer_var_in_features", "NeuronProcessingUnit"), # 0
    (linear_layer_var_out_features_exp, get_linear_layer_model, "linear_layer_var_out_features", "NeuronProcessingUnit"), # 1
    (linear_layer_var_iq_exp, get_linear_layer_model, "linear_layer_var_iq", "NeuronProcessingUnit"), # 2
    (linear_layer_var_wq_exp, get_linear_layer_model, "linear_layer_var_wq", "NeuronProcessingUnit"), # 3

    (conv_layer_var_input_ch_exp, get_conv_layer_model, "conv_layer_var_input_ch", "NeuronProcessingUnit"), # 4
    (conv_layer_var_output_ch_exp, get_conv_layer_model, "conv_layer_var_output_ch", "NeuronProcessingUnit"), # 5
    (dw_conv_layer_var_output_ch_exp, get_conv_layer_model, "dw_conv_layer_var_output_ch", "NeuronProcessingUnit"), # 6
    (conv_layer_var_iq_exp, get_conv_layer_model, "conv_layer_var_iq", "NeuronProcessingUnit"), # 7
    (conv_layer_var_wq_exp, get_conv_layer_model, "conv_layer_var_wq", "NeuronProcessingUnit"), # 8
    
    (maxpool_layer_var_input_size, get_maxpool_layer_model, "maxpool_layer_var_input_size", "OrderProcessingUnit"), # 9
    (maxpool_layer_var_channels, get_maxpool_layer_model, "maxpool_layer_var_channels", "OrderProcessingUnit"), # 10
    (maxpool_layer_var_kernel_size, get_maxpool_layer_model, "maxpool_layer_var_kernel_size", "OrderProcessingUnit"), # 11
)
current_exp = 0


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
    specific_dir = functools.reduce(lambda a,b: a + "_" + b, str_list)
    return SCRIPT_DIR + base + specific_dir

def run_test(*args):
    global current_exp
    exp_dict = experiments[current_exp][0]
    model_gen = experiments[current_exp][1]
    exp_name = experiments[current_exp][2]
    top_name = experiments[current_exp][3]
    work_dir = get_work_dir(exp_dict.keys(), args[0], base=f"/circuits/{exp_name}/")
    brevitas_model, data = model_gen(*args[0])
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
    args = parser.parse_args()
    if args.experiment_num >= 0:
        experiments = (experiments[args.experiment_num], )

    create_server('chisel4ml/out/chisel4ml/assembly.dest/out.jar', args.num_workers)
    results = []
    for exp in experiments:
        print(f"Running {exp[2]}")
        feat_list = list(itertools.product(*exp[0].values()))
        with ThreadPool(args.num_workers) as pool:
            exp_results = pool.map(run_test, feat_list)
        current_exp += 1
        results.append(exp_results)

        ser_res = json.dumps(results)
        with open(f'{SCRIPT_DIR}/results_exp_{exp[2]}.json', 'w') as f:
            f.write(ser_res)
        print(f"Finnished {exp[2]}")
