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
from server import get_server, create_server
from test_model import test_model
import argparse
import json


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

linear_layer_var_in_features_exp = {
    "in_features": (2,3,),
    "out_features": (3,),
    "bias":  (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

experiments = (
    (linear_layer_var_in_features_exp, get_linear_layer_model, "linear_layer_var_in_features"),
)
current_exp = 0


def get_work_dir(keys, values, base):
    str_list = list(map(lambda kv: f"{kv[0]}{str(kv[1])}", zip(keys, values)))
    specific_dir = functools.reduce(lambda a,b: a + "_" + b, str_list)
    return SCRIPT_DIR + base + specific_dir

def run_test(*args):
    global current_exp
    exp_dict = experiments[current_exp][0]
    model_gen = experiments[current_exp][1]
    exp_name = experiments[current_exp][2]
    work_dir = get_work_dir(exp_dict.keys(), args[0], base=f"/circuits/{exp_name}/")
    brevitas_model, data = model_gen(*args[0])
    return test_model(brevitas_model, data, work_dir, os.getcwd())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='c4ml_test_runs')
    parser.add_argument(
        '--num_workers',
        '-n',
        type=int,
        default=2,
        help='Number of parallel workers'
    )
    args = parser.parse_args()
    create_server('chisel4ml/out/chisel4ml/assembly.dest/out.jar', args.num_workers)
    results = []
    for exp in experiments:
        feat_list = list(itertools.product(*exp[0].values()))
        with ThreadPool(args.num_workers) as pool:
            exp_results = pool.map(run_test, feat_list)
        current_exp += 1
        results.append(exp_results)

    ser_res = json.dumps(results)
    with open(f'{SCRIPT_DIR}/results.json', 'w') as f:
        f.write(ser_res)
