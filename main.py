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

gen_in_dict = {
    "in_features": (2,3,),
    "out_features": (3,),
    "bias":  (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}


def get_work_dir(keys, values, base="/circuits/"):
    str_list = list(map(lambda kv: f"{kv[0]}{str(kv[1])}", zip(keys, values)))
    specific_dir = functools.reduce(lambda a,b: a + "_" + b, str_list)
    return os.getcwd() + base + specific_dir

def run_test(*args):
    work_dir = get_work_dir(gen_in_dict.keys(), args[0])
    brevitas_model, data = get_linear_layer_model(*args[0])
    test_model(brevitas_model, data, work_dir, os.getcwd())
    

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
    feat_list = list(itertools.product(*gen_in_dict.values()))
    with ThreadPool(args.num_workers) as pool:
        pool.map(run_test, feat_list)

