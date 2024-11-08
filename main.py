import itertools
import functools
import torch
import numpy as np
import chisel4ml
from chisel4ml import transform
from chisel4ml import generate
from linear_model import get_linear_layer_model
from server import get_server, create_server
from test_model import test_model

gen_in_dict = {
    "in_features": (2,),
    "out_features": (3,),
    "bias":  (True,),
    "iq": (4,),
    "wq": (4,),
    "bq": (8,),
    "oq": (4,)
}

def get_work_dir(keys, values, base="circuits/"):
    str_list = list(map(lambda kv: f"{kv[0]}{str(kv[1])}", zip(keys, values)))
    specific_dir = functools.reduce(lambda a,b: a + "_" + b, str_list)
    return base + specific_dir

if __name__ == "__main__":
    create_server('chisel4ml/out/chisel4ml/assembly.dest/out.jar')

    feat_list = list(itertools.product(*gen_in_dict.values()))
    for feat in feat_list:
        work_dir = get_work_dir(gen_in_dict.keys(), feat)
        brevitas_model, data = get_linear_layer_model(*feat)
        test_model(brevitas_model, data, work_dir)
