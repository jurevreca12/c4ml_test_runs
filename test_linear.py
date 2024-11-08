import itertools
import torch
import numpy as np
import chisel4ml
from chisel4ml import transform
from chisel4ml import generate
from linear_model import get_linear_layer_model
from server import get_server, create_server

def test_linear_in(in_features, out_features, bias, iq, wq, bq, oq):
    brevitas_model, data = get_linear_layer_model(
        in_features = in_features,
        out_features = out_features,
        bias = bias,
        iq = iq,
        wq = wq,
        bq = bq,
        oq = oq
    )
        
    qonnx_model = transform.brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    accelerators, lbir_model = generate.accelerators(
        qonnx_model,
        ishape=brevitas_model.ishape,
        minimize="delay"
    )
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=True,
        server=get_server()
    )
    for x in data:
        sw_res = (
            brevitas_model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.package(f"circuits/if{in_features}_of{out_features}_b{bias}_iq{iq}_wq{wq}_bq{bq}_oq{oq}/") 

if __name__ == "__main__":
    create_server('chisel4ml/out/chisel4ml/assembly.dest/out.jar')
    gen_in_dict = {
        "in_features": (64, 256, 1024),
        "out_features": (64,),
        "bias":  (True,),
        "iq": (4,),
        "wq": (4,),
        "bq": (8,),
        "oq": (4,)
    }
    feat_list = list(itertools.product(*gen_in_dict.values()))
    for feat in feat_list:
        test_linear_in(*feat)
