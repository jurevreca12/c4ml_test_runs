import os
import itertools
import torch
import onnx
import numpy as np
import chisel4ml
from chisel4ml import transform
from chisel4ml import generate
from linear_model import get_linear_layer_model
from server import get_server, create_server

def test_model(brevitas_model, test_data, work_dir): 
    qonnx_model = transform.brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    os.makedirs(f"{work_dir}/qonnx")
    onnx.save(qonnx_model.model, f"{work_dir}/qonnx/model.onnx")
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
    for x in test_data:
        sw_res = (
            brevitas_model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
    circuit.package(f"{work_dir}/c4ml/verilog")
