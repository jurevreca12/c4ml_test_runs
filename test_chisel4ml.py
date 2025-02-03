import os
import time
import multiprocessing
import torch
import onnx
import qonnx
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.core.onnx_exec import execute_onnx
import numpy as np
import subprocess
import hls4ml
from chisel4ml import transform
from chisel4ml import generate
from server import create_server
from parse_reports import parse_reports


def test_chisel4ml(qonnx_model, test_data, work_dir, base_dir, top_name):
    starttime = time.perf_counter()
    lbir_model = transform.qonnx_to_lbir(qonnx_model)
    accelerators = generate.accelerators(
        lbir_model,
        minimize="delay",
    )
    c4ml_server, c4ml_subp = create_server(f'{base_dir}/chisel4ml/out/chisel4ml/assembly.dest/out.jar')
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=True,
        gen_timeout_sec=9000,
        server=c4ml_server
    )
    for x in test_data:
        expanded_x = np.expand_dims(x, axis=0)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]
        hw_res = circuit(x)
        assert np.array_equal(qonnx_res.flatten(), qonnx_res.flatten())
    print("D")
    circuit.package(work_dir)
    c4ml_server.stop()
    c4ml_subp.terminate()
    commands = [
        "vivado",
        "-mode", "batch",
        "-source", f"{base_dir}/synth.tcl",
        "-nojournal",
        "-nolog",
        "-tclargs", work_dir, base_dir, top_name
    ]
    with open(f"{work_dir}/vivado.log", 'w') as log_file:
        cp = subprocess.run(commands, stdout=log_file, stderr=log_file)
    assert cp.returncode == 0
    duration = time.perf_counter() - starttime
    with open(f"{work_dir}/time.log", 'w') as time_file:
        time_file.write("CHISEL4ML SYNTHESIS TIME:\n")
        time_file.write(f"{str(duration)}\n")


