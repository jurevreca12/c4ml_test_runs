import os
import json
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
    start_time = time.perf_counter()
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
    compile_time = time.perf_counter()
    for x in test_data:
        expanded_x = np.expand_dims(x, axis=0)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]
        hw_res = circuit(x)
        assert np.array_equal(qonnx_res.flatten(), qonnx_res.flatten())
    simulation_time = time.perf_counter()
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
    synthesis_time = time.perf_counter()
    assert cp.returncode == 0
    info_dict = {}
    info_dict['compile_duration'] = compile_time - start_time
    info_dict['simulation_duration'] = simulation_time - compile_time
    info_dict['synthesis_duration'] = synthesis_time - simulation_time
    info_dict['total_duration'] = synthesis_time - start_time
    info_dict['exact_latency'] = len(lbir_model.layers) + 1 # C4ML injects a stage for every lbir layer
    info_dict['tool'] = 'chisel4ml'
    with open(f"{work_dir}/info.json", 'w') as info_file:
        json.dump(info_dict, info_file)
