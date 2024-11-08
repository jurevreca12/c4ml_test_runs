import os
import time
import itertools
import torch
import onnx
import qonnx
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
import numpy as np
import subprocess
import hls4ml
import chisel4ml
from chisel4ml import transform
from chisel4ml import generate
from linear_model import get_linear_layer_model
from server import get_server, create_server


def test_chisel4ml(qonnx_model, brevitas_model, test_data, work_dir):
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
    circuit.package(work_dir)
    commands = [
        "vivado", 
        "-mode", "batch", 
        "-source", "synth.tcl", 
        "-nojournal",
        "-nolog",
        "-tclargs", work_dir
    ]
    starttime = time.time()
    with open(f"{work_dir}/vivado.log", 'w') as log_file:
        cp = subprocess.run(commands, stdout=log_file, stderr=log_file)
    duration = time.time() - starttime
    assert cp.returncode == 0
    with open(f"{work_dir}/time.log", 'w') as time_file:
        time_file.write(f"CHISEL4ML SYNTHESIS TIME:\n")
        time_file.write(f"{str(duration)}\n")


def test_hls4ml(qonnx_model, work_dir):
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean())
    qonnx_model = qonnx_model.transform(GemmToMatMul())
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    hls_config = hls4ml.utils.config_from_onnx_model(
        qonnx_model,
        backend="Vitis",
        default_reuse_factor=1,
    )
    hls_config['Model']['ReuseFactor'] = 1
    hls_model = hls4ml.converters.convert_from_onnx_model(
        qonnx_model,
        output_dir=work_dir,
        backend="Vitis",
        io_type="io_parallel",
        hls_config=hls_config,
        part='xcvu9p-flga2104-2L-e'
    )
    hls_model.compile()
    starttime = time.time()
    hls_model.build(csim=False, vsynth=True)
    hls4ml.report.read_vivado_report('hls4ml')
    duration = time.time() - starttime
    with open(f"{work_dir}/time.log", 'w') as time_file:
        time_file.write(f"HLS4ML SYNTHESIS TIME:\n")
        time_file.write(f"{str(duration)}\n")
                        

def test_model(brevitas_model, test_data, work_dir): 
    qonnx_model = transform.brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    os.makedirs(f"{work_dir}/qonnx")
    onnx.save(qonnx_model.model, f"{work_dir}/qonnx/model.onnx")
    test_chisel4ml(qonnx_model, brevitas_model, test_data, f"{work_dir}/c4ml/")
    test_hls4ml(qonnx_model, f"{work_dir}/hls4ml/")
