import os
import time
import onnx
import qonnx
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
import numpy as np
import subprocess
import hls4ml
from chisel4ml import transform
from chisel4ml import generate
from server import create_server
from parse_reports import parse_reports


def test_hls4ml(qonnx_model, work_dir, base_dir):
    starttime = time.perf_counter()
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean())
    qonnx_model = qonnx_model.transform(GemmToMatMul())
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    hls_config = hls4ml.utils.config_from_onnx_model(
        qonnx_model,
        granularity='name',
        backend="Vitis",
        default_reuse_factor=1,
    )
    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Unrolled'
    for key in hls_config['LayerName'].keys():
        if "conv" in key.lower():
            hls_config['LayerName'][key]['ParallelizationFactor'] = 9999999
            print(f"Setting maximum parallelization in layer {key}.")
    hls_model = hls4ml.converters.convert_from_onnx_model(
        qonnx_model,
        output_dir=work_dir,
        backend="Vitis",
        io_type="io_parallel",
        hls_config=hls_config,
        part='xcvu9p-flga2104-2L-e'
    )
    hls_model.compile()
    hls_model.build(csim=False, synth=True, cosim=True, vsynth=True)
    #commands = [
    #    "vivado",
    #    "-mode", "batch",
    #    "-source", f"{base_dir}/synth_hls.tcl",
    #    "-nojournal",
    #    "-nolog",
    #    "-tclargs", work_dir, base_dir
    #]
    #with open(f"{work_dir}/vivado.log", 'w') as log_file:
    #    cp = subprocess.run(commands, stdout=log_file, stderr=log_file)
    assert cp.returncode == 0
    duration = time.perf_counter() - starttime
    with open(f"{work_dir}/time.log", 'w') as time_file:
        time_file.write("HLS4ML SYNTHESIS TIME:\n")
        time_file.write(f"{str(duration)}\n")

