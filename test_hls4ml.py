import os
import json
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
    os.system(f"cp {base_dir}/synth_hls.tcl {work_dir}/vivado_synth.tcl")
    ret = hls_model.build(csim=False, synth=True, cosim=True, vsynth=True)
    duration = time.perf_counter() - starttime
    ret['total_duration'] = duration
    ret['tool'] = 'hls4ml'
    with open(f"{work_dir}/info.json", 'w') as info_file:
        json.dump(ret, info_file)
