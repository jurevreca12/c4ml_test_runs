import os
import time
import multiprocessing
import torch
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


def test_chisel4ml(qonnx_model, brevitas_model, test_data, work_dir, base_dir, top_name):
    starttime = time.perf_counter()
    accelerators, lbir_model = generate.accelerators(
        qonnx_model,
        ishape=brevitas_model.ishape,
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
        sw_res = (
            brevitas_model.forward(torch.from_numpy(np.expand_dims(x, axis=0))).detach().numpy()
        )
        hw_res = circuit(x)
        assert np.array_equal(sw_res.flatten(), hw_res.flatten())
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
    hls_config['Model']['Strategy'] = 'Latency'
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
    hls_model.build(csim=False)
    commands = [
        "vivado",
        "-mode", "batch",
        "-source", f"{base_dir}/synth_hls.tcl",
        "-nojournal",
        "-nolog",
        "-tclargs", work_dir, base_dir
    ]
    with open(f"{work_dir}/vivado.log", 'w') as log_file:
        cp = subprocess.run(commands, stdout=log_file, stderr=log_file)
    assert cp.returncode == 0
    duration = time.perf_counter() - starttime
    with open(f"{work_dir}/time.log", 'w') as time_file:
        time_file.write("HLS4ML SYNTHESIS TIME:\n")
        time_file.write(f"{str(duration)}\n")


lock = multiprocessing.Lock()
def test_model(brevitas_model, test_data, work_dir, base_dir, top_name):
    global lock
    lock.acquire()
    qonnx_model = transform.brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    lock.release()
    if not os.path.exists(f"{work_dir}/qonnx"):
        os.makedirs(f"{work_dir}/qonnx")
    onnx.save(qonnx_model.model, f"{work_dir}/qonnx/model.onnx")
    if not os.path.exists(f"{work_dir}/c4ml/utilization.rpt"):
        print(f"Starting {work_dir}/c4ml run!")
        test_chisel4ml(qonnx_model, brevitas_model, test_data, f"{work_dir}/c4ml/", base_dir, top_name)
    else:
        print(f"Skipping {work_dir}/c4ml run. Already Exists!")
    if not os.path.exists(f"{work_dir}/hls4ml/utilization.rpt"):
        print(f"Starting {work_dir}/hls4ml run!")
        test_hls4ml(qonnx_model, f"{work_dir}/hls4ml/", base_dir)
    else:
        print(f"Skipping {work_dir}/hls4ml run. Already Exists!")
