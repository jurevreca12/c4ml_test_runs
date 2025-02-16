import os
import json
import time
from qonnx.core.onnx_exec import execute_onnx
import numpy as np
import subprocess
from chisel4ml import transform
from chisel4ml import generate
from server import create_server
from memory_profiler import ProcessContainer
from threading import Thread


def test_chisel4ml(qonnx_model, test_data, work_dir, base_dir, top_name):
    curr_dir = os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    start_time = time.perf_counter()
    lbir_model = transform.qonnx_to_lbir(qonnx_model)
    accelerators = generate.accelerators(
        lbir_model,
        minimize="delay",
    )
    c4ml_server, c4ml_subp = create_server("/c4ml/chisel4ml.jar")
    mem_prof = ProcessContainer(pid=c4ml_subp.pid)
    thread_handle = mem_prof.profile()
    circuit = generate.circuit(
        accelerators,
        lbir_model,
        use_verilator=True,
        gen_timeout_sec=9000,
        server=c4ml_server,
    )
    compile_time = time.perf_counter()
    for x in test_data:
        expanded_x = np.expand_dims(x, axis=0)
        input_name = qonnx_model.model.graph.input[0].name
        qonnx_res = execute_onnx(qonnx_model, {input_name: expanded_x})
        qonnx_res = qonnx_res[list(qonnx_res.keys())[0]]
        hw_res = circuit(x)
        assert np.array_equal(qonnx_res.flatten(), hw_res.flatten())
    simulation_time = time.perf_counter()
    circuit.package(work_dir)
    mem_prof.stop()
    thread_handle.join()
    c4ml_server.stop()
    c4ml_subp.terminate()
    c4ml_mem_dict = {
        "max_vms_memory": mem_prof.max_vms_memory,
        "max_rss_memory": mem_prof.max_rss_memory,
    } 
    mem_prof2 = ProcessContainer(pid=os.getpid())
    thread_handle2 = mem_prof2.profile()
    commands = [
        "vivado",
        "-mode",
        "batch",
        "-source",
        f"{base_dir}/synth.tcl",
        "-nojournal",
        "-nolog",
        "-tclargs",
        work_dir,
        base_dir,
        top_name,
    ]
    with open(f"{work_dir}/vivado.log", "w") as log_file:
        cp = subprocess.run(commands, stdout=log_file, stderr=log_file)
    synthesis_time = time.perf_counter()
    assert cp.returncode == 0
    mem_prof2.stop()
    thread_handle2.join()
    info_dict = {}
    info_dict["compile_max_vms_memory"] = mem_prof.max_vms_memory
    info_dict["compile_max_rss_memory"] = mem_prof.max_rss_memory
    info_dict["vivado_max_rss_memory"] = mem_prof2.max_rss_memory
    info_dict["vivado_max_vms_memory"] = mem_prof2.max_vms_memory
    info_dict["total_max_rss_memory"] = max(mem_prof.max_rss_memory, mem_prof2.max_rss_memory)
    info_dict["total_max_vms_memory"] = max(mem_prof.max_vms_memory, mem_prof2.max_vms_memory)
    info_dict["compile_duration"] = compile_time - start_time
    info_dict["simulation_duration"] = simulation_time - compile_time
    info_dict["synthesis_duration"] = synthesis_time - simulation_time
    info_dict["total_duration"] = synthesis_time - start_time
    info_dict["exact_latency"] = (
        len(lbir_model.layers) + 1
    )  # a stage for every lbir layer
    info_dict["tool"] = "chisel4ml"
    with open(f"{work_dir}/info.json", "w") as info_file:
        json.dump(info_dict, info_file)
    os.chdir(curr_dir)
