import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
import os
import time
import shutil
import json
from memory_profiler import ProcessContainer
from qonnx.core.modelwrapper import ModelWrapper
from pathlib import Path
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames


def onnx_set_attr(node, attr_name: str, val) -> None:
    for attr in node.attribute:
        if attr.name == attr_name:
            attr.i = val
            break

def onnx_get_attr(node, attr_name: str):
    for attr in node.attribute:
        if attr.name == attr_name:
            return attr.i


def step_set_max_parallelization(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueNodeNames())
    mvau_hls_nodes = model.get_nodes_by_op_type('MVAU_hls')
    for node in mvau_hls_nodes:
        mh = onnx_get_attr(node, 'MH')
        mw = onnx_get_attr(node, 'MW')
        onnx_set_attr(node, 'PE',  mh) 
        onnx_set_attr(node, 'SIMD',  mw)
    return model

_steps_custom = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    step_set_max_parallelization,
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

def test_finn(qonnx_model_file, work_dir, base_dir):
    global _steps_custom
    curr_dir = os.getcwd()
    pid = os.getpid()
    mem_prof = ProcessContainer(pid=pid)
    thread_handle = mem_prof.profile()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    starttime = time.perf_counter()
    cfg_stitched_ip = build.DataflowBuildConfig(
        output_dir          = work_dir,
        synth_clk_period_ns = 10.0,
        fpga_part           = "xcvu9p-flga2104-2L-e",
        verbose             = True,
        steps               = _steps_custom,
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ]
    )
    build.build_dataflow_cfg(qonnx_model_file, cfg_stitched_ip)
    duration = time.perf_counter() - starttime
    mem_prof.stop()
    thread_handle.join()
    info_dict = {}
    info_dict['total_duration'] = duration
    info_dict['total_max_vms_memory'] = mem_prof.max_vms_memory
    info_dict['total_max_rss_memory'] = mem_prof.max_rss_memory
    info_dict['tool'] = "finn"
    with open(f"{work_dir}/info.json", 'w') as info_file:
        json.dump(info_dict, info_file)
    os.chdir(curr_dir)
