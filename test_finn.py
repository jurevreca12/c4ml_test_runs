import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import time
import shutil
import json
from memory_profiler import ProcessContainer

def test_finn(qonnx_model_file, work_dir, base_dir):
    curr_dir = os.getcwd()
    pid = os.getpid()
    mem_prof = ProcessContainer(pid=pid)
    thread_handle = mem_prof.profile()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    # Delete previous run results if exist
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        print("Previous run results deleted!")
    starttime = time.perf_counter()
    cfg_stitched_ip = build.DataflowBuildConfig(
        output_dir          = work_dir,
        mvau_wwidth_max     = 99999999,
        target_fps          = 99999999,
        synth_clk_period_ns = 10.0,
        fpga_part           = "xcvu9p-flga2104-2L-e",
        verbose             = True,
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
