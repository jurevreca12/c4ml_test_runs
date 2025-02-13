import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import time
import shutil

def test_finn(qonnx_model_file, work_dir, base_dir):
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
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ]
    )
    build.build_dataflow_cfg(qonnx_model_file, cfg_stitched_ip)
    duration = time.perf_counter() - starttime
