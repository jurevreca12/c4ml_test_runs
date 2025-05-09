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
import onnx

import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from round_thresholds_old import RoundAndClipThresholdsOld
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    ApplyConfig,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
#from remove_transpose_flatten import RemoveTransposeFlatten

from qonnx.util.config import extract_model_config_to_json
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.base import Transformation
import numpy as np
from quant_bias_to_init import QuantizedBiasToInitializer

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
)
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.set_fifo_depths import (
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
)
from set_fifo_depths_custom import InsertAndSetFIFODepthsCustom
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance


def step_custom_lower_convs(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(QuantizedBiasToInitializer())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbTransposeIntoFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    return model


def step_custom_convert_to_hw_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PreferedImplStyle())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(RoundAndClipThresholdsOld())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbTransposeIntoFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PreferedImplStyle())
    return model

def onnx_set_attr(node, attr_name: str, val) -> None:
    for attr in node.attribute:
        if attr.name == attr_name:
            attr.i = val
            return
    nattr = onnx.helper.make_attribute(attr_name, val)
    node.attribute.append(nattr)    


def onnx_get_attr(node, attr_name: str):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == 2:  # INT
                return attr.i
            elif attr.type == 7:  # INTS
                return attr.ints
            else:
                raise ValueError
    raise ValueError


class PreferedImplStyle(Transformation):
    "Sets the attribute preferred_impl_style to hls. See SpecializeLayers transformation for context."
    
    def __init__(self, style="hls"):
        super().__init__()
        self.style = style
        self.finn_nodes = (
            "MVAU",
            "VVAU",
            "ConvolutionInputGenerator",
            "StreamingDataWidthConverter",
            "Thresholding"
        )
    def apply(self, model):
        for node in model.graph.node:
            if node.op_type in self.finn_nodes:
                onnx_set_attr(node, 'preferred_impl_style', 'hls')
        return model, False


def step_set_max_parallelization(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueNodeNames())
    mvau_hls_nodes = model.get_nodes_by_op_type('MVAU_hls')
    for node in mvau_hls_nodes:
        mh = onnx_get_attr(node, 'MH')
        mw = onnx_get_attr(node, 'MW')
        onnx_set_attr(node, 'PE',  mh) 
        onnx_set_attr(node, 'SIMD',  mw)
        onnx_set_attr(node, 'mem_mode',  "internal_embedded")

    conv_inp_hls_nodes = model.get_nodes_by_op_type('ConvolutionInputGenerator_hls')
    for node in conv_inp_hls_nodes:
        in_ch = onnx_get_attr(node, 'IFMChannels')
        onnx_set_attr(node, 'SIMD', in_ch) 
        onnx_set_attr(node, 'parallel_window', 1)
  
    vvau_hls_nodes = model.get_nodes_by_op_type('VVAU_hls')
    for node in vvau_hls_nodes:
        ch = onnx_get_attr(node, 'Channels')
        kernel = onnx_get_attr(node, 'Kernel')
        onnx_set_attr(node, 'PE', ch)
        onnx_set_attr(node, 'SIMD',  np.array(kernel).prod())
        onnx_set_attr(node, 'mem_mode',  "internal_embedded")

    pool_hls_nodes = model.get_nodes_by_op_type('Pool_hls')
    for node in pool_hls_nodes:
        ch = onnx_get_attr(node, 'Channels')
        onnx_set_attr(node, 'PE', ch)

    thresholding_nodes = model.get_nodes_by_op_type('Thresholding_hls')
    for node in thresholding_nodes:
        ch = onnx_get_attr(node, 'NumChannels')
        onnx_set_attr(node, 'PE', ch)

    # check that we did not get RTL nodes by mistake
    for node in model.graph.node:
        assert 'rtl' not in node.op_type
    return model


def step_global_in_quant_signed(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Changes the Quant nodes to be signed, because that is what FINN supports."
    quant_nodes = model.get_nodes_by_op_type('Quant')
    for node in quant_nodes:
        if 'global_in' in node.input or 'MaxPool' in node.input[0]:
            onnx_set_attr(node, 'signed', 1)
    return model


def step_custom_set_fifo_depth(model: ModelWrapper, cfg: DataflowBuildConfig):
    model_multi_io = len(model.graph.input) > 1 or len(model.graph.output) > 1
    force_python_sim = model_multi_io or cfg.force_python_rtlsim
    if model_multi_io:
        warnings.warn(
            "Multi-in/out streams currently not supported "
            + "in FINN C++ verilator driver, falling back to Python"
        )
    model = model.transform(
        InsertAndSetFIFODepthsCustom(
            cfg._resolve_fpga_part(),
            cfg._resolve_hls_clk_period(),
            swg_exception=cfg.default_swg_exception,
            vivado_ram_style=cfg.large_fifo_mem_style,
            force_python_sim=force_python_sim,
        )
    )
    # extract the final configuration and save it as json
    hw_attrs = [
        "PE",
        "SIMD",
        "parallel_window",
        "ram_style",
        "depth",
        "impl_style",
        "resType",
        "mem_mode",
        "runtime_writeable_weights",
        "inFIFODepths",
        "outFIFODepths",
        "depth_trigger_uram",
        "depth_trigger_bram",
    ]
    extract_model_config_to_json(model, cfg.output_dir + "/final_hw_config.json", hw_attrs)

    # perform FIFO splitting and shallow FIFO removal only after the final config
    # json file has been written. otherwise, since these transforms may add/remove
    # FIFOs, we get name mismatch problems when trying to reuse the final config.
    if cfg.split_large_fifos:
        model = model.transform(SplitLargeFIFOs())
    model = model.transform(RemoveShallowFIFOs())

    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    return model


_steps_custom = [
    step_global_in_quant_signed,
    step_custom_lower_convs,
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    step_custom_convert_to_hw_layers,
   "step_create_dataflow_partition",
    "step_specialize_layers",
    step_set_max_parallelization,
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    step_custom_set_fifo_depth,
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    #"step_make_pynq_driver",
    #"step_deployment_package",
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
