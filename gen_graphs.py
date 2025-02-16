import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from main import EXPERIMENTS
from main import get_work_dir
from parse_reports import parse_reports
from parse_reports import parse_finn_reports


def exp_get_time_list(data, tool="chisel4ml"):
    time_list = []
    for run in data:
        time_list.append(float(run[tool]["info_rpt"]["total_duration"]) / (60 * 60))
    return time_list


def exp_get_mem_list(data, tool="chisel4ml"):
    mem_list = []
    for run in data:
        max_rss = float(run[tool]['info_rpt']['total_max_rss_memory'])
        mem_list.append(max_rss / (1024 * 1024))
    return mem_list

def exp_get_elem_list(data, tool="chisel4ml", elem="CLB LUTs*"):
    elem_list = []
    for run in data:
        index = -1
        for ind, x in enumerate(run[tool]["util"]["CLB Logic"]):
            if x["Site Type"] == elem:
                index = ind
        if index == -1:
            raise ValueError
        elem_list.append(float(run[tool]["util"]["CLB Logic"][index]["Used"]))
    return elem_list

def exp_get_elem_list_finn(data, elem='LUT'):
    elem_list = []
    for run in data:
        elem_list.append(float(run['finn']['ooc_synth_and_timing'][elem]))
    return elem_list

def exp_get_delay_list(data, tool="chisel4ml", delay_type="Path Delay"):
    elem_list = []
    for run in data:
        val = run[tool]["design"][delay_type][0:5]
        elem_list.append(float(val))
    return elem_list

def exp_get_delay_list_finn(data):
    elem_list = []
    for run in data:
        elem_list.append(run['finn']['ooc_synth_and_timing']['Delay'])
    return elem_list


def exp_get_troughput_list(data, tool="chisel4ml"):
    throughput_list = []
    for run in data:
        delay = float(run[tool]["design"]["Path Delay"][0:5])
        if tool == "chisel4ml":
            latency_cycles = float(run[tool]["info_rpt"]["exact_latency"])
        else:
            latency_cycles = float(run[tool]["info_rpt"]["CosimReport"]["LatencyAvg"])
        throughput_list.append(latency_cycles * (10**9) / delay)
    return throughput_list

def exp_get_troughput_list_finn(data):
    throughput_list = []
    for run in data:
        throughput_list.append(run['finn']['ooc_synth_and_timing']['estimated_throughput_fps'])
    return throughput_list


def exp_get_total_latency_list(data, tool="chisel4ml"):
    latency_list = []
    for run in data:
        delay = float(run[tool]["design"]["Path Delay"][0:5])
        if tool == "chisel4ml":
            latency_cycles = float(run[tool]["info_rpt"]["exact_latency"])
        else:
            latency_cycles = float(run[tool]["info_rpt"]["CosimReport"]["LatencyAvg"])
        latency_list.append(latency_cycles * delay)
    return latency_list

def exp_get_total_latency_list_finn(data):
    latency_list = []
    for run in data:
        latency_cycles = run['finn']['rtlsim_performance']['latency_cycles']
        delay = run['finn']['ooc_synth_and_timing']['Delay']
        latency_list.append(latency_cycles * delay)
    return latency_list

key_to_name_dict = {
    "input_ch": "Input Channels",
    "output_ch": "Output Channels",
    "iq": "Input Bitwidth",
    "wq": "Weights Bitwidth",
    "in_features": "Input Features",
    "out_features": "Output Features",
    "channels": "Channels",
    "input_size": "Input Size",
    "kernel_size": "Kernel Size",
}


def get_x_axis(exp_dict):
    for key in exp_dict.keys():
        if len(exp_dict[key]) > 1:
            if isinstance(exp_dict[key][0], (tuple, list)):
                return list(map(lambda x: x[0], exp_dict[key])), key_to_name_dict[key]
            else:
                return exp_dict[key], key_to_name_dict[key]


def gather_results(exp):
    exp_name = exp[2]
    exp_base = f"/circuits/{exp_name}/"
    exp_keys = exp[0].keys()
    feat_list = list(itertools.product(*exp[0].values()))
    results = []
    for feat in feat_list:
        work_dir = get_work_dir(exp_keys, feat, base=exp_base)
        c4ml_res = parse_reports(f"{work_dir}/c4ml/")
        hls4ml_res = parse_reports(
            f"{work_dir}/hls4ml/", util_rpt_file="vivado_synth.rpt"
        )
        finn_res = parse_finn_reports(
            f"{work_dir}/finn"
        )
        test_res = {
            "work_dir": work_dir, 
            "chisel4ml": c4ml_res, 
            "hls4ml": hls4ml_res,
            "finn": finn_res
        }
        if os.path.exists(f"{work_dir}/acc.log"):
            with open(f"{work_dir}/acc.log", "r") as f:
                ftxt = f.readlines()
            acc = float(ftxt[1])
            test_res["acc"] = acc
        results.append(test_res)
    return results


def generate_report_for_exp(exp):
    data = gather_results(exp)

    x_axis, x_axis_name = get_x_axis(exp[0])

    c4ml_syn_time_list = exp_get_time_list(data, tool="chisel4ml")
    hls4ml_syn_time_list = exp_get_time_list(data, tool="hls4ml")
    finn_syn_time_list = exp_get_time_list(data, tool="finn")
    syn_time_arr = np.array([x_axis, c4ml_syn_time_list, hls4ml_syn_time_list, finn_syn_time_list])
    syn_time_df = pd.DataFrame(syn_time_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_syn_time_df = syn_time_df.melt(
        x_axis_name, var_name="tool", value_name="Synthesis Time [hours]"
    )

    c4ml_luts_list = exp_get_elem_list(data, tool="chisel4ml", elem="CLB LUTs*")
    hls4ml_luts_list = exp_get_elem_list(data, tool="hls4ml", elem="CLB LUTs*")
    finn_luts_list = exp_get_elem_list_finn(data, elem='LUT')
    lut_arr = np.array([x_axis, c4ml_luts_list, hls4ml_luts_list, finn_luts_list])
    lut_df = pd.DataFrame(lut_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_lut_df = lut_df.melt(x_axis_name, var_name="tool", value_name="Look-Up Tables")

    c4ml_ff_list = exp_get_elem_list(data, tool="chisel4ml", elem="CLB Registers")
    hls4ml_ff_list = exp_get_elem_list(data, tool="hls4ml", elem="CLB Registers")
    finn_ff_list = exp_get_elem_list_finn(data, elem='FF')
    ff_arr = np.array([x_axis, c4ml_ff_list, hls4ml_ff_list, finn_ff_list])
    ff_df = pd.DataFrame(ff_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_ff_df = ff_df.melt(
        x_axis_name, var_name='tool', value_name="Flip-Flops"
    )

    c4ml_delay_list = exp_get_delay_list(data, tool="chisel4ml")
    hls4ml_delay_list = exp_get_delay_list(data, tool="hls4ml")
    finn_delay_list = exp_get_delay_list_finn(data)
    delay_arr = np.array([x_axis, c4ml_delay_list, hls4ml_delay_list, finn_delay_list])
    delay_df = pd.DataFrame(delay_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_delay_df = delay_df.melt(
        x_axis_name, var_name="tool", value_name="Path Delay [ns]"
    )

    c4ml_mem_usage_list = exp_get_mem_list(data, tool="chisel4ml")
    hls4ml_mem_usage_list = exp_get_mem_list(data, tool="hls4ml")
    finn_mem_usage_list = exp_get_mem_list(data, tool="finn")
    mem_usage_arr = np.array([x_axis, c4ml_mem_usage_list, hls4ml_mem_usage_list, finn_mem_usage_list])
    mem_usage_df = pd.DataFrame(mem_usage_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_mem_usage_df = mem_usage_df.melt(
        x_axis_name, var_name="tool", value_name="Peak Memory [MB]"
    )

    c4ml_throughput_list = exp_get_troughput_list(data, tool="chisel4ml")
    hls4ml_throughput_list = exp_get_troughput_list(data, tool="hls4ml")
    finn_throughput_list = exp_get_troughput_list_finn(data)
    throughput_arr = np.array([x_axis, c4ml_throughput_list, hls4ml_throughput_list, finn_throughput_list])
    throughput_df = pd.DataFrame(throughput_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_throughput_df = throughput_df.melt(
        x_axis_name, var_name="tool", value_name="Throughput [Hz]"
    )

    c4ml_latency_list = exp_get_total_latency_list(data, tool="chisel4ml")
    hls4ml_latency_list = exp_get_total_latency_list(data, tool="hls4ml")
    finn_latency_list = exp_get_total_latency_list_finn(data)    
    latency_arr = np.array([x_axis, c4ml_latency_list, hls4ml_latency_list, finn_latency_list])
    latency_df = pd.DataFrame(latency_arr.T, columns=[x_axis_name, "chisel4ml", "hls4ml", "finn"])
    melt_latency_df = latency_df.melt(
        x_axis_name, var_name="tool", value_name="Total Latency [ns]"
    )

    for df in (lut_df, syn_time_df, delay_df, mem_usage_df, throughput_df, latency_df):
        df[x_axis_name] = df[x_axis_name].apply(lambda x: int(x))



    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    if not os.path.isdir(f"plots/{exp[2]}"):
        os.makedirs(f"plots/{exp[2]}")

    sns.catplot(
        x=x_axis_name,
        y="Look-Up Tables",
        hue="tool",
        data=melt_lut_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.savefig(f"plots/{exp[2]}/lut_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Flip-Flops",
        hue="tool",
        data=melt_ff_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.savefig(f"plots/{exp[2]}/ff_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Synthesis Time [hours]",
        hue="tool",
        data=melt_syn_time_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.savefig(f"plots/{exp[2]}/syn_time_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Path Delay [ns]",
        hue="tool",
        data=melt_delay_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.ylim(0)
    plt.savefig(f"plots/{exp[2]}/delay_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Peak Memory [MB]",
        hue="tool",
        data=melt_mem_usage_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.ylim(0)
    plt.savefig(f"plots/{exp[2]}/mem_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Throughput [Hz]",
        hue="tool",
        data=melt_throughput_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.ylim(0)
    plt.savefig(f"plots/{exp[2]}/throughput_plot.png")
    plt.close()

    sns.catplot(
        x=x_axis_name,
        y="Total Latency [ns]",
        hue="tool",
        data=melt_latency_df,
        kind="point",
        markers=["o", "s", "^"],
        legend_out=False,
        legend="brief",
    )
    plt.ylim(0)
    plt.savefig(f"plots/{exp[2]}/latency_plot.png")
    plt.close()


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        try:
            generate_report_for_exp(exp)
        except OSError:
            print(f"Error {exp[2]} not found. Skipping.")
