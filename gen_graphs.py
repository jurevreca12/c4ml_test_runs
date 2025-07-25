import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main import EXPERIMENTS
from main import get_work_dir
from main import get_exp_by_name
from parse_reports import parse_reports
from parse_reports import parse_finn_reports
import argparse

FINN_TARGET_CLK_NS = 10

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
    "bitwidth": "Bitwidth",
    "prune_rate": "Pruning Rate",
}


def get_total_latency_c4ml(run):
    delay = float(run["design"]["Path Delay"][0:5])
    latency_cycles = float(run["info_rpt"]["exact_latency"])
    total_latency = latency_cycles * delay
    return total_latency


def get_total_latency_hls4ml(run):
    delay = float(run["design"]["Path Delay"][0:5])
    latency_cycles = float(run["info_rpt"]["CosimReport"]["LatencyAvg"])
    total_latency = latency_cycles * delay
    return total_latency


def get_total_latency_finn(run):
    latency_cycles = run["rtlsim_performance"]["latency_cycles"]
    delay = FINN_TARGET_CLK_NS - run["ooc_synth_and_timing"]["WNS"]
    total_latency = latency_cycles * delay
    return total_latency


def get_throughput_hls4ml(run):
    path_delay_ns = float(run["design"]["Path Delay"][0:5])
    init_interval = int(run['info_rpt']['CSynthesisReport']['IntervalMax'])
    assert init_interval == int(run['info_rpt']['CSynthesisReport']['IntervalMin'])
    return (10**9) / (path_delay_ns * init_interval)

FIELDS = {
    'syn_time': {
        'chisel4ml': ['info_rpt', 'total_duration', lambda x: float(x) / (60 * 60)],
        'hls4ml': ['info_rpt', 'total_duration', lambda x: float(x) / (60 * 60)],
        'finn': ['info_rpt', 'total_duration', lambda x: float(x) / (60 * 60)],
        'long_name': 'Generation Time [hours]',
    },
    'lut': {
        'chisel4ml': ['util', 'CLB Logic', 0, 'Used', float, lambda x: x / 1000, int], # kLUT
        'hls4ml': ['util', 'CLB Logic', 0, 'Used', float, lambda x: x / 1000, int],
        'finn': ['ooc_synth_and_timing', 'LUT', float, lambda x: x / 1000, int],
        'assert': 'CLB LUTs*',  # assert that site-type column equals it
        'long_name': 'Look-Up Tables [kLUT]',
    },
    'ff': {
        'chisel4ml': ['util', 'CLB Logic', 3, 'Used', int],
        'hls4ml': ['util', 'CLB Logic', 3, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'FF', int],
        'ssert': 'CLB Registers',
        'long_name': 'Flip-Flops',
    },
    'bram_tile': {
        'chisel4ml': ['util', 'BLOCKRAM', 0, 'Used', int],
        'hls4ml': ['util', 'BLOCKRAM', 0, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'BRAM', int],
        'long_name': 'Block RAM Tile'
    },
    'bram_b36': {
        'chisel4ml': ['util', 'BLOCKRAM', 1, 'Used', int],
        'hls4ml': ['util', 'BLOCKRAM', 1, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'BRAM_36K', int], 
        'long_name': 'Block RAM 36kb'
    },
    'bram_b18': {
        'chisel4ml': ['util', 'BLOCKRAM', 2, 'Used', int],
        'hls4ml': ['util', 'BLOCKRAM', 2, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'BRAM_18K', float, int],
        'long_name': 'Block RAM 16kb'
    },
    'uram': {
        'chisel4ml': ['util', 'BLOCKRAM', 3, 'Used', int],
        'hls4ml': ['util', 'BLOCKRAM', 3, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'URAM', int],
        'long_name': 'UltraRAM'
    },
    'dsp': {
        'chisel4ml': ['util', 'ARITHMETIC', 0, 'Used', int],
        'hls4ml': ['util', 'ARITHMETIC', 0, 'Used', int],
        'finn': ['ooc_synth_and_timing', 'DSP', int],
        'long_name': 'DSP Block',
    },
    'path_delay': {
        'chisel4ml': ['design', 'Path Delay', lambda x: float(x[0:5])],
        'hls4ml': ['design', 'Path Delay', lambda x: float(x[0:5])],
        'finn': ['ooc_synth_and_timing', 'WNS', float, lambda x: FINN_TARGET_CLK_NS - x],
        'long_name': 'Path Delay [ns]',
    },
    'peak_mem_usage': {
        'chisel4ml': ['info_rpt', 'total_max_rss_memory', lambda x: int(x) / (1024 * 1024)],
        'hls4ml': ['info_rpt', 'total_max_rss_memory', lambda x: int(x) / (1024 * 1024)],
        'finn': ['info_rpt', 'total_max_rss_memory', lambda x: int(x) / (1024 * 1024)],
        'long_name': 'Peak Memory [MiB]',
    },
    'init_interval': {
        'chisel4ml': [lambda _: 1],
        'hls4ml': ['info_rpt', 'CSynthesisReport', 'IntervalMax', int],
        'finn': [lambda _: None],  # TODO
        'long_name': 'Initiation Interval',
    },
    'throughput': {
        'chisel4ml': ['design', 'Path Delay', lambda x: (10**9) / float(x[0:5]), lambda x: x / (10**6)],
        'hls4ml': [get_throughput_hls4ml, lambda x: x / (10**6)],
        'finn': ['ooc_synth_and_timing', 'estimated_throughput_fps', float, lambda x: x / (10**6)],
        'long_name': 'Throughput [MHz]',
    },
    'latency_cycles': {
        'chisel4ml': ['info_rpt', 'exact_latency', int],
        'hls4ml': ['info_rpt', 'CosimReport', 'LatencyAvg', int],
        'finn': ['rtlsim_performance', 'latency_cycles', int],
        'long_name': 'Latency Cycles',
    },
    'total_latency': {
        'chisel4ml': [get_total_latency_c4ml],
        'hls4ml': [get_total_latency_hls4ml],
        'finn': [get_total_latency_finn],
        'long_name': 'Total Latency [ns]',
    },
}



def gather_results(exp):
    exp_name = exp[2]
    exp_base = f"/circuits/{exp_name}/"
    exp_keys = exp[0].keys()
    feat_list = list(itertools.product(*exp[0].values()))
    results = []
    if not os.path.exists(f"./circuits/{exp_name}"):
        print(f"SKIPPING experiment {exp_name}. Directory does not exist!")
        return None
    for feat in feat_list:
        work_dir = get_work_dir(exp_keys, feat, base=exp_base)
        try:
            c4ml_res = parse_reports(f"{work_dir}/c4ml")
        except FileNotFoundError:
            print(f"WARNING: Could not parse {work_dir}/c4ml. Setting to None.")
            c4ml_res = None

        try:
            hls4ml_res = parse_reports(
                f"{work_dir}/hls4ml/", util_rpt_file="vivado_synth.rpt"
            )
        except FileNotFoundError:
            print(f"WARNING: Could not parse {work_dir}/hls4ml. Setting to None.")
            hls4ml_res = None

        try:
            finn_res = parse_finn_reports(
                f"{work_dir}/finn"
            )
        except:
            print(f"WARNING: Could not parse {work_dir}/finn. Setting to None.")
            finn_res = None

        test_res = {
            "work_dir": work_dir,
            "chisel4ml": c4ml_res,
            "hls4ml": hls4ml_res,
            "finn": finn_res
        }
        if os.path.exists(f"{work_dir}/acc.log"):
            with open(f"{work_dir}/acc.log", 'r') as f:
                ftxt = f.readlines()
            acc = float(ftxt[1])
            test_res["acc"] = acc
        results.append(test_res)
    return results


def get_x_axis(exp_dict):
    for key in exp_dict.keys():
        if len(exp_dict[key]) > 1:
            if isinstance(exp_dict[key][0], (tuple, list)):
                return list(map(lambda x: x[0], exp_dict[key])), key_to_name_dict[key]
            else:
                return exp_dict[key], key_to_name_dict[key]



def undict(dct, fields):
    value = dct
    for field in fields:
        if callable(field):
            value = field(value)
        elif isinstance(value, (dict, list)) and value[field] is None:
            return None
        else:
            value = value[field]
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="gen_graphs")
    parser.add_argument(
        "--exp-name", "-name", default="", help="Name of the experiment to run."
    )
    args = parser.parse_args()
    if args.exp_name != "":
        exp = get_exp_by_name(args.exp_name)
        EXPERIMENTS_MOD = (exp,)
    else:
        EXPERIMENTS_MOD = EXPERIMENTS
    for exp in EXPERIMENTS_MOD:
        data = gather_results(exp)
        if data is None:
            continue
        x_axis, x_axis_name = get_x_axis(exp[0])
        df = pd.DataFrame(
            columns=(
                'x_axis',
                'tool',
                *FIELDS.keys()
            ),
        )
        df['x_axis'] = x_axis
        df = pd.concat([
                df.assign(tool='chisel4ml'),
                df.assign(tool='hls4ml'),
                df.assign(tool='finn'),
            ],
            ignore_index=True
        )
        for col in FIELDS.keys():
            for tool in ('chisel4ml', 'hls4ml', 'finn'):
                for ind, xval in enumerate(x_axis):
                    val = undict(
                        data[ind],
                        fields=[tool] + FIELDS[col][tool]
                    )
                    cond = 'tool==@tool & x_axis==@xval'
                    cond_index = df.loc[df.eval(cond)].index
                    df.loc[cond_index, col] = val
        
        sns.set_style("darkgrid", {
                "axes.facecolor": ".9",
            }
        )
        sns.set(font_scale=1.3)
        if not os.path.isdir(f"plots/{exp[2]}"):
            os.makedirs(f"plots/{exp[2]}")

        df = df.rename(columns={'x_axis': x_axis_name})
        for col in FIELDS.keys():
            df = df.rename(columns={col: FIELDS[col]['long_name']})
            sns.catplot(
                x=x_axis_name,
                y=FIELDS[col]['long_name'],
                hue='tool',
                data=df,
                kind='point',
                markers=['o', 's', '^'],
                dodge=True,
                legend_out=False,
                legend='brief',
            )
            plt.ylim(0)
            plt.savefig(f'plots/{exp[2]}/{col}_plot.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
        df.to_csv(f'plots/{exp[2]}/{exp[2]}.csv')
