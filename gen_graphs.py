import os
import json
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import numpy as np
from main import EXPERIMENTS
from main import get_work_dir
from parse_reports import parse_reports

def exp_get_time_list(exp, tool='chisel4ml'):
    time_list = []
    for run in exp:
        time_list.append(float(run[tool]['syn_time']) / (60 * 60))
    return time_list


def exp_get_elem_list(exp, tool='chisel4ml', elem='CLB LUTs*'):
    elem_list = []
    for run in exp:
        index = -1
        for ind, x in enumerate(run[tool]['util']['CLB Logic']):
            if x['Site Type'] == elem:
                index = ind
        if index == -1:
            raise ValueError
        elem_list.append(float(run[tool]['util']['CLB Logic'][index]['Used']))
    return elem_list


def exp_get_delay_list(exp, tool='chisel4ml', delay_type='Path Delay'):
    elem_list = []
    for run in exp:
        val = run[tool]['design'][delay_type][0:5]
        elem_list.append(float(val))
    return elem_list


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
    "prune_rate": "Prune Rate [%]",
}


def get_x_axis(exp_dict):
    for key in exp_dict.keys():
        if len(exp_dict[key]) > 1:
            if isinstance(exp_dict[key][0], (tuple, list)):
                return list(map(lambda x: x[0], exp_dict[key])), key_to_name_dict[key]
            elif key == 'prune_rate':
                return (np.array(exp_dict[key]) * 100).tolist(), key_to_name_dict[key]
            else:
                return exp_dict[key], key_to_name_dict[key]


def gather_results(exp):
    exp_name = exp[2]
    exp_base = f'/circuits/{exp_name}/'
    exp_keys = exp[0].keys()
    feat_list = list(itertools.product(*exp[0].values()))
    results = []
    for feat in feat_list:
        work_dir = get_work_dir(exp_keys, feat, base=exp_base)
        c4ml_res = parse_reports(f'{work_dir}/c4ml/')
        hls4ml_res = parse_reports(f'{work_dir}/hls4ml/')
        test_res = {'work_dir': work_dir, 'chisel4ml': c4ml_res, 'hls4ml': hls4ml_res}
        if os.path.exists(f'{work_dir}/acc.log'):
            with open(f'{work_dir}/acc.log', 'r') as f:
                ftxt = f.readlines()
            acc = float(ftxt[1])
            test_res['acc'] = acc
        results.append(test_res)
    return results

def exp_get_acc_list(data):
    acc_list = []
    for d in data:
        acc_list.append(d['acc'])
    return acc_list

def generate_report_for_exp(exp):
    data = gather_results(exp)

    c4ml_syn_time_list = exp_get_time_list(data, tool='chisel4ml')
    hls4ml_syn_time_list = exp_get_time_list(data, tool='hls4ml')
    c4ml_luts_list = exp_get_elem_list(data, tool='chisel4ml', elem='CLB LUTs*')
    hls4ml_luts_list = exp_get_elem_list(data, tool='hls4ml', elem='CLB LUTs*')
    c4ml_ff_list = exp_get_elem_list(data, tool='chisel4ml', elem='CLB Registers')
    hls4ml_ff_list = exp_get_elem_list(data, tool='hls4ml', elem='CLB Registers')
    c4ml_delay_list = exp_get_delay_list(data, tool='chisel4ml')
    hls4ml_delay_list = exp_get_delay_list(data, tool='hls4ml')
    if 'acc' in data[0].keys():
        acc_list = exp_get_acc_list(data)

    x_axis, x_axis_name = get_x_axis(exp[0])
    lut_arr = np.array([x_axis, c4ml_luts_list, hls4ml_luts_list])
    time_arr = np.array([x_axis, c4ml_syn_time_list, hls4ml_syn_time_list])
    delay_arr = np.array([x_axis, c4ml_delay_list, hls4ml_delay_list])
    if 'acc' in data[0].keys():
        acc_arr = np.array([x_axis, acc_list])

    lut_df = pd.DataFrame(lut_arr.T, columns=[x_axis_name, 'chisel4ml', 'hls4ml'])
    lut_df[x_axis_name] = lut_df[x_axis_name].apply(lambda x: int(x))
    time_df = pd.DataFrame(time_arr.T, columns=[x_axis_name, 'chisel4ml', 'hls4ml'])
    time_df[x_axis_name] = time_df[x_axis_name].apply(lambda x: int(x))
    delay_df = pd.DataFrame(delay_arr.T, columns=[x_axis_name, 'chisel4ml', 'hls4ml'])
    delay_df[x_axis_name] = delay_df[x_axis_name].apply(lambda x: int(x))
    if 'acc' in data[0].keys():
        acc_df = pd.DataFrame(acc_arr.T, columns=[x_axis_name, 'Accuracy'])
        acc_df[x_axis_name] = acc_df[x_axis_name].apply(lambda x: int(x))
    melt_lut_df = lut_df.melt(x_axis_name, var_name='tool', value_name='Look-Up Tables')
    melt_time_df = time_df.melt(x_axis_name, var_name='tool', value_name='Synthesis Time [hours]')
    melt_delay_df = delay_df.melt(x_axis_name, var_name='tool', value_name='Path Delay [ns]')
    
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set(font_scale=1.5)
    if not os.path.isdir(f"plots/{exp[2]}"):
        os.makedirs(f"plots/{exp[2]}")
    sns.catplot(
        x=x_axis_name,
        y="Look-Up Tables",
        hue='tool',
        data=melt_lut_df,
        kind='point',
        markers=['o', 's'],
        legend_out=False,
        legend='brief'
    )
    
    plt.savefig(f'plots/{exp[2]}/lut_plot.png')
    plt.close()
    sns.catplot(
        x=x_axis_name,
        y="Synthesis Time [hours]",
        hue='tool',
        data=melt_time_df,
        kind='point',
        markers=['o', 's'],
        legend_out=False,
        legend='brief'
    )
    plt.ylim(0)
    plt.savefig(f'plots/{exp[2]}/syn_time_plot.png')
    plt.close()
    sns.catplot(
        x=x_axis_name,
        y="Path Delay [ns]",
        hue='tool',
        data=melt_delay_df,
        kind='point',
        markers=['o', 's'],
        legend_out=False,
        legend='brief'
    )
    plt.ylim(0)
    plt.savefig(f'plots/{exp[2]}/delay_plot.png')
    plt.close()

    if 'acc' in data[0].keys():
        sns.catplot(
            x=x_axis_name,
            y="Accuracy",
            data=acc_df,
            kind='point',
            color='k'
        )
        plt.savefig(f'plots/{exp[2]}/acc.png')
        plt.close()

if __name__ == "__main__":
    for exp in EXPERIMENTS:
        try:
            generate_report_for_exp(exp)
        except OSError:
            print(f"Results for {exp[2]} not found. Moving on.")
