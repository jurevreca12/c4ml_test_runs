import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from main import linear_layer_var_in_features_exp
from main import linear_layer_var_out_features_exp
from main import linear_layer_var_iq_exp
from main import linear_layer_var_wq_exp
from main import conv_layer_var_input_ch_exp
from main import conv_layer_var_output_ch_exp
from main import dw_conv_layer_var_output_ch_exp
from main import conv_layer_var_iq_exp
from main import conv_layer_var_wq_exp
from main import maxpool_layer_var_input_size
from main import maxpool_layer_var_channels
from main import maxpool_layer_var_kernel_size

experiments = [
    ("linear_layer_var_in_features", linear_layer_var_in_features_exp), 
    ("linear_layer_var_out_features", linear_layer_var_out_features_exp),
    ("linear_layer_var_iq", linear_layer_var_iq_exp),
    ("linear_layer_var_wq", linear_layer_var_wq_exp),

    ("conv_layer_var_input_ch", conv_layer_var_input_ch_exp),
    ("conv_layer_var_output_ch", conv_layer_var_output_ch_exp),
    #("dw_conv_layer_var_output_ch", dw_conv_layer_var_output_ch_exp),
    ("conv_layer_var_iq", conv_layer_var_iq_exp),
    ("conv_layer_var_wq", conv_layer_var_wq_exp),

    ("maxpool_layer_var_input_size", maxpool_layer_var_input_size),
    ("maxpool_layer_var_channels", maxpool_layer_var_channels),
    ("maxpool_layer_var_kernel_size", maxpool_layer_var_kernel_size),
]

def exp_get_time_list(exp, tool='chisel4ml'):
    time_list = []
    for run in exp:
        time_list.append(float(run[tool]['syn_time']))
    return time_list


def exp_get_elem_list(exp, tool='chisel4ml', elem='CLB LUTs*'):
    elem_list =[]       
    for run in exp:
        index = -1
        for ind, x in enumerate(run[tool]['util']['CLB Logic']):
            if x['Site Type'] == elem:
                index = ind
        if index == -1:
            raise ValueError
        elem_list.append(float(run[tool]['util']['CLB Logic'][index]['Used']))
    return elem_list


def get_x_axis(exp_dict):
    for key in exp_dict.keys():
        if len(exp_dict[key]) > 1:
            if isinstance(exp_dict[key][0], (tuple, list)):
                return list(map(lambda x: x[0], exp_dict[key])), key
            else:
                return exp_dict[key], key

if __name__ == "__main__":
    for exp in experiments:
        with open(f"results_exp_{exp[0]}.json", 'r') as f:
            json_str = f.read()
        data = json.loads(json_str)

        c4ml_syn_time_list = exp_get_time_list(data[0], tool='chisel4ml')
        hls4ml_syn_time_list = exp_get_time_list(data[0], tool='hls4ml')
        c4ml_luts_list = exp_get_elem_list(data[0], tool='chisel4ml', elem='CLB LUTs*')
        hls4ml_luts_list = exp_get_elem_list(data[0], tool='hls4ml', elem='CLB LUTs*')
        c4ml_ff_list = exp_get_elem_list(data[0], tool='chisel4ml', elem='CLB Registers')
        hls4ml_ff_list = exp_get_elem_list(data[0], tool='hls4ml', elem='CLB Registers')

        x_axis, x_axis_name = get_x_axis(exp[1])
        lut_arr = np.array([x_axis, c4ml_luts_list, hls4ml_luts_list])
        time_arr = np.array([x_axis,  c4ml_syn_time_list, hls4ml_syn_time_list])
        lut_df = pd.DataFrame(lut_arr.T, columns=[x_axis_name, 'c4ml_lut', 'hls4ml_lut'])
        time_df = pd.DataFrame(time_arr.T, columns=[x_axis_name, 'c4ml_syn_time', 'hls4ml_syn_time'])
        melt_lut_df = lut_df.melt(x_axis_name, var_name='tool', value_name='Look-Up Tables')
        melt_time_df = time_df.melt(x_axis_name, var_name='tool', value_name='Synthesis Time')

        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        if not os.path.isdir(f"plots/{exp[0]}"):
           os.makedirs(f"plots/{exp[0]}")
        lut_plot = sns.catplot(x=x_axis_name, y="Look-Up Tables", hue='tool', data=melt_lut_df, kind='point', markers=['o', 's'])
        plt.savefig(f'plots/{exp[0]}/lut_plot.png')
        time_plot = sns.catplot(x=x_axis_name, y="Synthesis Time", hue='tool', data=melt_time_df, kind='point', markers=['o', 's'])
        plt.savefig(f'plots/{exp[0]}/syn_time_plot.png')
