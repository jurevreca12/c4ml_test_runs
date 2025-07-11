import os
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main import EXPERIMENTS
from main import get_work_dir

key_to_name_dict = {
    'bitwidth': 'Bitwidth',
    'prune_rate': 'Prune Rate',
}

def get_x_axis(exp_dict):
    for key in exp_dict.keys():
        if len(exp_dict[key]) > 1:
            if isinstance(exp_dict[key][0], (tuple, list)):
                return list(map(lambda x: x[0], exp_dict[key])), key_to_name_dict[key]
            else:
                return exp_dict[key], key_to_name_dict[key]



if __name__ == '__main__':
    for exp in EXPERIMENTS[12:]:
        x_axis, x_axis_name = get_x_axis(exp[0])
        exp_name = exp[2]
        exp_base = f"/circuits/{exp_name}/"
        exp_keys = exp[0].keys()
        feat_list = list(itertools.product(*exp[0].values()))
        acc_list = []
        for feat in feat_list:
            work_dir = get_work_dir(exp_keys, feat, base=exp_base)
            with open(work_dir + '/acc.log', 'r') as f:
                lines = f.readlines()
                acc = float(lines[1])
            acc_list.append(acc)
        df = pd.DataFrame({
            x_axis_name: x_axis,
            'Accuracy': acc_list
        })
        sns.set_style("darkgrid", {
                "axes.facecolor": ".9"
            }
        )
        sns.set(font_scale=1.3)
        sns.catplot(
            x=x_axis_name,
            y='Accuracy',
            data=df,
            kind='point',
        )
        plt.ylim(0, 1)
        plt.savefig(f'plots/{exp_name}/acc_plot.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
