import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import argparse
from plot.plotting_utilities import *

sns.set_style("whitegrid")
sns.set()
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]

project_dir = this_directory + '../../'

parser = argparse.ArgumentParser(description='Bar plot that shows the memory footprint of H5PY and Numpy.',
                                 usage='python {} -i results/'.format(this_filename))

parser.add_argument('-i', '--inputfile', type=str, default='results/run3/results.csv',
                    help='The directory with the results.')

parser.add_argument('-o', '--outdir', type=str, default=None,
                    help='The directory to store the plots.'
                    'Default: In a plots directory inside the input results directory.')

parser.add_argument('-c', '--cases', type=str, nargs='+', default=['numpy', 'h5py'],
                    help='A list of the cases to plot. Default: numpy h5py')

parser.add_argument('-s', '--show', action='store_true',
                    help='Show the plots.')

gconfig = {

    'label': {
        'double': 'Base',
        'single': 'F32',
        'singleSRP': 'F32-SRP',
        'doubleSRP': 'SRP',
        'singleRDS': 'F32-RDS',
        'doubleRDS': 'RDS',
    },
    'colors': {
        'Base': 'tab:orange',
        'F32': 'tab:orange',
        'F32-SRP': 'tab:blue',
        'SRP': 'tab:blue',
        'F32-RDS': 'tab:green',
        'RDS': 'tab:green',
    },
    'hatches': {
        'Base': '',
        'F32': 'xx',
        'F32-SRP': 'xx',
        'SRP': '',
        'F32-RDS': 'xx',
        'RDS': '',
    },

    'x_name': 'n',
    'y_name': 'avg_time(sec)',
    'xlabel': {
        'xlabel': '8 Nodes 160 Cores'
    },
    'ylabel': 'Norm. Runtime',
    'title': {
        # 's': '',
        'fontsize': 10,
        'y': .96,
        'x': 0.1,
        'fontweight': 'bold',
    },
    'figsize': [5, 2.1],
    'annotate': {
        'fontsize': 10,
        'textcoords': 'data',
        'va': 'bottom',
        'ha': 'center'
    },
    'xticks': {'fontsize': 10, 'rotation': '0', 'fontweight': 'bold'},
    'ticks': {'fontsize': 10, 'rotation': '0'},
    'fontsize': 10,
    'legend': {
        'loc': 'upper left', 'ncol': 9, 'handlelength': 1.6, 'fancybox': True,
        'framealpha': 0., 'fontsize': 10, 'labelspacing': 0, 'borderpad': 0.5,
        'handletextpad': 0.2, 'borderaxespad': 0.1, 'columnspacing': 0.3,
        'bbox_to_anchor': (-0.01, 1.15)
    },
    'subplots_adjust': {
        'wspace': 0.0, 'hspace': 0.1, 'top': 0.93
    },
    'tick_params': {
        'pad': 1, 'top': 0, 'bottom': 0, 'left': 1,
        'direction': 'out', 'length': 3, 'width': 1,
    },
    'fontname': 'DejaVu Sans Mono',
    'ylim': [0.2, 1.105],
    'yticks': [0.2, 0.4, 0.6, 0.8, 1],
    'outfiles': [
        '{}/{}-{}.png',
        '{}/{}-{}.pdf'
    ],

}

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = os.path.join(os.path.basename(args['inputfile']), 'plots')
    if not os.path.exists(args['outdir']):
        os.makedirs(args['outdir'])

    df = pd.read_csv(args['inputfile'], delimiter='\t', header=0, index_col=False)

    h5py_df = df[df['exe'] == 'h5py']
    numpy_df = df[df['exe'] == 'numpy']

    # for N in df.N.unique():
    for N in [1, 2]:
        temp_df = df[df.N == N]
        mem_arr = temp_df.mem.unique()
        # for mem in mem_arr:
        for mem in [1000]:
            fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=[5,3])
            offset = 0
            step = 1
            width = 1. / 5
            for exe in df.exe.unique():
                data = df[(df.N == N) & (df.mem == mem) & (df.exe == exe)]
                x = data.wpn.to_numpy()
                y1 = data.maxMem0.to_numpy()
                y2 = data.maxMemN.to_numpy()
                yerr = data.maxMemN_std.to_numpy()
                plt.bar(np.arange(len(x))+offset, y1, label=f'{exe}-master-N{N}-mem{mem}',
                        width=width, edgecolor='0',
                        hatch=None, color=None)
                offset += width
                plt.bar(np.arange(len(x))+offset, y2, label=f'{exe}-worker-N{N}-mem{mem}',
                        width=width, edgecolor='0',
                        hatch=None, color=None)
                offset += width
            plt.xticks(np.arange(len(x))+2*width, x)
            plt.xlabel('Number of workers')
            plt.ylabel('Memory Footprint (MB)')
            plt.gca().tick_params(**gconfig['tick_params'])

            plt.legend()
            plt.tight_layout()
            for file in gconfig['outfiles']:
                file = file.format(args['outdir'], this_filename[:-3], f'N{N}-mem{mem}')
                save_and_crop(fig, file, dpi=600, bbox_inches='tight')
            if args['show']:
                plt.show()
            plt.close()