"""
Script for generating stability plot.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse

from stepback.record import Record
from stepback.utils import get_output_filenames
from stepback.plotting import plot_stability, plot_step_sizes


exp_id = 'vit_tiny_patch16_224'
save = False

output_names = get_output_filenames(exp_id, output_dir='output/clean')
############################################################

#%%
#%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%
R = Record(output_names, output_dir='output/clean/')

# rescale from [0,100] to [0,1]
for c in ['val_top1', 'val_top5']:
    R.raw_df[c] *= 1/100

# rename top1-->score
R.raw_df.rename(columns={'val_top1': 'val_score'},
                inplace=True)

# rebuild base df
R.base_df = R._build_base_df(agg='mean')


base_df = R.base_df                                 # base dataframe for all plots
id_df = R.id_df                                     # dataframe with the optimizer setups that were run

# check that 3 runs are complete for all settings
assert R.raw_df.groupby(['id', 'epoch']).size().min() == 3
assert R.raw_df.groupby(['id', 'epoch']).size().max() == 3

#%%


FIGSIZE = (4.8,3.2)

fig, ax =  plot_stability(R, 
                          score='val_score', 
                          xaxis='lr', 
                          sigma=1, 
                          legend=None,
                          ignore_columns=['weight_decay'], 
                          ylim=(0.6,0.75), 
                          figsize=FIGSIZE, 
                          save=False)

if save:
    fig.savefig(f'plots/{exp_id}/stability_lr_val_score.pdf')

fig, ax =  plot_stability(R, 
                          score='val_loss', 
                          xaxis='lr', 
                          sigma=1, 
                          legend=None,
                          ignore_columns=['weight_decay'], 
                          log_scale=False,
                          ylim=(0,5), 
                          figsize=FIGSIZE, 
                          save=False)

if save:
    fig.savefig(f'plots/{exp_id}/stability_lr_val_loss.pdf')

#%%

fig, axs = plot_step_sizes(R, method='momo-adam', grid=(2,2), start=None, stop=None, save=False)

for ax in axs.ravel():
    ax.set_xlim(0,10)

if save:
    fig.savefig(f'plots/{exp_id}/step_sizes_momo-adam.png', dpi=500)
