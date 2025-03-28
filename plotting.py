"""
Script for generating stability plot.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from stepback.record import Record
from stepback.utils import get_output_filenames, merge_subfolder


#%% for merging all runs into a single file

# merge_subfolder("schedules", fname='clean/resnet50', output_dir='output/')

#%%
exp_id = 'resnet50'
save = False

output_names = get_output_filenames(exp_id, output_dir='output/clean')

#%%
# %matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend',fontsize=10)

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

base_df = base_df.astype({"lr": float})

#%% stability

MAX_EPOCH = 99
FIGSIZE11 = (4, 3)

ylabel_map = {"val_loss": "Validation loss",
             "train_loss": "Train loss",
             "val_score": "Validation top-1",
             "learning_rate": r"Learning rate $\gamma \eta_t$"
}


metric = "val_loss"
final = base_df[base_df.epoch==MAX_EPOCH].groupby(["lr_schedule", "lr"], as_index=False)[metric].min()

fig, ax = plt.subplots(1,1,figsize=FIGSIZE11)

colors = {"wsd": (0.06251441753171857, 0.35750865051903113, 0.6429065743944637),
          "cosine": (0.6943944636678201, 0.07003460207612457, 0.09231833910034601)}

for sched in final["lr_schedule"].unique():
    this = final[final["lr_schedule"] == sched]
    this = this.sort_values("lr")
    ax.plot(this.lr.astype("float"),
            this[metric],
            c=colors[sched],
            lw=2,
            marker="o",
            label=sched
    )

ax.set_xlabel(r'Base learning rate $\gamma$')
ax.set_xscale("log", base=2)
ax.set_xticks([0.05, 0.1, 0.2, 0.4], [0.05, 0.1, 0.2, 0.4])
ax.set_ylabel(ylabel_map[metric])
ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
ax.legend()

fig.subplots_adjust(top=0.98,
bottom=0.15,
left=0.18,
right=0.99,)

if save:
    fig.savefig(f'plots/{exp_id}/stability_{metric}.pdf')


#%% training curves

reds = sns.color_palette("Reds", 5)[2:]
blues = sns.color_palette("Blues", 5)[2:]

metric = "val_loss"

fig, ax = plt.subplots(1,2,figsize=FIGSIZE11)

best = base_df[base_df.epoch==base_df.epoch.max()].groupby('lr_schedule')['val_loss'].nsmallest(3)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:].sort_values(["id", "epoch", "lr"])

counters = {"cosine": 0, "wsd": 0}

for id in df1.id.unique():
    this = df1[df1.id == id]
    this_sched = this.lr_schedule.values[0]
    
    col = reds[counters[this_sched]] if this_sched=="cosine" else blues[counters[this_sched]]
    counters[this_sched] += 1

    ax.plot(this.epoch,
            this[metric],
            c=col
    )

ax.set_xlabel(r'Epoch')
ax.set_ylabel(ylabel_map[metric])
if metric == "train_loss":
    ax.set_ylim(1.5, 3.8)
elif metric == "val_loss":
    ax.set_ylim(0.9, 3.8)

ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

fig.subplots_adjust(top=0.98,
bottom=0.15,
left=0.18,
right=0.99,)
if save:
    fig.savefig(f'plots/{exp_id}/{metric}.pdf')