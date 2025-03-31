"""
Script for generating stability plot.

NOTE: schedule field for linear-decay run was manually renamed.
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
ax.grid(axis='both', lw=0.2, ls='--', zorder=-1)

colors = {"wsd": (0.06251441753171857, 0.35750865051903113, 0.6429065743944637),
          "cosine": (0.6943944636678201, 0.07003460207612457, 0.09231833910034601)}

for sched in ["cosine", "wsd"]:
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
ax.set_ylim(0.95, 1.16)
ax.legend()

fig.subplots_adjust(top=0.98,
bottom=0.15,
left=0.18,
right=0.99,)

if save:
    fig.savefig(f'plots/{exp_id}/stability_{metric}.pdf')


#%% training curves
ALL_METRICS = ["train_loss", "val_loss", "val_score", "learning_rate"]
SHOW_BEST = 3

best = base_df[base_df.epoch==base_df.epoch.max()].groupby('lr_schedule')['val_loss'].nsmallest(SHOW_BEST)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx),:].sort_values(["lr_schedule", "lr", "epoch"])

reds = sns.color_palette("Reds", 5)[2:]
blues = sns.color_palette("Blues", 5)[2:]
green = "#C4D6B0"

for metric in ALL_METRICS:
    fig, ax = plt.subplots(1,1,figsize=FIGSIZE11)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=-1)
    counters = {"cosine": 0, "wsd": 0, "linear-decay": 0}

    for id in df1.id.unique():
        this = df1[df1.id == id]
        this_sched = this.lr_schedule.values[0]
        this_lr = this.lr.values[0]

        if this_sched=="cosine":
            col = reds[counters[this_sched]]
        elif this_sched=="wsd":
            col = blues[counters[this_sched]]
        else:
            col = green

        counters[this_sched] += 1

        if metric in ["learning_rate", "train_loss"]:
            ax.plot(this.epoch,
                    this[metric],
                    c=col,
                    label=f"{this_sched}, " + r"$\gamma = %.2f$" % this_lr
            )

        else:
            # smoothed line
            ax.plot(this.epoch,
                this[metric].rolling(5).mean(),
                c=col,
                label=f"{this_sched}, " + r"$\gamma = %.2f$" % this_lr,
                zorder=2
            )
            # original data
            ax.plot(this.epoch,
                this[metric],
                c=col,
                lw=0.5,
                alpha=0.6,
                zorder=5
            )

    ax.set_xlabel(r'Epoch')
    ax.set_ylabel(ylabel_map[metric])
    if metric == "train_loss":
        ax.set_ylim(1.5, 3.8)
    elif metric == "val_loss":
        ax.set_ylim(0.9, 3.8)
    elif metric == "val_score":
        ax.set_ylim(0.37, 0.82)

    ax.legend(loc="upper right" if metric != "val_score" else "lower left", fontsize=8, ncol=2, framealpha=0.9)

    fig.subplots_adjust(top=0.98,
    bottom=0.15,
    left=0.17,
    right=0.99,)
    if save:
        fig.savefig(f'plots/{exp_id}/{metric}.pdf')
    