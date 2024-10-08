import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paths
from matplotlib.patches import Patch
from matplotlib import ticker

FIG_Y = 2.2
TICKLBL_PAD = 0.6
FIGKWGS = dict(dpi=300, bbox_inches='tight', pad_inches=0)


def add_violin_plot(ax, data, positions, colors, legend_labels=None, legend_loc='upper left'):
    """
    Adds a violin plot to the given axis.

    Parameters:
    - ax: The Matplotlib axis to which the violin plot will be added.
    - data: List of datasets for the violin plots.
    - positions: List of positions for the violins.
    - colors: List of colors for the violins.
    - legend_labels: List of labels for the legend.
    - legend_loc: Location of the legend on the plot.
    """
    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True)

    # Customize violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_linewidth(0)
        pc.set_alpha(0.5)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolors(colors)
        vp.set_linewidth(1.5)

    # Add legend
    if legend_labels:
        legend_patches = [Patch(facecolor=colors[i], label=label) for i, label in enumerate(legend_labels)]
        ax.legend(
            handles=legend_patches, fontsize='small',
            handletextpad=0.15, labelspacing=0.05, handlelength=0.75,
            loc=legend_loc, bbox_to_anchor=(-0.015, 1.07)
        )

    # Customize axes
    ax.set_xticks([1.25, 3.25, 5.25])
    ax.set_xticklabels(['256', '512', '1024'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis='x', length=0)
    # redice space between ticklabels and axis
    ax.tick_params(axis='x', pad=TICKLBL_PAD)
    ax.tick_params(axis='y', pad=TICKLBL_PAD)



def get_speedups(data):
    d = data.T
    return np.array([[
        d[1] / d[0],
        d[3] / d[2],
        d[5] / d[4]
    ]]).T

def load_timing_violin_data():
    # Load data
    violin_data1 = pd.read_csv(f"{paths.data}/time_var2_SGVB_VNPC.csv").values
    violin_data2 = pd.read_csv(f"{paths.data}/time_vma1_SGVB_VNPC.csv").values
    violin_data1 = get_speedups(violin_data1)
    violin_data2 = get_speedups(violin_data2)

    return np.array([
        violin_data1[:, 0],
        violin_data2[:, 0],
        violin_data1[:, 1],
        violin_data2[:, 1],
        violin_data1[:, 2],
        violin_data2[:, 2]
    ])[..., 0].T




def main():
    # First panel: Original violin plot
    colors = ['C0', 'C1'] * 3
    legend_labels = ['VNPC', 'SGVB']

    positions = [1, 1.5, 3, 3.5, 5, 5.5]

    # Load data
    violin_data1 = pd.read_csv(f"{paths.data}/L2_errors_var2.csv").values
    violin_data2 = pd.read_csv(f"{paths.data}/L2_errors_vma1.csv").values
    violin_data3 = load_timing_violin_data()

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.5, FIG_Y))
    # Annotate subplots
    kwgs = dict(fontsize=10, verticalalignment='top', horizontalalignment='left', zorder=10)
    axs[0].text(0.8, 0.95, 'VAR(2)', transform=axs[0].transAxes, **kwgs)
    axs[1].text(0.8, 0.95, 'VMA(1)', transform=axs[1].transAxes, **kwgs)
    axs[1].set_xlabel(r'$n$', labelpad=0.5)
    lbl = r'$L_2$ error'
    # axs[0].set_ylabel(lbl, labelpad=0.9)
    # axs[1].set_ylabel(lbl, labelpad=0.9)
    # set common y label
    fig.text(0.0155, 0.5, lbl, va='center', rotation='vertical', fontsize=15)
    for ax in axs:
        ax.set_ylim(0, .4)
        ax.set_yticks([0.1, 0.3])
        # zero ytick len
        ax.tick_params(axis='x', length=0)
    # add a common y label spanning both subplots

    add_violin_plot(axs[0], violin_data1, positions, colors, legend_labels)
    add_violin_plot(axs[1], violin_data2, positions, colors)
    plt.subplots_adjust(hspace=0.)  # Adjust as needed
    plt.savefig(f"{paths.figures}/sim_l2error_violins.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.5, FIG_Y/2))
    add_violin_plot(axs, violin_data3, positions, ['C3', 'C4'] * 3, ['VAR(2)', 'VMA(1)'], legend_loc='upper left')
    # axs.text(0.02, 0.98, 'c)', transform=axs.transAxes, **kwgs)
    axs.set_ylabel(r'Speed up', labelpad=0.9)
    axs.set_yticks([1, 50, 100])
    axs.set_xlabel(r'$n$', labelpad=0.5)
    axs.set_ylim(bottom=1)
    plt.subplots_adjust(hspace=0.)  # Adjust as needed
    plt.savefig(f"{paths.figures}/sim_speed_violins.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == "__main__":
    main()
