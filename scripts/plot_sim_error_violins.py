import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paths
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker




def add_violin_plot(ax, data, positions, colors, legend_labels=None, legend_loc='upper right'):
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
        ax.legend(handles=legend_patches, fontsize='small', handletextpad=0.25, labelspacing=0.05, handlelength=1)

    # Customize axes
    ax.set_xticks([1.25, 3.25, 5.25])
    ax.set_xticklabels(['256', '512', '1024'])
    ax.set_ylabel(r'$L_2$ error')
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())


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
    legend_labels = ['VNPC', 'VB']

    positions = [1, 1.5, 3, 3.5, 5, 5.5]

    # Load data
    violin_data1 = pd.read_csv(f"{paths.data}/L2_errors_var2.csv").values
    violin_data2 = pd.read_csv(f"{paths.data}/L2_errors_vma1.csv").values
    violin_data3 = load_timing_violin_data()

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    # Annotate subplots
    kwgs = dict(fontsize=10, verticalalignment='top', horizontalalignment='left', zorder=10)
    axs[0].text(0.02, 0.98, 'a)', transform=axs[0].transAxes, **kwgs)
    axs[1].text(0.02, 0.98, 'b)', transform=axs[1].transAxes, **kwgs)
    axs[2].text(0.02, 0.98, 'c)', transform=axs[2].transAxes, **kwgs)
    axs[2].set_xlabel("Data Length")
    add_violin_plot(axs[0], violin_data1, positions, colors, legend_labels)
    add_violin_plot(axs[1], violin_data2, positions, colors)
    add_violin_plot(axs[2], violin_data3, positions, ['C3', 'C4'] * 3, ['VAR', 'VMA'])
    axs[2].set_ylabel(r'$\times$ Speed', labelpad=0.9)
    plt.subplots_adjust(hspace=0.)  # Adjust as needed
    plt.savefig(f"{paths.figures}/sim_error_violins.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
