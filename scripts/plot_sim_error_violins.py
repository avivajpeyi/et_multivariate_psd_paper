import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paths
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



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
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.5)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolors(colors)
        vp.set_linewidth(5)

    # Add legend
    if legend_labels:
        legend_patches = [Patch(facecolor=f'C{i}', label=label) for i, label in enumerate(legend_labels)]
        ax.legend(handles=legend_patches, loc=legend_loc)

    # Customize axes
    ax.set_xticks([1.25, 3.25, 5.25])
    ax.set_xticklabels(['256', '512', '1024'])
    ax.set_ylabel('L2 Errors')



# First panel: Original violin plot
colors = ['C0', 'C1'] * 3
legend_labels = ['VNPC', 'VB']

positions = [1, 1.5, 3, 3.5, 5, 5.5]

# Load data
violin_data1 = pd.read_csv(f"{paths.data}/L2_errors_var2.csv").values
violin_data2 = pd.read_csv(f"{paths.data}/L2_errors_vma1.csv").values


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
add_violin_plot(axs[0], violin_data1, positions, colors, legend_labels)
add_violin_plot(axs[1], violin_data2, positions, colors)


# Annotate subplots
axs[0].text(0.02, 0.98, '(a)', transform=axs[0].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')
axs[1].text(0.02, 0.98, '(b)', transform=axs[1].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')
axs[1].set_xlabel("Data Length")
plt.subplots_adjust(hspace=0.)  # Adjust as needed
plt.savefig(f"{paths.figures}/sim_error_violins_with_panel.pdf", dpi=300)
plt.close()
