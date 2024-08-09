import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import paths
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator

PSD_FILL_ALPHA = 0.3
LABEL_FS = 'x-large'


def plot_coherences(coherence_file_path, ax=None):
    # #---------------------------------------------------------------------------------------------------------
    # #estimated squared coherence (median, lower, upper)

    freq = np.load(f"{paths.data}/freq.npz")['arr_0']
    with h5py.File(coherence_file_path, 'r') as f:
        coh_med = f['ETnoise_correlated_GP_coh_median_XYZ'][:]
        coh_lower = f['ETnoise_correlated_GP_coh_lower_XYZ'][:]
        coh_upper = f['ETnoise_correlated_GP_coh_upper_XYZ'][:]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_xlim([5, 128])
    for i, label in enumerate(['XY', 'XZ', 'YZ']):
        ax.plot(freq, np.squeeze(coh_med[:, i]), color=f'C{i}', linestyle="-", label=label, zorder=-10)
        ax.fill_between(freq, np.squeeze(coh_lower[:, i]), np.squeeze(coh_upper[:, i]),
                        color=[f'C{i}'], alpha=PSD_FILL_ALPHA, edgecolor='none', zorder=-10)
    ax.set_ylim(bottom=0)
    return ax.get_figure()


def main():
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.6252 * 0.8))
    plot_coherences(
        coherence_file_path=f'{paths.data}/ETnoise_correlated_GP_uniform_squared_coh_XYZ.hdf5',
        ax=axes[0]
    )
    fig = plot_coherences(
        coherence_file_path=f'{paths.data}/ETnoise_uncorrelated_GP_uniform_squared_coh_XYZ.hdf5',
        ax=axes[1]
    )
    for ax in axes:
        ax.tick_params(axis='both', which='both', zorder=10)
        # turn off minor ticks
        ax.xaxis.set_minor_locator(FixedLocator([]))
    legend = axes[0].legend(
        handlelength=1,  # Length of the legend line handles
        markerscale=2,  # Scale factor for marker size in the legend
        fontsize='small',
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(3)

    axes[0].text(0.02, 0.98, 'Case A', transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    axes[1].text(0.02, 0.98, 'Case B', transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')

    # remove verrical whitespaace beteweenn subplots
    axes[1].set_xlabel('Frequency [Hz]', fontsize=LABEL_FS, labelpad=10)
    plt.subplots_adjust(hspace=0., wspace=0.)
    fig.text(0.04, 0.5, 'Squared Coherency', va='center', ha='center', rotation='vertical', fontsize=LABEL_FS)
    plt.subplots_adjust(left=0.2)  # Adjust left margin to make room for the y-axis label
    fig.savefig(f'{paths.figures}/caseAB_coh.pdf')


if __name__ == '__main__':
    main()
