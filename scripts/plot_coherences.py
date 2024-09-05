import h5py
import matplotlib.pyplot as plt
import paths
from matplotlib.ticker import FixedLocator
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
PSD_FILL_ALPHA = 0.3
LABEL_FS = 'x-large'


def compute_true_coherence():
    file_path = f'{paths.data}/ET(1).txt'
    ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak10Hz_new.txt'
    Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak50Hz_new.txt'
    Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak90Hz_new.txt'
    Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    ASD_X = ET_1[:, 1] + Peak10Hz[:, 1] + Peak50Hz[:, 1]
    ASD_Y = ET_1[:, 1] + Peak10Hz[:, 1] + Peak90Hz[:, 1]
    ASD_Z = ET_1[:, 1] + Peak50Hz[:, 1] + Peak90Hz[:, 1]

    c_xy = Peak10Hz[:, 1] ** 2 / (ASD_X * ASD_Y)
    c_xz = Peak50Hz[:, 1] ** 2 / (ASD_X * ASD_Z)
    c_yz = Peak90Hz[:, 1] ** 2 / (ASD_Y * ASD_Z)
    freq_true = ET_1[:, 0]
    return dict(c_xy=c_xy, c_xz=c_xz, c_yz=c_yz, freq_true=freq_true)




def plot_coherences(coherence_file_path, ax=None, case='A'):
    # Load frequency data
    freq = np.load(f"{paths.data}/freq.npz")['arr_0']
    
    # Load coherence data from HDF5 file
    with h5py.File(coherence_file_path, 'r') as f:
        coh_med = f['ETnoise_correlated_GP_coh_median_XYZ'][:]
        coh_lower = f['ETnoise_correlated_GP_coh_lower_XYZ'][:]
        coh_upper = f['ETnoise_correlated_GP_coh_upper_XYZ'][:]
        
    # Load additional datasets for case A
    if case == 'A':
        file_path = f'{paths.data}/ET(1).txt'
        ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

        file_path = f'{paths.data}/Peak10Hz_new.txt'
        Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

        file_path = f'{paths.data}/Peak50Hz_new.txt'
        Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

        file_path = f'{paths.data}/Peak90Hz_new.txt'
        Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

        ASD_X = ET_1[:,1] + Peak10Hz[:,1] + Peak50Hz[:,1]
        ASD_Y = ET_1[:,1] + Peak10Hz[:,1] + Peak90Hz[:,1]
        ASD_Z = ET_1[:,1] + Peak50Hz[:,1] + Peak90Hz[:,1]
        
        c_xy = Peak10Hz[:,1]**2 / (ASD_X * ASD_Y)
        c_xz = Peak50Hz[:,1]**2 / (ASD_X * ASD_Z)
        c_yz = Peak90Hz[:,1]**2 / (ASD_Y * ASD_Z)
        freq_true = ET_1[:,0]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_xlim([5, 128])
    for i, label in enumerate(['XY', 'XZ', 'YZ']):
        # Plot the median coherence
        ax.plot(freq, np.squeeze(coh_med[:, i]), color=f'C{i}', linestyle="-", label=label, zorder=-10)
        ax.fill_between(freq, np.squeeze(coh_lower[:, i]), np.squeeze(coh_upper[:, i]),
                        color=f'C{i}', alpha=PSD_FILL_ALPHA, edgecolor='none', zorder=-10)
    
    # Plot c_xy, c_xz, c_yz only for Case A
    true_kwgs = dict(color='k', linestyle='--', lw=0.5)
    if case == 'A':
        ax.plot(freq_true, c_xy, **true_kwgs)
        ax.plot(freq_true, c_xz, **true_kwgs)
        ax.plot(freq_true, c_yz, **true_kwgs, label='True')
    if case == 'B':
        num_blocks = 125
        ax.axhline(1/num_blocks, **true_kwgs)


    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.5)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    return ax.get_figure()



def main():
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.6252 * 0.8))
    plot_coherences(
        coherence_file_path=f'{paths.data}/ETnoise_correlated_GP_uniform_squared_coh_XYZ.hdf5',
        ax=axes[0], case='A'
    )
    fig = plot_coherences(
        coherence_file_path=f'{paths.data}/ETnoise_uncorrelated_GP_uniform_squared_coh_XYZ.hdf5',
        ax=axes[1], case='B'
    )
    for ax in axes:
        ax.tick_params(axis='both', which='both', zorder=10)
        # turn off minor ticks
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_minor_locator(FixedLocator([]))


    legend = axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color='k',  label='Expected'),
            plt.Line2D([0], [0], color='C0', label='XY'),
            plt.Line2D([0], [0], color='C1', label='XZ'),
            plt.Line2D([0], [0], color='C2', label='YZ'),
        ],
        handlelength=1.3,  # Length of the legend line handles
        markerscale=2,  # Scale factor for marker size in the legend
        fontsize='small',
        labelspacing=0.05, columnspacing=0.15, handletextpad=0.4 ,
        loc='upper right',
        bbox_to_anchor=(1.025, 1)  # Move the legend further to the right
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(3)
    legend.legend_handles[0].set_linestyle('--')
    legend.legend_handles[0].set_linewidth(0.75)



    axes[0].text(0.02, 0.98, 'Case A', transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    axes[1].text(0.02, 0.98, 'Case B', transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')

    # axes0 set yticks to 1, 0.1, 0.01
    axes[0].set_yticks([1, 0.1, 0.01])
    axes[0].set_yticklabels(['1', '0.1', '0.01'])

    axes[1].set_yticks([1, 0.1, 0.01, 0.001])
    axes[1].set_yticklabels(['1', '0.1', '0.01', '0.001'])


    # reduce ytick label size
    for ax in axes:
        ax.tick_params(axis='both', labelsize='small')
        ax.tick_params(axis='y', which='major', pad=0.2)


    # remove verrical whitespaace beteweenn subplots
    axes[1].set_xlabel('Frequency [Hz]')
    plt.subplots_adjust(hspace=0., wspace=0.)
    for ax in axes:
        ax.set_ylabel(r'$C_{xy}$', labelpad=-8)
    fig.align_ylabels(axes, )
    fig.savefig(f'{paths.figures}/caseAB_coh.pdf')


if __name__ == '__main__':
    main()

######################################################################################################



