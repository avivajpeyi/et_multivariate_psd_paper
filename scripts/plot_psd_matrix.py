import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import paths
from matplotlib.lines import Line2D
import pandas as pd

PSD_FILL_ALPHA = 0.5
SYM_THRESH = 1e-49
TICK_LN = 5
THICKNESS = 2

LABELS = ['X', 'Y', 'Z']
TRUE_STYLE = dict(color='k', lw=0.8, ls="-", alpha=0.6)

def plot_et_matrix(
        channel_pth,
        matrix_file_path,
        psd_plot_fname=None,
        psd_col = "C0",
        axes=None,
        label=""
):
    channels = []
    for i, c in enumerate('XYZ'):
        with h5py.File(channel_pth.format(c), 'r') as f:
            channels.append(f[f'E{i + 1}:STRAIN'][:])

    # ----------------------------------------------------------------------------------------------
    channels = np.column_stack(channels)

    # ---------------------------------------------------------------------------------------------------------
    # estimated psd (median, lower, upper)
    with h5py.File(matrix_file_path, 'r') as f:
        spec_mat_median = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
        spec_mat_lower = f['ETnoise_correlated_GP_spec_mat_lower_XYZ'][:]
        spec_mat_upper = f['ETnoise_correlated_GP_spec_mat_upper_XYZ'][:]
        
    # upload the true psd data
    file_path =f'{paths.data}/\ET(1).txt'
    ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak10Hz_new.txt'
    Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak50Hz_new.txt'
    Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak90Hz_new.txt'
    Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    x_channel_real = ((ET_1[:,1] + Peak10Hz[:,1] + Peak50Hz[:,1])**2)/2
    y_channel_real = ((ET_1[:,1] + Peak10Hz[:,1] + Peak90Hz[:,1])**2)/2
    z_channel_real = ((ET_1[:,1] + Peak50Hz[:,1] + Peak90Hz[:,1])**2)/2

    q = 10 ** 22 / 1.0
    time_interval = 2000
    nchunks = 125
    required_part = 128

    Ts = 1 / (channels.shape[0] / time_interval)
    freq_original = np.fft.fftfreq(int(np.size(channels, 0) / nchunks), Ts)

    n = int(np.size(channels, 0) / nchunks)
    if np.mod(n, 2) == 0:
        # n is even
        freq_original = freq_original[0:int((n / 2))]
    else:
        # n is odd
        freq_original = freq_original[0:int((n - 1) / 2)]

    total_len = channels.shape[0]
    freq_range = total_len / time_interval / 2
    freq = freq_original[0:int(required_part / freq_range * freq_original.shape[0])]
    freq_true = ET_1[:,0]

    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True)


    for i in range(3):
        for j in range(3):

            axes[i, j].text(0.05, 0.95, f"{LABELS[i]}{LABELS[j]}", transform=axes[i, j].transAxes,
                            horizontalalignment='left', verticalalignment='top', fontsize=14)


            if i == j:
                f, Pxx_den0 = signal.periodogram(channels[:, i], fs=channels.shape[0] / time_interval)
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:] / 2
                axes[i, j].plot(f, Pxx_den0, marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3, zorder=-10)

                axes[i, j].plot(freq, spec_mat_median[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5), linewidth=1,
                                color=psd_col, linestyle="-")
                axes[i, j].fill_between(freq, spec_mat_lower[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5),
                                        spec_mat_upper[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5),
                                        color=psd_col, alpha=PSD_FILL_ALPHA)

                axes[i, j].text(0.95, 0.95, r'$f_{{{}, {}}}$'.format(i + 1, i + 1), transform=axes[i, j].transAxes,
                                horizontalalignment='right', verticalalignment='top', fontsize=14)

                axes[i, j].set_xlim([5, 128])
                axes[i, j].set_ylim([10 ** (-52), 10 ** (-46)])
                axes[i, j].set_yscale('log')


                if i == 0:
                    axes[i, j].plot(freq_true, x_channel_real, **TRUE_STYLE)
                elif i == 1:
                    axes[i, j].plot(freq_true, y_channel_real,**TRUE_STYLE)
                elif i == 2:
                    axes[i, j].plot(freq_true, z_channel_real, **TRUE_STYLE)


            elif i < j:

                axes[i, j].plot(freq, np.real(spec_mat_median[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                linewidth=1, color=psd_col, linestyle="-")
                axes[i, j].fill_between(freq, np.real(spec_mat_lower[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                        np.real(spec_mat_upper[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                        color=psd_col, alpha=PSD_FILL_ALPHA)

                y = np.apply_along_axis(np.fft.fft, 0, channels)
                n = channels.shape[0]
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0:int(n / 2)]
                else:
                    # n is odd
                    y = y[0:int((n - 1) / 2)]
                y = y / np.sqrt(n)
                # y = y[1:]
                cross_spectrum_fij = y[:, i] * np.conj(y[:, j])

                axes[i, j].plot(f, np.real(cross_spectrum_fij) / (freq_original[-1] / 0.5),
                                marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3, zorder=-10)

                axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i + 1, j + 1), transform=axes[i, j].transAxes,
                                horizontalalignment='right', verticalalignment='top', fontsize=14)

                axes[i, j].set_xlim([5, 128])

                axes[i, j].set_yscale('symlog', linthresh=SYM_THRESH)
            else:

                axes[i, j].plot(freq, np.imag(spec_mat_median[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                linewidth=1, color=psd_col, linestyle="-")
                axes[i, j].fill_between(freq, (spec_mat_lower[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                        (spec_mat_upper[..., i, j]) / (q) ** 2 / (freq_original[-1] / 0.5),
                                        color=psd_col, alpha=PSD_FILL_ALPHA)

                y = np.apply_along_axis(np.fft.fft, 0, channels)
                n = channels.shape[0]
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0:int(n / 2)]
                else:
                    # n is odd
                    y = y[0:int((n - 1) / 2)]
                y = y / np.sqrt(n)
                # y = y[1:]
                cross_spectrum_fij = y[:, i] * np.conj(y[:, j])

                axes[i, j].plot(f, np.imag(cross_spectrum_fij) / (freq_original[-1] / 0.5),
                                marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3, zorder=-10)

                axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i + 1, j + 1), transform=axes[i, j].transAxes,
                                horizontalalignment='right', verticalalignment='top', fontsize=14)

                axes[i, j].set_xlim([5, 128])

                axes[i, j].set_yscale('symlog', linthresh=SYM_THRESH)
    axes[2, 1].set_xlabel('Frequency [Hz]', ha='center', va='center', fontsize=20, labelpad=10)
    axes[1, 0].set_ylabel('Strain PSD [1/Hz]', ha='center', va='center', rotation='vertical', fontsize=20, labelpad=12.5)

    axes[0, 2].legend(
        handles=[
            Line2D([], [], color='lightgray', label='Periodogram', lw=3),
            Line2D([], [], color=psd_col, label=label, lw=3),
            Line2D([],[],  label='True PSD', **TRUE_STYLE)
        ],
        loc='lower right', fontsize=10)



    diag_ylims = (min([axes[i, i].get_ylim()[0] for i in range(3)]), max([axes[i, i].get_ylim()[1] for i in range(3)]))
    off_ylims = (min([axes[i, j].get_ylim()[0] for i in range(3) for j in range(3) if i != j]),
                 max([axes[i, j].get_ylim()[1] for i in range(3) for j in range(3) if i != j]))

    ###  Adjust axes
    # remove space ebtween subplots
    plt.subplots_adjust(hspace=0., wspace=0.)
    for rowi in range(3):
        for colj in range(3):
            ax = axes[rowi, colj]
            ax.tick_params('both', length=0, width=0, which='minor', bottom=False, top=False, left=False, right=False)
            ax.tick_params(axis='y', direction='in', pad=-10)
            ax.yaxis.set_tick_params(which='both', labelleft=True, zorder=3)
            for label in ax.get_yticklabels():
                label.set_horizontalalignment('left')

            if rowi == colj:
                # increase ax-spine linewidth
                for spine in ax.spines.values():
                    spine.set_linewidth(THICKNESS)
                    spine.set_zorder(10)
                ax.set_ylim(diag_ylims)
                ax.set_yticks([1e-51, 1e-49, 1e-47])
                ax.tick_params('both', length=TICK_LN, width=THICKNESS, which='major')

            else:
                ax.set_ylim(off_ylims)
                ax.patch.set_color('lightgray')  # or whatever color you like
                ax.patch.set_alpha(.3)
                ax.tick_params('both', length=TICK_LN, width=1, which='major')

    # plt.show()
    if psd_plot_fname:
        plt.savefig(psd_plot_fname, dpi=300)
    else:
        return axes


def main():
    plot_et_matrix(
        channel_pth=str(paths.data) + "/{}_ETnoise_GP.hdf5",
        matrix_file_path=f'{paths.data}/ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5',
        psd_plot_fname=f'{paths.figures}/caseA_psd.pdf',
        label="Case A PSD"
    )

    axes = plot_et_matrix(
        channel_pth=str(paths.data) + "/{}_ETnoise_GP_uncorr.hdf5",
        matrix_file_path=f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5',
        psd_col="C1",
        label="Case B PSD"
    )
    axes = plot_et_matrix(
        channel_pth=str(paths.data) + "/{}_ETnoise_GP_uncorr.hdf5",
        matrix_file_path=f'{paths.data}/ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5',
        psd_col="C2",
        axes=axes,
        label="Case C PSD"
    )


    axes[0, 2].legend(
        handles=[
            Line2D([], [], color='lightgray', label='Periodogram', lw=3),
            Line2D([], [], color="C1", label="Case B PSD", lw=3),
            Line2D([], [], color="C2", label="Case C PSD", lw=3),
    Line2D([], [], label='True PSD', **TRUE_STYLE)
        ],
        loc='lower right', fontsize=10)

    axes[0,0].get_figure().savefig(f'{paths.figures}/caseBC_psd.pdf')

if __name__ == '__main__':
    main()