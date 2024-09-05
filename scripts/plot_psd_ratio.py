import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import paths
from matplotlib.lines import Line2D
import pandas as pd
from scipy import interpolate as interp


PSD_FILL_ALPHA = 0.5
SYM_THRESH = 1e-49
TICK_LN = 5
THICKNESS = 2

LABELS = ['X', 'Y', 'Z']
TRUE_STYLE = dict(color='k', lw=0.8, ls="-", alpha=0.6)


def load_posterior_matrix(matrix_file_path):
    # estimated psd (median, lower, upper)
    with h5py.File(matrix_file_path, 'r') as f:
        spec_mat_median = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
        spec_mat_lower = f['ETnoise_correlated_GP_spec_mat_lower_XYZ'][:]
        spec_mat_upper = f['ETnoise_correlated_GP_spec_mat_upper_XYZ'][:]
    return spec_mat_median, spec_mat_lower, spec_mat_upper


def load_raw_data(channel_pth):
    channels = []
    for i, c in enumerate('XYZ'):
        with h5py.File(channel_pth.format(c), 'r') as f:
            channels.append(f[f'E{i + 1}:STRAIN'][:])
    channels = np.column_stack(channels)
    return channels


def load_true_psd():
    # upload the true psd data
    file_path = f'{paths.data}/ET(1).txt'
    ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak10Hz_new.txt'
    Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak50Hz_new.txt'
    Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak90Hz_new.txt'
    Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    x_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak50Hz[:, 1] ** 2)) / 2
    y_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2
    z_channel_real = ((ET_1[:, 1] ** 2 + Peak50Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2

    return ET_1, x_channel_real, y_channel_real, z_channel_real


def load_freq():
    q = 10 ** 22 / 1.0
    time_interval = 2000
    nchunks = 125
    required_part = 128

    channel_pth = str(paths.data) + "/{}_ETnoise_GP_uncorr.hdf5"
    channels = load_raw_data(channel_pth)
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
    norm_factor = (q) ** 2 / (freq_original[-1] / 0.5)
    return freq, freq_original, norm_factor


def plot_et_matrix(
        channel_pth,
        matrix_file_path,
        psd_plot_fname=None,
        psd_col="C0",
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
    file_path = f'{paths.data}/ET(1).txt'
    ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak10Hz_new.txt'
    Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak50Hz_new.txt'
    Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = f'{paths.data}/Peak90Hz_new.txt'
    Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    x_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak50Hz[:, 1] ** 2)) / 2
    y_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2
    z_channel_real = ((ET_1[:, 1] ** 2 + Peak50Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2

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
    freq_true = ET_1[:, 0]

    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True)


    ## PLOT PSD
    i = 0
    f, Pxx_den0 = signal.periodogram(channels[:, i], fs=channels.shape[0] / time_interval)
    f = f[1:]
    Pxx_den0 = Pxx_den0[1:] / 2

    spec_mat_median = spec_mat_median[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5)
    spec_mat_lower = spec_mat_lower[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5)
    spec_mat_upper = spec_mat_upper[..., i, i] / (q) ** 2 / (freq_original[-1] / 0.5)


    # ax = axes[0]
    # ax.plot(f, Pxx_den0, marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3,
    #                 zorder=-10)
    # ax.plot(freq, spec_mat_median, linewidth=1,
    #                 color=psd_col, linestyle="-")
    # ax.fill_between(freq, spec_mat_lower,
    #                         spec_mat_upper,
    #                         color=psd_col, alpha=PSD_FILL_ALPHA)
    #
    # ax.set_xlim([5, 128])
    # ax.set_ylim([10 ** (-52), 10 ** (-46)])
    # ax.set_yscale('log')
    # ax.plot(freq_true, x_channel_real, **TRUE_STYLE)
    # ax.set_ylabel('PSD [1/Hz]')


    ## PLOT ERROR

    true_sxx = interp.interp1d(freq_true, x_channel_real, kind='cubic', bounds_error=False, fill_value='extrapolate')(freq)
    # compute absolute relative error at each frequency
    error = np.exp(np.log(np.array([
        spec_mat_lower  ,
        spec_mat_median ,
        spec_mat_upper ,
    ]))  - np.log(true_sxx) )


    ax = axes
    ax.plot(freq, error[1], color=psd_col, label=label,  alpha=0.6, )
    ax.fill_between(freq, error[0], error[2], color=psd_col, alpha=0.2, lw=0)
    ax.set_ylabel('Estimated PSD / True PSD', fontsize=12)
    ax.set_ylim(0.6, 1.4)

    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim([5, 128])

    # dashed line along 1
    ax.axhline(1, color='k', linestyle='--', lw=0.5)


    plt.tight_layout()

    # plt.show()
    if psd_plot_fname:
        plt.savefig(psd_plot_fname, dpi=300)
    else:
        return axes


def main():
    axes = plot_et_matrix(
        channel_pth=str(paths.data) + "/{}_ETnoise_GP.hdf5",
        matrix_file_path=f'{paths.data}/ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5',
        psd_col="C0",
        label="Case A PSD"
    )
    axes = plot_et_matrix(
        channel_pth=str(paths.data) + "/{}_ETnoise_GP_uncorr.hdf5",
        matrix_file_path=f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5',
        axes=axes,
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


    axes.legend(
        handles=[
            Line2D([], [], color="C0", label="Case A", lw=3),
            Line2D([], [], color="C1", label="Case B", lw=3),
            Line2D([], [], color="C2", label="Case C", lw=3),
        ],
        loc='upper right', fontsize=10)

    plt.tight_layout()
    axes.get_figure().savefig(f'{paths.figures}/Sxx_ratio.pdf')



if __name__ == '__main__':
    main()


