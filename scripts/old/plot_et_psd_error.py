import numpy as np
import matplotlib.pyplot as plt
import paths
from plot_psd_matrix import load_true_psd, load_posterior_matrix, load_freq
import scipy.interpolate as interp

# load true Case A, B, C's S_xx
# divide each S_xx by the true S_xx -- the ratio of the estimated to the true
# plot the ratio and 90% CI for each case



def load_Sxx(case)->np.ndarray:
    """load Case A, B, C's S_xx

    Returns:
        np.ndarray: shape (n_samples, n_freq)
    """
    if case == 'A':
        s = load_posterior_matrix(f'{paths.data}/ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5')
    elif case == 'B':
        s = load_posterior_matrix(f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5')
    elif case == 'C':
        s = load_posterior_matrix(f'{paths.data}/ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5')

    spec_mat_median, spec_mat_lower, spec_mat_upper = s
    spec_mat_median = spec_mat_median[..., 0, 0]
    spec_mat_lower = spec_mat_lower[..., 0, 0]
    spec_mat_upper = spec_mat_upper[..., 0, 0]

    freq, freq_orig, norm = load_freq()
    mask = (freq >= 5) & (freq <= 128)
    freq = freq[mask]
    spec = np.real(np.array([
        spec_mat_lower[mask],
        spec_mat_median[ mask],
        spec_mat_upper[mask]
    ]))/norm
    return freq, spec

def load_true_Sxx()->np.ndarray:
    """load true Case A, B, C's S_xx

    Returns:
        np.ndarray: shape (n_freq,)
    """
    freq, x_channel_real, y_channel_real, z_channel_real = load_true_psd()
    return freq[:,0],x_channel_real


def plot_Sxx_ratio(ax, case, color):
    """divide each S_xx by the true S_xx -- the ratio of the estimated to the true
    plot the ratio and 90% CI for each case
    """
    f, sxx = load_Sxx(case)
    true_f, true_sxx = load_true_Sxx()
    # interpolate the true S_xx to match f of the estimated S_xx
    true_sxx = interp.interp1d(true_f, true_sxx, kind='cubic', bounds_error=False, fill_value='extrapolate')(f)

    # # compute absolute error at each frequency
    # error  = np.array([
    #     np.abs(sxx[0] - true_sxx),
    #     np.abs(sxx[1] - true_sxx),
    #     np.abs(sxx[2] - true_sxx)
    # ])
    # ax.plot(f, error[1], color=color, label=f"Case {case}")
    # ax.fill_between(f, error[0], error[2], color=color, alpha=0.3, lw=0)
    # ax.set_ylabel('Absolute Error')
    # ax.set_xlabel('Frequency [Hz]')


    # ratio = np.exp(np.log(sxx) - np.log(true_sxx))
    # ax.plot(f, ratio[1], color=color, label=f"Case {case}")
    # ax.fill_between(f, ratio[0], ratio[2], color=color, alpha=0.3, lw=0)
    #
    #

    # TODO: jianan -- i think the scaling is different...
    ax.plot(f,sxx[1], color=color)
    ax.fill_between(f, sxx[0], sxx[2], color=color, alpha=0.3, lw=0)
    ax.plot(f,true_sxx, color='k', label=case)


    ax.set_xlim([5, 128])



def main():
    fig, ax = plt.subplots(1, 1)
    plot_Sxx_ratio(ax, 'A', 'C0')
    plot_Sxx_ratio(ax, 'B', 'C1')
    plot_Sxx_ratio(ax, 'C', 'C2')
    ax.legend(
        labelspacing=0.05, columnspacing=0.25, handlelength=1
    )
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.savefig(f'{paths.figures}/Sxx_ratio.pdf', dpi=300)


if __name__ == "__main__":
    main()