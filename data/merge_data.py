import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from sgvb_psd.postproc.plot_psd import plot_psdq, format_axes

HERE = os.path.dirname(os.path.abspath(__file__))

CASE_A_PSD = 'ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5'
CASE_B_PSD = 'ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
CASE_C_PSD = 'ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'

OLD_NAMES = dict(
    CaseA=CASE_A_PSD,
    CaseB=CASE_B_PSD,
    CaseC=CASE_C_PSD
)

NEW_NAME = "ET-Case-ABC-SGVB-PSD.h5"
GROUPS = ["CaseA", "CaseB", "CaseC"]

keys = ['ETnoise_correlated_GP_spec_mat_lower_XYZ', 'ETnoise_correlated_GP_spec_mat_median_XYZ', 'ETnoise_correlated_GP_spec_mat_upper_XYZ']
new_keys = ['lower_05', 'median', 'upper_95']

freq_data = np.load(os.path.join(HERE, "freq_psd.npz"))

q = 10 ** 22 / 1.0
origFmax = 1024



def load_file(fname):
    print(f"Loading file: {fname}")
    psd_quantiles = {}
    with h5py.File(os.path.join(HERE, fname), 'r') as f:
        for key, new_key in zip(keys, new_keys):
            psd_q = f[key][:]
            re_ = np.real(psd_q) / q / origFmax
            im_ = np.imag(psd_q) / q / origFmax
            psd_quantiles[new_key] = re_ + 1j * im_

    return psd_quantiles

def load_files():
    case_psds = {}
    for group, fname in OLD_NAMES.items():
        case_psds[group] = load_file(fname)
    return case_psds






def save_data(data):
    print(f"Saving data to: {NEW_NAME}")
    with h5py.File(os.path.join(HERE, NEW_NAME), 'w') as f:
        for group in GROUPS:
            for key, new_key in zip(keys, new_keys):
                f.create_dataset(f"{group}/{new_key}", data=data[group][new_key])
        f['freq'] = freq_data['arr_0']




def load_new_data(case):
    """Load the new data

    Return:
        quantiles: [lower_05, median, upper_95]
        freq: frequency data
    """
    with h5py.File(os.path.join(HERE, NEW_NAME), 'r') as f:
        quantiles = [f[f"{case}/{key}"][:] for key in new_keys]
        freq = f['freq'][:]
    return np.array(quantiles), freq


# save_data(data)
# data = load_files()

caseA = load_new_data("CaseA")

axes = plot_psdq(
        caseA[0], caseA[1],
)
format_axes(axes, xlims=[5, 128], sylmog_thresh=1e-49, off_ylims=[-3, 3])
plt.savefig("TEST.png")


