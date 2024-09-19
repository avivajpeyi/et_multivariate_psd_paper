import h5py
import numpy as np
import paths
from scipy import interpolate as interp

with h5py.File(f'{paths.data}/ET_caseA_noise.h5', 'r') as f:
    
    true_psd_group = f['True PSD']
    
    ET_1 = true_psd_group['ET_1'][:]
    Peak10Hz = true_psd_group['Peak10Hz'][:]
    Peak50Hz = true_psd_group['Peak50Hz'][:]
    Peak90Hz = true_psd_group['Peak90Hz'][:]

x_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak50Hz[:, 1] ** 2)) / 2
y_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2
z_channel_real = ((ET_1[:, 1] ** 2 + Peak50Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2

q = 10 ** 22 / 1.0
time_interval = 2000
nchunks = 125
required_part = 128
total_len = 4096000

Ts = 1 / (total_len / time_interval)
freq_original = np.fft.fftfreq(int(total_len / nchunks), Ts)

n = int(total_len / nchunks)
if np.mod(n, 2) == 0:
    # n is even
    freq_original = freq_original[0:int((n / 2))]
else:
    # n is odd
    freq_original = freq_original[0:int((n - 1) / 2)]

freq_range = total_len / time_interval / 2
freq = freq_original[0:int(required_part / freq_range * freq_original.shape[0])]
freq_true = ET_1[:, 0]

true_s = [
        interp.interp1d(freq_true, x_channel_real, kind='cubic', bounds_error=False, fill_value='extrapolate')(freq),
        interp.interp1d(freq_true, y_channel_real, kind='cubic', bounds_error=False, fill_value='extrapolate')(freq),
        interp.interp1d(freq_true, z_channel_real, kind='cubic', bounds_error=False, fill_value='extrapolate')(freq)
    ]

#frequency band start at 5Hz
freq_start = int(5/required_part * freq.shape[0])

def calculate_rmsd(spec_file_path, q, freq_original, true_s):
    
    with h5py.File(f'{paths.data}/{spec_file_path}', 'r') as f:
        spec_mat_median = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]

    spec_mat_median = spec_mat_median / q**2 / (freq_original[-1] / 0.5)
    p = spec_mat_median.shape[-1]

    rmsd_channels = []
    for i in range(p):   
        rmsd_channel = np.sqrt(np.mean((np.real(spec_mat_median[..., i, i][freq_start:]) - true_s[i][freq_start:]) ** 2))
        rmsd_channels.append(rmsd_channel)

    return np.mean(rmsd_channels)

RMSD_case_B = calculate_rmsd('ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5', q, freq_original, true_s)
RMSD_case_C = calculate_rmsd('ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5', q, freq_original, true_s)

print(f'RMSD Case B: {RMSD_case_B}')
print(f'RMSD Case C: {RMSD_case_C}')



















