import h5py
import numpy as np
import paths

q = 10 ** 22 / 1.0
freq_original = 1024
def compute_average_RMS(file_path, dataset_prefix):
    
    with h5py.File(file_path, 'r') as f:
        spec_mat_median = f[f'{dataset_prefix}_spec_mat_median_XYZ'][:]
        spec_mat_lower = f[f'{dataset_prefix}_spec_mat_lower_XYZ'][:]
        spec_mat_upper = f[f'{dataset_prefix}_spec_mat_upper_XYZ'][:]

    spec_mat_median = np.real(spec_mat_median)
    spec_mat_median = spec_mat_median/q**2/(freq_original/0.5)
    p = spec_mat_median.shape[-1]

    RMS = []
    for i in range(p):
        for j in range(p):
            if i == j:
                
                RMS_ij = np.sqrt(np.mean(np.real(spec_mat_median[..., i, j])**2))
            elif i < j:
                
                RMS_ij = np.sqrt(np.mean(np.real(spec_mat_median[..., i, j])**2))
            else:
                
                RMS_ij = np.sqrt(np.mean(np.imag(spec_mat_median[..., i, j])**2))

            RMS.append(RMS_ij)

    average_RMS = np.mean(RMS)
    return average_RMS

# Case B
file_path_B = f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
average_RMS_B = compute_average_RMS(file_path_B, 'ETnoise_correlated_GP')
print(f'Average RMS for case B: {average_RMS_B}')

# Case C
file_path_C = f'{paths.data}/ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
average_RMS_C = compute_average_RMS(file_path_C, 'ETnoise_correlated_GP')
print(f'Average RMS for case C: {average_RMS_C}')







