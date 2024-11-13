import h5py
import numpy as np
import matplotlib.pyplot as plt
import paths
from pingouin import multivariate_normality
import pandas as pd

'''
Load PSD data from HDF5 file and apply scaling for all cases.
'''
def load_and_scale_psd(file_path, scaling_factor, true_fmax, original_fmax):
    with h5py.File(file_path, 'r') as f:
        psd_data = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
    
    return psd_data / (scaling_factor**2) / (true_fmax/original_fmax)   

scaling_factor = 10.0**22
original_fmax = 0.5
fs = 4096000/2000
true_fmax = fs/2

cases_psd = {
    'A': f'{paths.data}/ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5',
    'B': f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5',
    'C': f'{paths.data}/ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
}

psd_medians = {
    f'case{case}_psd_median': load_and_scale_psd(
        file_path,
        scaling_factor,
        true_fmax,
        original_fmax
    )
    for case, file_path in cases_psd.items()
}

caseA_psd_median = psd_medians['caseA_psd_median']
caseB_psd_median = psd_medians['caseB_psd_median']
caseC_psd_median = psd_medians['caseC_psd_median']


'''
Load ET noise data for all cases.
'''
caseA_data_pth = str(paths.data) + "/{}_ETnoise_GP.hdf5"
caseB_data_pth = str(paths.data) + "/{}_ETnoise_GP_uncorr.hdf5"

def load_raw_data(channel_pth):
    channels = []
    for i, c in enumerate('XYZ'):
        with h5py.File(channel_pth.format(c), 'r') as f:
            channels.append(f[f'E{i + 1}:STRAIN'][:])
    channels = np.column_stack(channels)
    return channels

caseA_data = load_raw_data(caseA_data_pth)
caseB_data = load_raw_data(caseB_data_pth)


'''
Chunk the whole dataset into 125 blocks
Fourier transform each block of data
then select the required part for all cases
'''
num_segments=125
fmax_for_analysis=128

def process_fft_data(data, num_segments, fmax_for_analysis, true_fmax):
    len_chunk = data.shape[0] // num_segments
    segmented_data = np.array(np.split(data[0:len_chunk*num_segments,:], num_segments))
    
    y = np.array([
        np.apply_along_axis(np.fft.fft, 0, segment)
        for segment in segmented_data
    ])
    
    n = segmented_data.shape[1]
    y = y / np.sqrt(n)

    if np.mod(n, 2) == 0:
        y = y[:, 0:int(n/2), :]
    else:
        y = y[:, 0:int((n-1)/2), :]

    y = y[:, 0:int(fmax_for_analysis/true_fmax * y.shape[1]), :]
    return y

cases_data = {
    'A': caseA_data,
    'B': caseB_data
}

y_cases = {
    f'y_{case}': process_fft_data(
        data, 
        num_segments, 
        fmax_for_analysis, 
        true_fmax
    )
    for case, data in cases_data.items()
}

y_caseA = y_cases['y_A']
y_caseB = y_cases['y_B']


'''
Find the inverse squared root for all median psd matrices
'''
def calculate_psd_inverse_sqrt(psd_matrix):
    psd_inv_sqrt = np.zeros_like(psd_matrix, dtype=complex)
    
    for i in range(psd_matrix.shape[0]):
        eigenvalues, eigenvectors = np.linalg.eigh(psd_matrix[i])
        inv_sqrt_eigenvalues = np.diag(1 / np.sqrt(eigenvalues))
        psd_inv_sqrt[i] = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.conj().T
        
    return psd_inv_sqrt

cases_psd = {
    'A': caseA_psd_median,
    'B': caseB_psd_median,
    'C': caseC_psd_median
}

psd_inv_sqrt_results = {
    f'case{case}_psd_inv_sqrt': calculate_psd_inverse_sqrt(psd_matrix)
    for case, psd_matrix in cases_psd.items()
}

caseA_psd_inv_sqrt = psd_inv_sqrt_results['caseA_psd_inv_sqrt']
caseB_psd_inv_sqrt = psd_inv_sqrt_results['caseB_psd_inv_sqrt']
caseC_psd_inv_sqrt = psd_inv_sqrt_results['caseC_psd_inv_sqrt']


'''
Whiten the Fourier transformed dataset and test the normality
'''
def whiten_process_test(psd_inv_sqrt, y_data, case_name):
    
    n_segments = y_data.shape[0]
    n_frequencies = y_data.shape[1]
    results = []
    
    for segment in range(n_segments):
        whitened_data = np.zeros((n_frequencies, y_data.shape[-1]), dtype=complex)
        
        for freq in range(n_frequencies):
            whitened_data[freq] = psd_inv_sqrt[freq] @ y_data[segment, freq]

        real_part = whitened_data.real
        imag_part = whitened_data.imag

        real_normality_test = multivariate_normality(real_part)
        imag_normality_test = multivariate_normality(imag_part)

        real_p_value = real_normality_test[1]
        real_is_normal = real_normality_test[2]
        imag_p_value = imag_normality_test[1]
        imag_is_normal = imag_normality_test[2]

        results.append({
            'Segment': segment,
            'Real p-value': real_p_value,
            'Real is normal': real_is_normal,
            'Imaginary p-value': imag_p_value,
            'Imaginary is normal': imag_is_normal
        })

    df = pd.DataFrame(results)
    file_name = f'{case_name}_normality_test_results.csv'
    df.to_csv(file_name, index=False)

whiten_process_test(caseA_psd_inv_sqrt, y_caseA, 'caseA')
whiten_process_test(caseB_psd_inv_sqrt, y_caseB, 'caseB')
whiten_process_test(caseC_psd_inv_sqrt, y_caseB, 'caseC')

#---------------------------------------------------------------------------------------------------
'''
Plot the whitened Fourier transformed dataset for all cases
'''
def plot_whitened_data(psd_inv_sqrt, y_data, select_chunk, case_name="", save_path=None):
    """
    Create enhanced 3D scatter visualization of whitened data
    """
    n_frequencies = y_data.shape[1]
    whitened_data = np.zeros((n_frequencies, y_data.shape[-1]), dtype=complex)
    
    for freq in range(n_frequencies):
        whitened_data[freq] = psd_inv_sqrt[freq] @ y_data[select_chunk, freq]
    
    real_part = whitened_data.real
    imag_part = whitened_data.imag
    
    fig = plt.figure(figsize=(16, 7))
    
    # Real part plot
    ax_real = fig.add_subplot(121, projection='3d')
    scatter_real = ax_real.scatter(real_part[:, 0], real_part[:, 1], real_part[:, 2],
                                 c=real_part[:, 2],  
                                 cmap='viridis',
                                 alpha=0.7,
                                 s=15)
    
    ax_real.set_title(f'{case_name} Real Part of Whitened Data', pad=15, fontsize=13)
    ax_real.set_xlabel('Dimension 1 (Real)', labelpad=10)
    ax_real.set_ylabel('Dimension 2 (Real)', labelpad=10)
    ax_real.set_zlabel('Dimension 3 (Real)', labelpad=10)
    
    ax_real.view_init(elev=25, azim=45)
    
    plt.colorbar(scatter_real, ax=ax_real, label='Dimension 3 value')
    
    # Imaginary part plot
    ax_imag = fig.add_subplot(122, projection='3d')
    scatter_imag = ax_imag.scatter(imag_part[:, 0], imag_part[:, 1], imag_part[:, 2],
                                 c=imag_part[:, 2],
                                 cmap='viridis',     
                                 alpha=0.7,          
                                 s=15)               
    
    ax_imag.set_title(f'{case_name} Imaginary Part of Whitened Data', pad=15, fontsize=13)
    ax_imag.set_xlabel('Dimension 1 (Imag)', labelpad=10)
    ax_imag.set_ylabel('Dimension 2 (Imag)', labelpad=10)
    ax_imag.set_zlabel('Dimension 3 (Imag)', labelpad=10)
    
    ax_imag.view_init(elev=25, azim=45)
    
    plt.colorbar(scatter_imag, ax=ax_imag, label='Dimension 3 value')
    
    for ax in [ax_real, ax_imag]:

        ax.grid(True, alpha=0.2)
        
        ax.xaxis._axinfo["grid"].update({"alpha": 0.2})
        ax.yaxis._axinfo["grid"].update({"alpha": 0.2})
        ax.zaxis._axinfo["grid"].update({"alpha": 0.2})
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        ax.set_box_aspect([1,1,1])
        
        ax.tick_params(axis='x', rotation=20)
        ax.tick_params(axis='y', rotation=20)
    
    plt.tight_layout(pad=3.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

plot_whitened_data(caseA_psd_inv_sqrt, y_caseA, 10, "Case A", 
                  save_path=f'{paths.figures}/caseA_normality_test.pdf')
plot_whitened_data(caseB_psd_inv_sqrt, y_caseB, 10, "Case B", 
                  save_path=f'{paths.figures}/caseB_normality_test.pdf')
plot_whitened_data(caseC_psd_inv_sqrt, y_caseB, 10, "Case C", 
                  save_path=f'{paths.figures}/caseC_normality_test.pdf')























