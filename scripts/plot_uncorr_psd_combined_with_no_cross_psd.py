import h5py
import numpy as np   
from scipy import signal 
import matplotlib.pyplot as plt
import paths

file_path_x_uncorr = f'{paths.data}/X_ETnoise_GP_uncorr.hdf5'
with h5py.File(file_path_x_uncorr, 'r') as f:
    X_ETnoise_GP_uncorr = f['E1:STRAIN']

file_path_y_uncorr = f'{paths.data}/Y_ETnoise_GP_uncorr.hdf5'
with h5py.File(file_path_y_uncorr, 'r') as f:
    Y_ETnoise_GP_uncorr = f['E2:STRAIN']

file_path_z_uncorr = f'{paths.data}/Z_ETnoise_GP_uncorr.hdf5'
with h5py.File(file_path_z_uncorr, 'r') as f:
    Z_ETnoise_GP_uncorr = f['E3:STRAIN']

#----------------------------------------------------------------------------------------------

channels = np.column_stack((X_ETnoise_GP_uncorr, Y_ETnoise_GP_uncorr, Z_ETnoise_GP_uncorr))

#---------------------------------------------------------------------------------------------------------
#estimated psd (median, lower, upper)
file_path = f'{paths.data}/ETnoise_no_cross_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
with h5py.File(file_path, 'r') as f:
    spec_mat_median_no_cross = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
    spec_mat_lower_no_cross = f['ETnoise_correlated_GP_spec_mat_lower_XYZ'][:]
    spec_mat_upper_no_cross = f['ETnoise_correlated_GP_spec_mat_upper_XYZ'][:]

file_path = f'{paths.data}/ETnoise_uncorrelated_GP_uniform_spec_matrices_XYZ.hdf5'
with h5py.File(file_path, 'r') as f:
    spec_mat_median = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
    spec_mat_lower = f['ETnoise_correlated_GP_spec_mat_lower_XYZ'][:]
    spec_mat_upper = f['ETnoise_correlated_GP_spec_mat_upper_XYZ'][:]


q = 10**22/1.0
time_interval = 2000
nchunks = 125
required_part = 128

Ts = 1/ (channels.shape[0]/time_interval)
freq_original = np.fft.fftfreq(int(np.size(channels,0)/nchunks),Ts)

n = int(np.size(channels,0)/nchunks)
if np.mod(n, 2) == 0:
    # n is even
    freq_original = freq_original[0:int((n/2))]
else:
    # n is odd
    freq_original = freq_original[0:int((n-1)/2)]
       
total_len = channels.shape[0]
freq_range = total_len/time_interval/2
freq = freq_original[0:int(required_part/freq_range * freq_original.shape[0])]

    
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
from matplotlib.lines import Line2D

for i in range(3):
    for j in range(3):
        if i == j:
            f, Pxx_den0 = signal.periodogram(channels[:,i], fs=channels.shape[0]/time_interval)
            f = f[1:]
            Pxx_den0 = Pxx_den0[1:] / 2
            axes[i, j].plot(f, Pxx_den0, marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].plot(freq, spec_mat_median[..., i, i]/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].plot(freq, spec_mat_median_no_cross[..., i, i]/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='blue', linestyle="-")
            
            axes[i, j].fill_between(freq, spec_mat_lower[..., i, i]/(q)**2/(freq_original[-1]/0.5),
                            spec_mat_upper[..., i, i]/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            axes[i, j].fill_between(freq, spec_mat_lower_no_cross[..., i, i]/(q)**2/(freq_original[-1]/0.5),
                            spec_mat_upper_no_cross[..., i, i]/(q)**2/(freq_original[-1]/0.5), color='lightblue', alpha=1)
            
            axes[i, j].text(0.95, 0.95, r'$f_{{{}, {}}}$'.format(i+1, i+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
            axes[i, j].set_ylim([10**(-52), 10**(-46)])
            axes[i, j].set_yscale('log')
            
            
        elif i < j:  
            
            axes[i, j].plot(freq, np.real(spec_mat_median[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].plot(freq, np.real(spec_mat_median_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='blue', linestyle="-")
            
            axes[i, j].fill_between(freq, np.real(spec_mat_lower[..., i, j])/(q)**2/(freq_original[-1]/0.5), 
                                    np.real(spec_mat_upper[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            axes[i, j].fill_between(freq, np.real(spec_mat_lower_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), 
                                    np.real(spec_mat_upper_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightblue', alpha=1)
            
            y = np.apply_along_axis(np.fft.fft, 0, channels)
            n = channels.shape[0]
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.real(cross_spectrum_fij)/(freq_original[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
        
            axes[i, j].set_yscale('symlog', linthresh=1e-49)
        else:
            
            axes[i, j].plot(freq, np.imag(spec_mat_median[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].plot(freq, np.real(spec_mat_median_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='blue', linestyle="-")
            
            axes[i, j].fill_between(freq, (spec_mat_lower[..., i, j])/(q)**2/(freq_original[-1]/0.5),
                                    (spec_mat_upper[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            axes[i, j].fill_between(freq, np.real(spec_mat_lower_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), 
                                    np.real(spec_mat_upper_no_cross[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightblue', alpha=1)
            
            y = np.apply_along_axis(np.fft.fft, 0, channels)
            n = channels.shape[0]
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.imag(cross_spectrum_fij)/(freq_original[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
            
            axes[i, j].set_yscale('symlog', linthresh=1e-49)
fig.text(0.5, 0.1, 'Frequency [Hz]', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, 'Strain PSD [1/Hz]', ha='center', va='center', rotation='vertical', fontsize=20)

fig.legend(handles=[Line2D([], [], color='lightgray', label='Periodogram'),
                Line2D([], [], color='green', label='uncorrelated noise estimated PSD'),
                Line2D([], [], color='lightgreen', label='90% CI uncorr'),
                Line2D([], [], color='blue', label='uncorrelated noise estimated PSD no cross'),
                Line2D([], [], color='lightblue', label='90% CI uncorr no cross')],
             loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=14)

plt.savefig(f'{paths.figures}/uncorr_psd_combined_with_no_cross_psd.pdf', dpi=300)
'''
#---------------------------------------------------------------------------------------------------------
#estimated squared coherence (median, lower, upper)
file_path = 'D:/optimal lrs for ET data/results/ETnoise_no_cross_uncorrelated_GP_uniform_squared_coh_XYZ.hdf5'
with h5py.File(file_path, 'r') as f:
    coh_med = f['ETnoise_correlated_GP_coh_median_XYZ'][:]
    coh_lower = f['ETnoise_correlated_GP_coh_lower_XYZ'][:]
    coh_upper = f['ETnoise_correlated_GP_coh_upper_XYZ'][:]

fig, ax = plt.subplots(1,1, figsize = (20, 6))
plt.xlim([5, 128])
plt.plot(freq, np.squeeze(coh_med[:,0]), color = 'green', linestyle="-", label = 'coherence for X Y')
plt.fill_between(freq, np.squeeze(coh_lower[:,0]), np.squeeze(coh_upper[:,0]),
                    color = ['lightgreen'], alpha = 1, label = '90% CI for X Y')


plt.plot(freq, np.squeeze(coh_med[:,1]), color = 'blue', linestyle="-", label = 'coherence for X Z')
plt.fill_between(freq, np.squeeze(coh_lower[:,1]), np.squeeze(coh_upper[:,1]),
                    color = ['lightblue'], alpha = 1, label = '90% CI for X Z')


plt.plot(freq, np.squeeze(coh_med[:,2]), color = 'red', linestyle="-", label = 'coherence for Y Z')
plt.fill_between(freq, np.squeeze(coh_lower[:,2]), np.squeeze(coh_upper[:,2]),
                    color = ['lightcoral'], alpha = 1, label = '90% CI for Y Z')

plt.xlabel('Frequency [Hz]', fontsize=20, labelpad=10)   
plt.ylabel('Squared Coherency', fontsize=20, labelpad=10)   
#plt.title('Squared coherence for ET noise with correlated GP', pad=20, fontsize = 20)

plt.legend(loc='upper left', fontsize='medium')
plt.ylim([0, 0.7])

plt.grid(True)

'''
























