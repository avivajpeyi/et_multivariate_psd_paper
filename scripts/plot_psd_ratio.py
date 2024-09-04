import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import paths
from matplotlib.lines import Line2D
import pandas as pd

# upload the true psd data
file_path =f'{paths.data}/ET(1).txt'
ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

file_path = f'{paths.data}/Peak10Hz_new.txt'
Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

file_path = f'{paths.data}/Peak50Hz_new.txt'
Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

file_path = f'{paths.data}/Peak90Hz_new.txt'
Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

x_channel_real = ((ET_1[:,1]**2 + Peak10Hz[:,1]**2 + Peak50Hz[:,1]**2))/2
y_channel_real = ((ET_1[:,1]**2 + Peak10Hz[:,1]**2 + Peak90Hz[:,1]**2))/2
z_channel_real = ((ET_1[:,1]**2 + Peak50Hz[:,1]**2 + Peak90Hz[:,1]**2))/2

# upload the estiamted psd for case A
caseA=f'{paths.data}/ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5'
with h5py.File(caseA, 'r') as f:
    spec_mat_median = f['ETnoise_correlated_GP_spec_mat_median_XYZ'][:]
    spec_mat_lower = f['ETnoise_correlated_GP_spec_mat_lower_XYZ'][:]
    spec_mat_upper = f['ETnoise_correlated_GP_spec_mat_upper_XYZ'][:]













