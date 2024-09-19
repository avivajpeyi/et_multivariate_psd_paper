import h5py
import numpy as np
import pandas as pd


data = []
CHANNELS = ["X", "Y", "Z"]

file_path = r"C:\Users\jliu812\OneDrive - The University of Auckland\Desktop\gravitational wave data analysis\HDF5Files_NZ"


for i, channel in enumerate(CHANNELS):
    file_name = f'{channel}_ETnoise_GP.hdf5' 
    full_path = f'{file_path}\\{file_name}'  
    with h5py.File(full_path, "r") as f:
        data.append(f[f"E{i+1}:STRAIN"][:])  


time = np.linspace(0, 2000, len(data[0]))


data = np.array(data)
data = data[:,0:2**20]

with h5py.File("ET_caseA_noise_small.h5", "w") as f:
   
    f.create_dataset("X", data=data[0])
    f.create_dataset("Y", data=data[1])
    f.create_dataset("Z", data=data[2])
  
    f.create_dataset("time", data=time)


file_path_base = r'C:\Users\jliu812\OneDrive - The University of Auckland\Desktop\gravitational wave data analysis\ADD_files'


file_path = f'{file_path_base}\\ET(1).txt'
ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak10Hz_new.txt'
Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak50Hz_new.txt'
Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak90Hz_new.txt'
Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


with h5py.File("ET_caseA_noise_small.h5", "a") as f:
   
    true_psd_group = f.create_group("True PSD")
    
    true_psd_group.create_dataset("ET_1", data=ET_1)
    true_psd_group.create_dataset("Peak10Hz", data=Peak10Hz)
    true_psd_group.create_dataset("Peak50Hz", data=Peak50Hz)
    true_psd_group.create_dataset("Peak90Hz", data=Peak90Hz)




    
    
    
    

#######
#######-------------------------------------------------------------------------------------------------
import h5py
import numpy as np
import pandas as pd


data = []
CHANNELS = ["X", "Y", "Z"]

file_path = r"C:\Users\jliu812\OneDrive - The University of Auckland\Desktop\gravitational wave data analysis\ADD_files"


for i, channel in enumerate(CHANNELS):
    file_name = f'{channel}_ETnoise_GP_uncorr.hdf5' 
    full_path = f'{file_path}\\{file_name}'  
    with h5py.File(full_path, "r") as f:
        data.append(f[f"E{i+1}:STRAIN"][:])  

time = np.linspace(0, 2000, len(data[0]))


data = np.array(data)
data = data[:,0:2**20]

with h5py.File("ET_caseB_noise_small.h5", "w") as f:
    
    f.create_dataset("X", data=data[0])
    f.create_dataset("Y", data=data[1])
    f.create_dataset("Z", data=data[2])
  
    f.create_dataset("time", data=time)


file_path_base = r'C:\Users\jliu812\OneDrive - The University of Auckland\Desktop\gravitational wave data analysis\ADD_files'


file_path = f'{file_path_base}\\ET(1).txt'
ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak10Hz_new.txt'
Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak50Hz_new.txt'
Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


file_path = f'{file_path_base}\\Peak90Hz_new.txt'
Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values


with h5py.File("ET_caseB_noise_small.h5", "a") as f:

    true_psd_group = f.create_group("True PSD")

    true_psd_group.create_dataset("ET_1", data=ET_1)
    true_psd_group.create_dataset("Peak10Hz", data=Peak10Hz)
    true_psd_group.create_dataset("Peak50Hz", data=Peak50Hz)
    true_psd_group.create_dataset("Peak90Hz", data=Peak90Hz)


















