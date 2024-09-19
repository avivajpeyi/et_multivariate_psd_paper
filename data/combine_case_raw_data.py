import os

import h5py
import numpy as np
import pandas as pd
from sgvb_psd.utils.periodogram import get_periodogram

CHANNELS = ["X", "Y", "Z"]

HERE = os.path.dirname(os.path.abspath(__file__))

CASE_A = "{}_ETnoise_GP.hdf5"
CASE_B = "{}_ETnoise_GP_uncorr.hdf5"

RAW_FPATHS = dict(caseA=CASE_A, caseB=CASE_B)


def load_et_true():
    file_path = os.path.join(HERE, "ET(1).txt")
    ET_1 = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = os.path.join(HERE, "Peak10Hz_new.txt")
    Peak10Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = os.path.join(HERE, "Peak50Hz_new.txt")
    Peak50Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    file_path = os.path.join(HERE, "Peak90Hz_new.txt")
    Peak90Hz = pd.read_csv(file_path, delim_whitespace=True, header=None).values

    with h5py.File("ET_caseA_noise_small.h5", "a") as f:
        true_psd_group = f.create_group("True PSD")

        true_psd_group.create_dataset("ET_1", data=ET_1)
        true_psd_group.create_dataset("Peak10Hz", data=Peak10Hz)
        true_psd_group.create_dataset("Peak50Hz", data=Peak50Hz)
        true_psd_group.create_dataset("Peak90Hz", data=Peak90Hz)

    x_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak50Hz[:, 1] ** 2)) / 2
    y_channel_real = ((ET_1[:, 1] ** 2 + Peak10Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2
    z_channel_real = ((ET_1[:, 1] ** 2 + Peak50Hz[:, 1] ** 2 + Peak90Hz[:, 1] ** 2)) / 2
    true_frq = ET_1[:, 0]
    return np.array([x_channel_real, y_channel_real, z_channel_real]), true_frq


ET_TRUE = load_et_true()


def load_data(case, large=True):
    data = []

    print("RUNNING ", case)

    for i, channel in enumerate(CHANNELS):
        file_name = RAW_FPATHS[case].format(channel)
        full_path = os.path.join(HERE, file_name)
        print(f"Loading file: {full_path}")
        with h5py.File(full_path, "r") as f:
            data.append(f[f"E{i + 1}:STRAIN"][:])
    time = np.linspace(0, 2000, len(data[0]))

    data = np.array(data)
    print("Data shape: ", data.shape)

    fname = f"ET_{case}_noise.h5"
    if not large:
        fname = f"ET_{case}_noise_small.h5"

    print(f"Creating file: {fname}")
    with h5py.File(fname, "w") as f:
        if large:
            f.create_dataset("raw_XYZ", data=data)
            f.create_dataset("time", data=time)
        pdgrm, frq = get_periodogram(data.T, fs=2048)

        # create a group for the periodogram
        pdgrm_group = f.create_group("periodogram")
        pdgrm_group.create_dataset("pdgrm", data=pdgrm)
        pdgrm_group.create_dataset("freq", data=frq)

        # create a group for the true PSD
        true_psd_group = f.create_group("true_psd")
        true_psd, true_frq = ET_TRUE
        true_psd_group.create_dataset("psd", data=true_psd)
        true_psd_group.create_dataset("freq", data=true_frq)

        # add samling frequency
        f.attrs["fs"] = 2048

    print(f"Data saved to: {fname}")




if __name__ == "__main__":
    for case in RAW_FPATHS:
        load_data(case)
        load_data(case, large=False)



# AFTERWARDS


