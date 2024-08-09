import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths
from utils import plot_basis_vs_lnl


def main():
    csvs = [
        "ET_corr_lr_low_basis_vs_lnl_combined_results.csv",
        "ET_uncorr_lr_low_basis_vs_lnl_combined_results.csv",
    ]
    labels = [
        "Case 1",
        "Case 2 & 3",
    ]
    kwgs = [
    dict(color="C0", ls="-", lw=2),
    dict(color="C1", ls="-", lw=2),
    ]

    fig, ax = plt.subplots(1, 1)
    for i in range(2):
        plot_basis_vs_lnl(ax, f"{paths.data}/{csvs[i]}", kwgs=kwgs[i], label=labels[i])

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{paths.figures}/et_basis_fns.pdf", dpi=300)

if __name__ == "__main__":
    main()
