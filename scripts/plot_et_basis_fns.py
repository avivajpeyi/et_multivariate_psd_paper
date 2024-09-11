import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths
from paths import LBL_FONTSIZE

def plot_basis_vs_lnl(ax, csv_path, label, kwgs={}):
    # Load data from CSV
    data = pd.read_csv(csv_path)
    data = data.values
    number_basis = data[:,0]
    max_lnl = data[:,1]

    # Sort data by number of basis functions
    sorted_indices = np.argsort(number_basis)
    number_basis_sorted = number_basis[sorted_indices]
    max_lnl_sorted = max_lnl[sorted_indices]

    # Normalize the max_lnl_sorted
    max_lnl_normalized = max_lnl_sorted - max_lnl_sorted.max()
    ax.plot(number_basis_sorted, max_lnl_normalized, label=label, **kwgs)
    ax.set_xlabel(r'$M$', fontsize=LBL_FONTSIZE*0.95)
    ax.set_ylabel(r'Normalised log MLE', fontsize=LBL_FONTSIZE*0.95)
    ax.set_xlim(min(number_basis), max(number_basis))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    # turn off minor
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())


def main():
    csvs = [
        "ET_corr_lr_low_basis_vs_lnl_combined_results.csv",
        "ET_uncorr_lr_low_basis_vs_lnl_combined_results.csv",
    ]
    labels = [
        "Case A",
        "Case B & C",
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
    plt.savefig(f"{paths.figures}/et_basis_fns.pdf", dpi=300, pad_inches=0.025)

if __name__ == "__main__":
    main()
