import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths

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

    ax.plot(number_basis_sorted, max_lnl_sorted, label=label, **kwgs)
    ax.set_xlabel(r'$M$', fontsize=15)
    ax.set_ylabel(r'Maximized $\log \mathcal{L}(d|\theta)$', fontsize=15)
