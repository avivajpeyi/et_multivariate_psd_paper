import pandas as pd
import matplotlib.pyplot as plt
import paths
from matplotlib.lines import Line2D

DATA = pd.read_csv(f'{paths.data}/basis_fun_vs_max_lnl_results_var2_vma1.csv').values
X = range(1, 71)


def plot_basis_vs_lnl(ax, data, kwgs={}):
    for i in range(3):
        y = data[:, i] - data[:, i].max()
        ax.plot(X, y, color=f"C{i}", lw=2, **kwgs)


def main():

    # number of basis funs vs max log lokelihood for var2 model with length 256, 512 and 1024.
    fig = plt.figure()
    ax = fig.gca()




    plot_basis_vs_lnl(ax, DATA[:, 3:])
    plot_basis_vs_lnl(ax, DATA[:, :3], kwgs=dict(ls='--'))

    # custom legend
    lnkwg = dict(lw=2)
    lgd = ax.legend(handles=
    [
        Line2D([0], [0], color='k', lw=2, label='VAR(2)'),
        Line2D([0], [0], color='k', lw=2, label='VMA(1)', ls='--'),
        Line2D([0], [0], color='C0', lw=2, label='n=256'),
        Line2D([0], [0], color='C1', lw=2, label='n=512'),
        Line2D([0], [0], color='C2', lw=2, label='n=1024'),
    ], loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.xlabel(r'$M$')
    plt.ylabel(r'Normalised $\log \mathcal{L}(d|\theta)$')
    plt.xlim(min(X), max(X))
    fig.savefig(f'{paths.figures}/sim_basis.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()