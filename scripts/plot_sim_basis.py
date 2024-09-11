import pandas as pd
import matplotlib.pyplot as plt
import paths
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from paths import LBL_FONTSIZE

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
    plot_basis_vs_lnl(ax, DATA[:, :3], kwgs=dict(ls='dotted'))

    # custom legend
    lnkwg = dict(lw=2)
    lgd = ax.legend(handles=
    [
        Line2D([0], [0], color='C0', lw=2, label=r'$n=256$'),
        Line2D([0], [0], color='C1', lw=2, label=r'$n=512$'),
        Line2D([0], [0], color='C2', lw=2, label=r'$n=1024$'),
        Line2D([0], [0], color='k', lw=2, label='VAR(2)'),
        Line2D([0], [0], color='k', lw=2, label='VMA(1)', ls='dotted'),
    ], loc='lower right', ncol=2, labelspacing=0.05, columnspacing=0.25, handlelength=1)
    # bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel(r'$M$', fontsize=LBL_FONTSIZE*0.8)
    ax.set_ylabel(r'Normalised log MLE', fontsize=LBL_FONTSIZE*0.95)

    # use scientific notation use x100 instead of 1e2
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    # turn off minor
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.xlim(min(X), max(X))
    fig.savefig(f'{paths.figures}/sim_basis.pdf', bbox_inches='tight', pad_inches=0.025)


if __name__ == "__main__":
    main()