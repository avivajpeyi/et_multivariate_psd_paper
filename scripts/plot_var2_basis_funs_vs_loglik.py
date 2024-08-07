import pandas as pd
import matplotlib.pyplot as plt
import paths

basis_fun_vs_max_lnl_results = pd.read_csv(f'{paths.data}/basis_fun_vs_max_lnl_results_var2_vma1.csv')
basis_vs_max_lnl = basis_fun_vs_max_lnl_results.values
x = range(1,71)

#number of basis funs vs max log lokelihood for var2 model with length 256, 512 and 1024.
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, sharey=False)
labels = ['n = 256', 'n = 512', 'n = 1024']
for i, ax in enumerate(axes):
    ax.plot(x, basis_vs_max_lnl[:, i])
    ax.text(0.05, 0.95, labels[i], transform=ax.transAxes, verticalalignment='top')

axes[-1].set_xlabel('Number of Basis Functions')
fig.text(0.04, 0.5, 'Maximized Log Likelihood', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0)

plt.savefig(f'{paths.figures}/var2_basis_funs_vs_loglik.pdf', dpi=300)















