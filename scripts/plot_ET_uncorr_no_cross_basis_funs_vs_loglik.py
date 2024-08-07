import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths

#-------------------------------------------------------------------------------------------------------
#number of basis functions vs maximised log likelihood for ET data with correlated noise given lr = 0.002
data = pd.read_csv(f'{paths.data}/no_cross_basis_vs_lnl_ET_uncorr_lr_low.csv')
data = data.values
number_basis = data[:,0]
max_lnl = data[:,1]
sorted_indices = np.argsort(number_basis)
number_basis_sorted = number_basis[sorted_indices]
max_lnl_sorted = max_lnl[sorted_indices]

fig, ax = plt.subplots(1,1, figsize = (10, 6))
plt.plot(number_basis_sorted, max_lnl_sorted)
plt.xlabel('Number of Basis Functions', fontsize=15)
plt.ylabel('Maximized Log Likelihood', fontsize=15)
#plt.title('number of basis function vs maximized log likelihood', fontsize=20)
plt.savefig(f'{paths.figures}/ET_uncorr_no_cross_basis_funs_vs_loglik.pdf', dpi=300)