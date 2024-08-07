import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy import signal
#import true_var
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
'''
num_rep = 50
# load data
n = 2**17
sigma = np.array([[1., 0.9], [0.9, 1.]])
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1., 0.], [0., 1.]]])

all_data = []

for _ in range(num_rep):
    Simulation = true_var.VarmaSim(n=n)
    x = Simulation.simData(varCoef, vmaCoef, sigma=sigma)
    all_data.append(x)

data = np.vstack(all_data)


import modified_spec_vi_nochunks_loglike_only
original_loglik = []
for i in range(num_rep):
    x = data[(i*n):((i+1)*n),:]
    Spec = modified_spec_vi_nochunks_loglike_only.SpecVI(x)  
    log_lik = Spec.runModel(N_delta=30, N_theta=30, sparse_op=False)
    original_loglik.append(log_lik.numpy())
    
result_df = pd.DataFrame(original_loglik)
result_df.to_csv('likelihood without chunk.csv', index=False)

#----------------------------------------------------------------------------------------------------------
#50 realizations for number of chunks vs value of log likelihood
import modified_spec_vi_chunks_loglike_only

log_lik_matrix = []
for i in range(num_rep):
    x = data[(i*n):((i+1)*n),:]
    
    log_lik_row = []
    
    for j in range(8):
        Spec = modified_spec_vi_chunks_loglike_only.SpecVI(x)  
        log_lik = Spec.runModel(N_delta=30, N_theta=30, sparse_op=False, nchunks=2**j)
        log_lik_row.append(log_lik.numpy())
    
    log_lik_matrix.append(log_lik_row)

columns = [f"log_lik_chunk_{2**j}" for j in range(8)]
result_df = pd.DataFrame(log_lik_matrix, columns=columns)

result_df.to_csv('likelihood_with_chunks.csv', index=False)

#----------------------------------------------------------------------------------------------------------
#plots of 50 realizations for number of chunks vs values of log likelihood

loglik_values_no_chunks = pd.read_csv('D:/log like vs num chunks/likelihood without chunk.csv')
loglik_values_no_chunks = loglik_values_no_chunks.values

loglik_values_chunks = pd.read_csv('D:/log like vs num chunks/likelihood_with_chunks.csv')
loglik_values_chunks = loglik_values_chunks.values

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = range(8)

for i in range(len(loglik_values_no_chunks)):
    ax.plot(x, [loglik_values_no_chunks[i]] * len(x))

for i in range(loglik_values_chunks.shape[0]):
    ax.plot(x, loglik_values_chunks[i, :])

power_labels = [f'$2^{i}$' for i in range(8)]
ax.set_xticks(x)
ax.set_xticklabels(power_labels)

plt.xlabel('Number of Chunks',fontsize=15)
plt.ylabel('Log Likelihood',fontsize=15)

#-------------------------------------------------------------------------------------------------------
#plot of 1 realization for number of chunks vs values of log likelihood
loglik_values_no_chunks = pd.read_csv('D:/log like vs num chunks/likelihood without chunk.csv')
loglik_values_no_chunks = loglik_values_no_chunks.values

loglik_values_chunks = pd.read_csv('D:/log like vs num chunks/likelihood_with_chunks.csv')
loglik_values_chunks = loglik_values_chunks.values

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = range(8)
ax.plot(x, [loglik_values_no_chunks[5]] * len(x))
ax.plot(x, loglik_values_chunks[5, :])

power_labels = [f'$2^{i}$' for i in range(8)]
ax.set_xticks(x)
ax.set_xticklabels(power_labels)

plt.xlabel('Number of Chunks',fontsize=15)
plt.ylabel('Log Likelihood',fontsize=15)
'''
#--------------------------------------------------------------------------------------------------------
#plot the average of 50 realizations for number of chunks vs values of log likelihood
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loglik_values_no_chunks = pd.read_csv('D:/log like vs num chunks/likelihood without chunk.csv').values
average_loglik_values_no_chunks = np.mean(loglik_values_no_chunks)

loglik_values_chunks = pd.read_csv('D:/log like vs num chunks/likelihood_with_chunks.csv').values
average_loglik_values_chunks = np.mean(loglik_values_chunks, axis=0)

total_length = 2**17
x = range(8)
chunk_lengths = [total_length // (2**i) for i in x]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(x, [average_loglik_values_no_chunks] * len(x), label='Without Chunks')
ax.plot(x, average_loglik_values_chunks, label='With Chunks')

power_labels = [f'$2^{i}$' for i in x]
ax.set_xticks(x)
ax.set_xticklabels(power_labels)

plt.xlabel('Number of Chunks', fontsize=15)
plt.ylabel('Log Likelihood', fontsize=15)

secax = ax.secondary_xaxis('top')
secax.set_xticks(x)
chunk_length_labels = [f'$2^{{{int(np.log2(length))}}}$' for length in chunk_lengths]
secax.set_xticklabels(chunk_length_labels)
secax.set_xlabel('Chunk Length', fontsize=15)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()









    