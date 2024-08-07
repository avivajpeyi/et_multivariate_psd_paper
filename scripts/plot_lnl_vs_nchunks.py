#--------------------------------------------------------------------------------------------------------
#plot the average of 50 realizations for number of chunks vs values of log likelihood

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths


loglik_values_no_chunks = pd.read_csv(f'{paths.data}/likelihood_without_chunk.csv').values
average_loglik_values_no_chunks = np.mean(loglik_values_no_chunks)

loglik_values_chunks = pd.read_csv(f'{paths.data}/likelihood_with_chunks.csv').values
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
plt.savefig(f'{paths.figures}/lnl_vs_nchunks.pdf', dpi=300)
