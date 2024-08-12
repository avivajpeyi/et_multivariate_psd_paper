#--------------------------------------------------------------------------------------------------------
#plot the average of 50 realizations for number of chunks vs values of log likelihood

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths


def main():
    lnl_no_chunks = pd.read_csv(f'{paths.data}/likelihood_without_chunk.csv').values
    lnl_chunks = pd.read_csv(f'{paths.data}/likelihood_with_chunks.csv').values
    rel_lnl = lnl_chunks - lnl_no_chunks
    lnl_quantiles = np.quantile(rel_lnl, [0.05, 0.5, 0.95], axis=0)

    total_length = 2**17
    x = range(8)
    chunk_lengths = [total_length // (2**i) for i in x]

    fig, ax = plt.subplots(1, 1)
    # ax.plot(x, [average_loglik_values_no_chunks[1]] * len(x), label='Without Chunks', color=f"C0")
    # ax.fill_between(x, [average_loglik_values_no_chunks[0]] * len(x), [average_loglik_values_no_chunks[2]] * len(x), label='Without Chunks', color=f"C0")


    ax.plot(x, lnl_quantiles[1], label='With Chunks', color=f"C1")
    ax.fill_between(x, lnl_quantiles[0], lnl_quantiles[2], color=f"C1", alpha=0.3, lw=0)

    power_labels = [f'$2^{i}$' for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(power_labels)
    ax.set_ylim(bottom=-300)
    ax.set_xlim(x[0],x[-1])

    plt.xlabel(r'$N_b$')
    plt.ylabel(r'$\log \mathcal{L}(\bf{d}) - \log \mathcal{L}_b(\bf{d})$')
    secax = ax.secondary_xaxis('top')
    ax.xaxis.set_tick_params(which='minor',top=False, bottom=False)
    secax.xaxis.set_tick_params(which='minor',top=False, bottom=False)
    ax.yaxis.set_tick_params(which='minor',left=False, right=False)
    secax.set_xticks(x)
    chunk_length_labels = [f'$2^{{{int(np.log2(length))}}}$' for length in chunk_lengths]
    secax.set_xticklabels(chunk_length_labels)
    secax.set_xlabel('Chunk Length', fontsize=15)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{paths.figures}/lnl_vs_nchunks.pdf', dpi=300)


main()
