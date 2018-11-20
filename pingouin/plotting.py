"""Plotting functions."""

# Author: Raphael Vallat <raphaelvallat9@gmail.com>
#         Nicolas Legrand <legrand@cyceron.fr>

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_skipped(x, y, n_boot=2000, seed=None):
    """Plot the bootstrapped 95% confidence intervals and distribution
    of a robust Skipped correlation.

    Parameters
    ----------
    x, y : 1D-arrays or list
        Samples
    n_boot : int
        Number of bootstrap iterations for the computation of the
        confidence intervals
    seed : int
        Random seed generator.
    """

    from pingouin.correlation import skipped
    from scipy.stats import spearmanr, pearsonr
    from pingouin.effsize import compute_bootci

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.size == y.size

    # Skipped Spearman / Pearson correlations
    r, p, outliers = skipped(x, y, method='spearman')
    r_pearson, _ = pearsonr(x[~outliers], y[~outliers])

    # Bootstrapped skipped Spearman distribution & CI
    spearman_ci, spearman_dist = compute_bootci(
        x=x[~outliers], y=y[~outliers], func='spearman',
        n_boot=n_boot, return_dist=True, seed=seed)

    # Bootstrapped skipped Pearson distribution & CI
    pearson_ci, pearson_dist = compute_bootci(
        x=x[~outliers], y=y[~outliers], func='pearson',
        n_boot=n_boot, return_dist=True, seed=seed)

    # START THE PLOT
    sns.set(style='ticks', context='notebook', font_scale=1.2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    plt.subplots_adjust(wspace=0.3)
    sns.despine()

    # Scatter plot and regression lines
    sns.regplot(x[~outliers], y[~outliers], ax=ax1, color='darkcyan')
    ax1.scatter(x[outliers],  y[outliers],  color='indianred', label='outliers')
    ax1.scatter(x[~outliers],  y[~outliers],  color='seagreen', label='good')

    # Labels
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Weight')
    ax1.set_title('Outliers (n={})'.format(sum(outliers)), y=1.05)

    # Spearman distribution
    sns.distplot(spearman_dist, kde=True, ax=ax2, color='darkcyan')
    for i in spearman_ci:
        ax2.axvline(x = i, color='coral', lw=2)
    ax2.axvline(x=0, color='k', ls='--', lw=1.5)
    ax2.set_ylabel('Density of bootstrap samples')
    ax2.set_xlabel('Correlation coefficient')
    ax2.set_title(
        'Skipped Spearman r = {}\nCI = [{}, {}]'.format(r.round(2),
                                                        spearman_ci[0],
                                                        spearman_ci[1]),
                                                        y=1.05)

    # Pearson dististribution
    sns.distplot(pearson_dist, kde=True, ax=ax3, color='steelblue')
    for i in pearson_ci:
        ax3.axvline(x=i, color='coral', lw=2)
    ax3.axvline(x=0, color='k', ls='--', lw=1.5)
    ax3.set_xlabel('Correlation coefficient')
    ax3.set_title(
        'Skipped Pearson r = {}\nCI = [{}, {}]'.format(r_pearson.round(2),
                                                        pearson_ci[0],
                                                        pearson_ci[1]),
                                                        y=1.05)
