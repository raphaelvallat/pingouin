"""Plotting functions.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <legrand@cyceron.fr>
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set default Seaborn preferences
sns.set(style='ticks', context='notebook')

__all__ = ["plot_skipped_corr"]


def plot_skipped_corr(x, y, xlabel=None, ylabel=None, n_boot=2000, seed=None):

    """Plot the bootstrapped 95% confidence intervals and distribution
    of a robust Skipped correlation.

    Parameters
    ----------
    x, y : 1D-arrays or list
        Samples
    xlabel, ylabel : str
        Axes labels
    n_boot : int
        Number of bootstrap iterations for the computation of the
        confidence intervals
    seed : int
        Random seed generator for the bootstrap confidence intervals.

    Returns
    --------
    fig : matplotlib Figure instance
        Matplotlib Figure. To get the individual axes, use fig.axes.

    Notes
    -----
    This function is inspired by the Matlab Robust Correlation Toolbox (Pernet,
    Wilcox and Rousselet, 2012). It uses the skipped correlation to determine
    the outliers. Note that this function requires the scikit-learn package.

    References
    ----------
    .. [1] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
           analyses: false positive and power validation using a new open
           source matlab toolbox. Front. Psychol. 3, 606.
           https://doi.org/10.3389/fpsyg.2012.00606

    Examples
    --------

    Plot a robust Skipped correlation with bootstrapped confidence intervals

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(123)
        >>> mean, cov, n = [170, 70], [[20, 10], [10, 20]], 30
        >>> x, y = np.random.multivariate_normal(mean, cov, n).T
        >>> # Introduce two outliers
        >>> x[10], y[10] = 160, 100
        >>> x[8], y[8] = 165, 90
        >>> fig = pg.plot_skipped_corr(x, y, xlabel='Height', ylabel='Weight')
    """
    from pingouin.correlation import skipped
    from scipy.stats import pearsonr
    from pingouin.effsize import compute_bootci

    # Safety check
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.2))
    # plt.subplots_adjust(wspace=0.3)
    sns.despine()

    # Scatter plot and regression lines
    sns.regplot(x[~outliers], y[~outliers], ax=ax1, color='darkcyan')
    ax1.scatter(x[outliers], y[outliers], color='indianred', label='outliers')
    ax1.scatter(x[~outliers], y[~outliers], color='seagreen', label='good')

    # Labels
    xlabel = 'x' if xlabel is None else xlabel
    ylabel = 'y' if ylabel is None else ylabel
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title('Outliers (n={})'.format(sum(outliers)), y=1.05)

    # Spearman distribution
    sns.distplot(spearman_dist, kde=True, ax=ax2, color='darkcyan')
    for i in spearman_ci:
        ax2.axvline(x=i, color='coral', lw=2)
    ax2.axvline(x=0, color='k', ls='--', lw=1.5)
    ax2.set_ylabel('Density of bootstrap samples')
    ax2.set_xlabel('Correlation coefficient')
    ax2.set_title(
        'Skipped Spearman r = {}\n95% CI = [{}, {}]'.format(r.round(2),
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
        'Skipped Pearson r = {}\n95% CI = [{}, {}]'.format(r_pearson.round(2),
                                                           pearson_ci[0],
                                                           pearson_ci[1]),
        y=1.05)

    # Optimize layout
    plt.tight_layout()

    return fig
