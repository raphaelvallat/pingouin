"""Plotting functions.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <legrand@cyceron.fr>
"""
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# Set default Seaborn preferences
sns.set(style='ticks', context='notebook')

__all__ = ["plot_skipped_corr", "qqplot"]


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


def _ppoints(n, a=0.5):
    """
    Ordinates For Probability Plotting.

    Numpy analogue or `R`'s `ppoints` function.

    Parameters
    ----------
    n : int
        Number of points generated
    a : float
        Offset fraction (typically between 0 and 1)

    Returns
    -------
    p : array
        Sequence of probabilities at which to evaluate the inverse
        distribution.
    """
    a = 3 / 8 if n <= 10 else 0.5
    return (np.arange(n) + 1 - a) / (n + 1 - 2 * a)


def qqplot(x, dist='norm', sparams=(), confidence=.95, figsize=(5, 4),
           ax=None):
    """Quantile-Quantile plot.

    Parameters
    ----------
    x : array_like
        Sample data.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name. The default is 'norm' for a
        normal probability plot. Objects that look enough like a
        `scipy.stats.distributions` instance (i.e. they have a ``ppf`` method)
        are also accepted.
    sparams : tuple, optional
        Distribution-specific shape parameters (shape parameters, location,
        and scale). See `scipy.stats.probplot` for more details.
    confidence : float
        Confidence level (.95 = 95%) for point-wise confidence envelope.
        Pass False for no envelope.
    figsize : tuple
        Figsize in inches
    ax : matplotlib axes
        Axis on which to draw the plot

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    This function returns a scatter plot of the quantile of the sample data `x`
    against the theoretical quantiles of the distribution given in `dist`
    (default = 'norm').

    The points plotted in a Q–Q plot are always non-decreasing when viewed
    from left to right. If the two distributions being compared are identical,
    the Q–Q plot follows the 45° line y = x. If the two distributions agree
    after linearly transforming the values in one of the distributions,
    then the Q–Q plot follows some line, but not necessarily the line y = x.
    If the general trend of the Q–Q plot is flatter than the line y = x,
    the distribution plotted on the horizontal axis is more dispersed than
    the distribution plotted on the vertical axis. Conversely, if the general
    trend of the Q–Q plot is steeper than the line y = x, the distribution
    plotted on the vertical axis is more dispersed than the distribution
    plotted on the horizontal axis. Q–Q plots are often arced, or "S" shaped,
    indicating that one of the distributions is more skewed than the other,
    or that one of the distributions has heavier tails than the other.

    In addition, the function also plots a best-fit line (linear regression)
    for the data and annotates the plot with the coefficient of
    determination :math:`R^2`. Note that the intercept and slope of the
    linear regression between the quantiles gives a measure of the relative
    location and relative scale of the samples.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot

    .. [2] https://github.com/cran/car/blob/master/R/qqPlot.R

    .. [3] Fox, J. (2008), Applied Regression Analysis and Generalized Linear
           Models, 2nd Ed., Sage Publications, Inc.

    Examples
    --------

    Q-Q plot using a normal theoretical distribution:

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=50)
        >>> ax = pg.qqplot(x, dist='norm')

    Two Q-Q plots using two separate axes:

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=50)
        >>> x_exp = np.random.exponential(size=50)
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        >>> ax1 = pg.qqplot(x, dist='norm', ax=ax1, confidence=False)
        >>> ax2 = pg.qqplot(x_exp, dist='expon', ax=ax2)

    Using custom location / scale parameters as well as another Seaborn style

    .. plot::

        >>> import numpy as np
        >>> import seaborn as sns
        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=50)
        >>> mean, std = 0, 0.8
        >>> sns.set_style('darkgrid')
        >>> ax = pg.qqplot(x, dist='norm', sparams=(mean, std))
    """
    if isinstance(dist, str):
        dist = getattr(stats, dist)

    x = np.asarray(x)

    # Extract quantiles and regression
    quantiles = stats.probplot(x, sparams=sparams, dist=dist, fit=False)
    theor, observed = quantiles[0], quantiles[1]

    fit_params = dist.fit(x)
    loc = fit_params[-2]
    scale = fit_params[-1]

    # Observed values to observed quantiles
    if loc != 0 and scale != 1:
        observed = (np.sort(observed) - fit_params[-2]) / fit_params[-1]

    # Linear regression
    slope, intercept, r, _, _ = stats.linregress(theor, observed)

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(theor, observed, 'bo')

    stats.morestats._add_axis_labels_title(ax,
                                           xlabel='Theoretical quantiles',
                                           ylabel='Ordered quantiles',
                                           title='Q-Q Plot')

    # Add diagonal line
    end_pts = [ax.get_xlim(), ax.get_ylim()]
    end_pts[0] = min(end_pts[0])
    end_pts[1] = max(end_pts[1])
    ax.plot(end_pts, end_pts, color='slategrey', lw=1.5)
    ax.set_xlim(end_pts)
    ax.set_ylim(end_pts)

    # Add regression line and annotate R2
    fit_val = slope * theor + intercept
    ax.plot(theor, fit_val, 'r-', lw=2)
    posx = end_pts[0] + 0.60 * (end_pts[1] - end_pts[0])
    posy = end_pts[0] + 0.10 * (end_pts[1] - end_pts[0])
    ax.text(posx, posy, "$R^2=%.3f$" % r**2)

    if confidence is not False:
        # Confidence envelope
        n = x.size
        P = _ppoints(n)
        crit = stats.norm.ppf(1 - (1 - confidence) / 2)
        se = (slope / dist.pdf(theor)) * np.sqrt(P * (1 - P) / n)
        upper = fit_val + crit * se
        lower = fit_val - crit * se
        ax.plot(theor, upper, 'r--', lw=1.25)
        ax.plot(theor, lower, 'r--', lw=1.25)

    return ax
