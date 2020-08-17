"""Plotting functions.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <legrand@cyceron.fr>
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85
# sns.set(style='ticks', context='notebook')

__all__ = ["plot_blandaltman", "qqplot", "plot_paired",
           "plot_shift", "plot_rm_corr", "plot_circmean"]


def plot_blandaltman(x, y, agreement=1.96, confidence=.95, figsize=(5, 4),
                     dpi=100, ax=None):
    """
    Generate a Bland-Altman plot to compare two sets of measurements.

    Parameters
    ----------
    x, y : np.array or list
        First and second measurements.
    agreement : float
        Multiple of the standard deviation to plot limit of agreement bounds.
        The defaults is 1.96.
    confidence : float
        If not ``None``, plot the specified percentage confidence interval on
        the mean and limits of agreement.
    figsize : tuple
        Figsize in inches
    dpi : int
        Resolution of the figure in dots per inches.
    ax : matplotlib axes
        Axis on which to draw the plot

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Bland-Altman plots [1]_ are extensively used to evaluate the agreement
    among two different instruments or two measurements techniques.
    They allow identification of any systematic difference between the
    measurements (i.e., fixed bias) or possible outliers.

    The mean difference is the estimated bias, and the SD of the differences
    measures the random fluctuations around this mean. If the mean value of the
    difference differs significantly from 0 on the basis of a 1-sample t-test,
    this indicates the presence of fixed bias. If there is a consistent bias,
    it can be adjusted for by subtracting the mean difference from the new
    method.

    It is common to compute 95% limits of agreement for each comparison
    (average difference ± 1.96 standard deviation of the difference), which
    tells us how far apart measurements by 2 methods were more likely to be
    for most individuals. If the differences within mean ± 1.96 SD are not
    clinically important, the two methods may be used interchangeably.
    The 95% limits of agreement can be unreliable estimates of the population
    parameters especially for small sample sizes so, when comparing methods
    or assessing repeatability, it is important to calculate confidence
    intervals for 95% limits of agreement.

    The code is an adaptation of the
    `PyCompare <https://github.com/jaketmp/pyCompare>`_ package. The present
    implementation is a simplified version; please refer to the original
    package for more advanced functionalities.

    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for assessing
           agreement between two methods of clinical measurement. The lancet,
           327(8476), 307-310.

    Examples
    --------
    Bland-Altman plot

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(123)
        >>> mean, cov = [10, 11], [[1, 0.8], [0.8, 1]]
        >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
        >>> ax = pg.plot_blandaltman(x, y)
    """
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    n = x.size
    mean = np.vstack((x, y)).mean(0)
    diff = x - y
    md = diff.mean()
    sd = diff.std(axis=0, ddof=1)

    # Confidence intervals
    if confidence is not None:
        assert 0 < confidence < 1
        ci = dict()
        ci['mean'] = stats.norm.interval(confidence, loc=md,
                                         scale=sd / np.sqrt(n))
        seLoA = ((1 / n) + (agreement**2 / (2 * (n - 1)))) * (sd**2)
        loARange = np.sqrt(seLoA) * stats.t.ppf((1 - confidence) / 2, n - 1)
        ci['upperLoA'] = ((md + agreement * sd) + loARange,
                          (md + agreement * sd) - loARange)
        ci['lowerLoA'] = ((md - agreement * sd) + loARange,
                          (md - agreement * sd) - loARange)

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot the mean diff, limits of agreement and scatter
    ax.axhline(md, color='#6495ED', linestyle='--')
    ax.axhline(md + agreement * sd, color='coral', linestyle='--')
    ax.axhline(md - agreement * sd, color='coral', linestyle='--')
    ax.scatter(mean, diff, alpha=0.5)

    loa_range = (md + (agreement * sd)) - (md - agreement * sd)
    offset = (loa_range / 100.0) * 1.5

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(0.98, md + offset, 'Mean', ha="right", va="bottom",
            transform=trans)
    ax.text(0.98, md - offset, '%.2f' % md, ha="right", va="top",
            transform=trans)

    ax.text(0.98, md + (agreement * sd) + offset, '+%.2f SD' % agreement,
            ha="right", va="bottom", transform=trans)
    ax.text(0.98, md + (agreement * sd) - offset,
            '%.2f' % (md + agreement * sd), ha="right", va="top",
            transform=trans)

    ax.text(0.98, md - (agreement * sd) - offset, '-%.2f SD' % agreement,
            ha="right", va="top", transform=trans)
    ax.text(0.98, md - (agreement * sd) + offset,
            '%.2f' % (md - agreement * sd), ha="right", va="bottom",
            transform=trans)

    if confidence is not None:
        ax.axhspan(ci['mean'][0], ci['mean'][1],
                   facecolor='#6495ED', alpha=0.2)

        ax.axhspan(ci['upperLoA'][0], ci['upperLoA'][1],
                   facecolor='coral', alpha=0.2)

        ax.axhspan(ci['lowerLoA'][0], ci['lowerLoA'][1],
                   facecolor='coral', alpha=0.2)

    # Labels and title
    ax.set_ylabel('Difference between methods')
    ax.set_xlabel('Mean of methods')
    ax.set_title('Bland-Altman plot')

    # Despine and trim
    sns.despine(trim=True, ax=ax)

    return ax


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
        Distribution or distribution function name. The default is `'norm'`
        for a normal probability plot.
    sparams : tuple, optional
        Distribution-specific shape parameters (shape parameters, location,
        and scale). See :py:func:`scipy.stats.probplot` for more details.
    confidence : float
        Confidence level (.95 = 95%) for point-wise confidence envelope.
        Can be disabled by passing False.
    figsize : tuple
        Figsize in inches
    ax : matplotlib axes
        Axis on which to draw the plot

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Raises
    ------
    ValueError
        If ``sparams`` does not contain the required parameters for ``dist``.
        (e.g. :py:class:`scipy.stats.t` has a mandatory degrees of
        freedom parameter *df*.)

    Notes
    -----
    This function returns a scatter plot of the quantile of the sample data
    ``x`` against the theoretical quantiles of the distribution given in
    ``dist`` (default = *'norm'*).

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

    .. warning:: Be extra careful when using fancier distributions with several
        parameters. If you can, always double-check your results with another
        software or package.

    References
    ----------
    * https://github.com/cran/car/blob/master/R/qqPlot.R

    * Fox, J. (2008), Applied Regression Analysis and Generalized Linear
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
    x = x[~np.isnan(x)]  # NaN are automatically removed

    # Check sparams: if single parameter, tuple becomes int
    if not isinstance(sparams, (tuple, list)):
        sparams = (sparams,)
    # For fancier distributions, check that the required parameters are passed
    if len(sparams) < dist.numargs:
        raise ValueError("The following sparams are required for this "
                         "distribution: %s. See scipy.stats.%s for details."
                         % (dist.shapes, dist.name))

    # Extract quantiles and regression
    quantiles = stats.probplot(x, sparams=sparams, dist=dist, fit=False)
    theor, observed = quantiles[0], quantiles[1]

    fit_params = dist.fit(x)
    loc = fit_params[-2]
    scale = fit_params[-1]
    shape = fit_params[:-2] if len(fit_params) > 2 else None

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
        pdf = dist.pdf(theor) if shape is None else dist.pdf(theor, *shape)
        se = (slope / pdf) * np.sqrt(P * (1 - P) / n)
        upper = fit_val + crit * se
        lower = fit_val - crit * se
        ax.plot(theor, upper, 'r--', lw=1.25)
        ax.plot(theor, lower, 'r--', lw=1.25)

    return ax


def plot_paired(data=None, dv=None, within=None, subject=None, order=None,
                boxplot=True, boxplot_in_front=False,
                figsize=(4, 4), dpi=100, ax=None,
                colors=['green', 'grey', 'indianred'],
                pointplot_kwargs={'scale': .6, 'markers': '.'},
                boxplot_kwargs={'color': 'lightslategrey', 'width': .2}):
    """
    Paired plot.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataFrame.
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor. Note that
        ``within`` must have exactly two within-subject levels
        (= two unique values).
    subject : string
        Name of column containing the subject identifier.
    order : list of str
        List of values in ``within`` that define the order of elements on the
        x-axis of the plot. If None, uses alphabetical order.
    boxplot : boolean
        If True, add a boxplot to the paired lines using the
        :py:func:`seaborn.boxplot` function.
    boxplot_in_front : boolean
        If True, the boxplot is plotted on the foreground (i.e. above the
        individual lines) and with a slight transparency. This makes the
        overall plot more readable when plotting a large numbers of subjects.

        .. versionadded:: 0.3.8
    figsize : tuple
        Figsize in inches
    dpi : int
        Resolution of the figure in dots per inches.
    ax : matplotlib axes
        Axis on which to draw the plot.
    colors : list of str
        Line colors names. Default is green when value increases from A to B,
        indianred when value decreases from A to B and grey when the value is
        the same in both measurements.
    pointplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        :py:func:`seaborn.pointplot` function.
    boxplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        :py:func:`seaborn.boxplot` function.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Data must be a long-format pandas DataFrame.

    Examples
    --------
    Default paired plot:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset('mixed_anova').query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> ax = pg.plot_paired(data=df, dv='Scores', within='Time',
        ...                     subject='Subject', dpi=150)

    Paired plot on an existing axis (no boxplot and uniform color):

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset('mixed_anova').query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        >>> pg.plot_paired(data=df, dv='Scores', within='Time',
        ...                subject='Subject', ax=ax1, boxplot=False,
        ...                colors=['grey', 'grey', 'grey'])  # doctest: +SKIP

    With the boxplot on the foreground:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset('mixed_anova').query("Time != 'January'")
        >>> df = df.query("Group == 'Control'")
        >>> ax = pg.plot_paired(data=df, dv='Scores', within='Time',
        ...                     subject='Subject', boxplot_in_front=True)
    """
    from pingouin.utils import _check_dataframe, remove_rm_na

    # Update default kwargs with specified inputs
    _pointplot_kwargs = {'scale': .6, 'markers': '.'}
    _pointplot_kwargs.update(pointplot_kwargs)
    _boxplot_kwargs = {'color': 'lightslategrey', 'width': .2}
    _boxplot_kwargs.update(boxplot_kwargs)

    # Set boxplot in front of Line2D plot (zorder=2 for both) and add alpha
    if boxplot_in_front:
        _boxplot_kwargs.update({
            'boxprops': {'zorder': 2},
            'whiskerprops': {'zorder': 2},
            'zorder': 2,
        })

    # Validate args
    _check_dataframe(data=data, dv=dv, within=within, subject=subject,
                     effects='within')

    # Remove NaN values
    data = remove_rm_na(dv=dv, within=within, subject=subject, data=data)

    # Extract subjects
    subj = data[subject].unique()

    # Extract within-subject level (alphabetical order)
    x_cat = np.unique(data[within])
    assert len(x_cat) == 2, 'Within must have exactly two unique levels.'

    if order is None:
        order = x_cat
    else:
        assert len(order) == 2, 'Order must have exactly two elements.'

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for idx, s in enumerate(subj):
        tmp = data.loc[data[subject] == s, [dv, within, subject]]
        x_val = tmp[tmp[within] == order[0]][dv].to_numpy()[0]
        y_val = tmp[tmp[within] == order[1]][dv].to_numpy()[0]
        if x_val < y_val:
            color = colors[0]
        elif x_val > y_val:
            color = colors[2]
        elif x_val == y_val:
            color = colors[1]

        # Plot individual lines using Seaborn
        sns.pointplot(data=tmp, x=within, y=dv, order=order, color=color,
                      ax=ax, **_pointplot_kwargs)

    if boxplot:
        sns.boxplot(data=data, x=within, y=dv, order=order, ax=ax,
                    **_boxplot_kwargs)

        # Set alpha to patch of boxplot but not to whiskers
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .75))

    # Despine and trim
    sns.despine(trim=True, ax=ax)

    return ax


def plot_shift(x, y, paired=False, n_boot=1000,
               percentiles=np.arange(10, 100, 10),
               ci=.95, seed=None, show_median=True, violin=True):
    """Shift plot.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations.
    paired : bool
        Specify whether ``x`` and ``y`` are related (i.e. repeated
        measures) or independent.

        .. versionadded:: 0.3.0
    n_boot : int
        Number of bootstrap iterations. The higher, the better, the slower.
    percentiles: array_like
        Sequence of percentiles to compute, which must be between 0 and 100
        inclusive. Default set to [10, 20, 30, 40, 50, 60, 70, 80, 90].
    ci: float
        Confidence level (0.95 = 95%).
    seed : int or None
        Random seed for generating bootstrap samples, can be integer or
        None for no seed (default).
    show_median: boolean
        If True (default), show the median with black lines.
    violin: boolean
        If True (default), plot the density of X and Y distributions.
        Defaut set to True.

    Returns
    -------
    fig : matplotlib Figure instance
        Matplotlib Figure. To get the individual axes, use fig.axes.

    See also
    --------
    harrelldavis

    Notes
    -----
    The shift plot is described in [1]_.
    It computes a shift function [2]_ for two (in)dependent groups using the
    robust Harrell-Davis quantile estimator in conjunction with bias-corrected
    bootstrap confidence intervals.

    References
    ----------
    .. [1] Rousselet, G. A., Pernet, C. R. and Wilcox, R. R. (2017). Beyond
           differences in means: robust graphical methods to compare two groups
           in neuroscience. Eur J Neurosci, 46: 1738-1748.
           doi:10.1111/ejn.13610

    .. [2] https://garstats.wordpress.com/2016/07/12/shift-function/

    Examples
    --------
    Default shift plot

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(42)
        >>> x = np.random.normal(5.5, 2, 50)
        >>> y = np.random.normal(6, 1.5, 50)
        >>> fig = pg.plot_shift(x, y)

    With different options

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(42)
        >>> x = np.random.normal(5.5, 2, 30)
        >>> y = np.random.normal(6, 1.5, 30)
        >>> fig = pg.plot_shift(x, y, paired=True, n_boot=2000,
        ...                     percentiles=[25, 50, 75],
        ...                     show_median=False, seed=456, violin=False)
    """
    from pingouin.regression import _bca
    from pingouin.nonparametric import harrelldavis as hd

    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    percentiles = np.asarray(percentiles) / 100  # Convert to 0 - 1 range
    assert x.ndim == 1, 'x must be 1D.'
    assert y.ndim == 1, 'y must be 1D.'
    nx, ny = x.size, y.size
    assert not np.isnan(x).any(), 'Missing values are not allowed.'
    assert not np.isnan(y).any(), 'Missing values are not allowed.'
    assert nx >= 10, 'x must have at least 10 samples.'
    assert ny >= 10, 'y must have at least 10 samples.'
    assert 0 < ci < 1, 'ci must be between 0 and 1.'
    if paired:
        assert nx == ny, 'x and y must have the same size when paired=True.'

    # Robust percentile
    x_per = hd(x, percentiles)
    y_per = hd(y, percentiles)
    delta = y_per - x_per

    # Compute bootstrap distribution of differences
    rng = np.random.RandomState(seed)
    if paired:
        bootsam = rng.choice(np.arange(nx), size=(nx, n_boot), replace=True)
        bootstat = (hd(y[bootsam], percentiles, axis=0) -
                    hd(x[bootsam], percentiles, axis=0))
    else:
        x_list = rng.choice(x, size=(nx, n_boot), replace=True)
        y_list = rng.choice(y, size=(ny, n_boot), replace=True)
        bootstat = (hd(y_list, percentiles, axis=0) -
                    hd(x_list, percentiles, axis=0))

    # Find upper and lower confidence interval for each quantiles
    # Bias-corrected confidence interval
    lower, median_per, upper = [], [], []
    for i, d in enumerate(delta):
        ci = _bca(bootstat[i, :], d, n_boot)
        median_per.append(_bca(bootstat[i, :], d, n_boot, alpha=1)[0])
        lower.append(ci[0])
        upper.append(ci[1])

    lower = np.asarray(lower)
    median_per = np.asarray(median_per)
    upper = np.asarray(upper)

    # Create long-format dataFrame for use with Seaborn
    data = pd.DataFrame({'value': np.concatenate([x, y]),
                         'variable': ['X'] * nx + ['Y'] * ny})

    #############################
    # Plots X and Y distributions
    #############################
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)

    # Boxplot X & Y
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    for dis, pos in zip([x, y], [1.2, -0.2]):
        qrt1, medians, qrt3 = np.percentile(dis, [25, 50, 75])
        whiskers = adjacent_values(np.sort(dis), qrt1, qrt3)
        ax1.plot(medians, pos, marker='o', color='white', zorder=10)
        ax1.hlines(pos, qrt1, qrt3, color='k',
                   linestyle='-', lw=7, zorder=9)
        ax1.hlines(pos, whiskers[0], whiskers[1],
                   color='k', linestyle='-', lw=2, zorder=9)

    ax1 = sns.stripplot(data=data, x='value', y='variable',
                        orient='h', order=['Y', 'X'],
                        palette=['#88bedc', '#cfcfcf'])

    if violin:
        vl = plt.violinplot([y, x], showextrema=False, vert=False, widths=1)

        # Upper plot
        paths = vl['bodies'][0].get_paths()[0]
        paths.vertices[:, 1][paths.vertices[:, 1] >= 1] = 1
        paths.vertices[:, 1] = paths.vertices[:, 1] - 1.2
        vl['bodies'][0].set_edgecolor('k')
        vl['bodies'][0].set_facecolor('#88bedc')
        vl['bodies'][0].set_alpha(0.8)

        # Lower plot
        paths = vl['bodies'][1].get_paths()[0]
        paths.vertices[:, 1][paths.vertices[:, 1] <= 2] = 2
        paths.vertices[:, 1] = paths.vertices[:, 1] - 0.8
        vl['bodies'][1].set_edgecolor('k')
        vl['bodies'][1].set_facecolor('#cfcfcf')
        vl['bodies'][1].set_alpha(0.8)

        # Rescale ylim
        ax1.set_ylim(2, -1)

    for i in range(len(percentiles)):
        # Connection between quantiles
        if upper[i] < 0:
            col = '#4c72b0'
        elif lower[i] > 0:
            col = '#c34e52'
        else:
            col = 'darkgray'
        plt.plot([y_per[i], x_per[i]], [0.2, 0.8],
                 marker='o', color=col, zorder=10)
        # X quantiles
        plt.plot([x_per[i], x_per[i]], [0.8, 1.2], 'k--', zorder=9)
        # Y quantiles
        plt.plot([y_per[i], y_per[i]], [-0.2, 0.2], 'k--', zorder=9)

    if show_median:
        x_med, y_med = np.median(x), np.median(y)
        plt.plot([x_med, x_med], [0.8, 1.2], 'k-')
        plt.plot([y_med, y_med], [-0.2, 0.2], 'k-')

    plt.xlabel('Scores (a.u.)', size=15)
    ax1.set_yticklabels(['Y', 'X'], size=15)
    ax1.set_ylabel('')

    #######################
    # Plots quantiles shift
    #######################
    ax2 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3)
    for i, per in enumerate(x_per):
        if upper[i] < 0:
            col = '#4c72b0'
        elif lower[i] > 0:
            col = '#c34e52'
        else:
            col = 'darkgray'
        plt.plot([per, per], [upper[i], lower[i]], lw=3, color=col, zorder=10)
        plt.plot(per, median_per[i], marker='o', ms=10, color=col, zorder=10)

    plt.axhline(y=0, ls='--', lw=2, color='gray')

    ax2.set_xlabel('X quantiles', size=15)
    ax2.set_ylabel('Y - X quantiles \n differences (a.u.)', size=10)
    sns.despine()
    plt.tight_layout()

    return fig


def plot_rm_corr(data=None, x=None, y=None, subject=None, legend=False,
                 kwargs_facetgrid=dict(height=4, aspect=1)):
    """Plot a repeated measures correlation.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Dataframe.
    x, y : string
        Name of columns in ``data`` containing the two dependent variables.
    subject : string
        Name of column in ``data`` containing the subject indicator.
    legend : boolean
        If True, add legend to plot. Legend will show all the unique values in
        ``subject``.
    kwargs_facetgrid : dict
        Optional keyword argument passed to :py:class:`seaborn.FacetGrid`

    Returns
    -------
    g : :py:class:`seaborn.FacetGrid`
        Seaborn FacetGrid.

    See also
    --------
    rm_corr

    Notes
    -----
    Repeated measures correlation [1]_ (rmcorr) is a statistical technique
    for determining the common within-individual association for paired
    measures assessed on two or more occasions for multiple individuals.

    Results have been tested against the
    `rmcorr <https://github.com/cran/rmcorr>` R package. Note that this
    function requires `statsmodels
    <https://www.statsmodels.org/stable/index.html>`_.

    Missing values are automatically removed from the ``data``
    (listwise deletion).

    References
    ----------
    .. [1] Bakdash, J.Z., Marusich, L.R., 2017. Repeated Measures Correlation.
           Front. Psychol. 8, 456. https://doi.org/10.3389/fpsyg.2017.00456

    Examples
    --------
    Default repeated mesures correlation plot

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset('rm_corr')
        >>> g = pg.plot_rm_corr(data=df, x='pH', y='PacO2', subject='Subject')

    With some tweakings

    .. plot::

        >>> import pingouin as pg
        >>> import seaborn as sns
        >>> df = pg.read_dataset('rm_corr')
        >>> sns.set(style='darkgrid', font_scale=1.2)
        >>> g = pg.plot_rm_corr(data=df, x='pH', y='PacO2',
        ...                     subject='Subject', legend=True,
        ...                     kwargs_facetgrid=dict(height=4.5, aspect=1.5,
        ...                                           palette='Spectral'))
    """
    # Check that stasmodels is installed
    from pingouin.utils import _is_statsmodels_installed
    _is_statsmodels_installed(raise_error=True)
    from statsmodels.formula.api import ols

    # Safety check (duplicated from pingouin.rm_corr)
    assert isinstance(data, pd.DataFrame), 'Data must be a DataFrame'
    assert x in data.columns, 'The %s column is not in data.' % x
    assert y in data.columns, 'The %s column is not in data.' % y
    assert data[x].dtype.kind in 'bfiu', '%s must be numeric.' % x
    assert data[y].dtype.kind in 'bfiu', '%s must be numeric.' % y
    assert subject in data.columns, 'The %s column is not in data.' % subject
    if data[subject].nunique() < 3:
        raise ValueError('rm_corr requires at least 3 unique subjects.')

    # Remove missing values
    data = data[[x, y, subject]].dropna(axis=0)

    # Calculate rm_corr
    # rmc = pg.rm_corr(data=data, x=x, y=y, subject=subject)

    # Fit ANCOVA model
    # https://patsy.readthedocs.io/en/latest/builtins-reference.html
    # C marks the data as categorical
    # Q allows to quote variable that do not meet Python variable name rule
    # e.g. if variable is "weight.in.kg" or "2A"
    formula = "Q('%s') ~ C(Q('%s')) + Q('%s')" % (y, subject, x)
    model = ols(formula, data=data).fit()

    # Fitted values
    data['pred'] = model.fittedvalues

    # Define color palette
    if 'palette' not in kwargs_facetgrid:
        kwargs_facetgrid['palette'] = sns.hls_palette(data[subject].nunique())

    # Start plot
    g = sns.FacetGrid(data, hue=subject, **kwargs_facetgrid)
    g = g.map(sns.regplot, x, "pred", scatter=False, ci=None, truncate=True)
    g = g.map(sns.scatterplot, x, y)

    if legend:
        g.add_legend()

    return g


def plot_circmean(angles, figsize=(4, 4), dpi=None, ax=None,
                  kwargs_markers=dict(color='tab:blue', marker='o',
                  mfc='none', ms=10), kwargs_arrow=dict(width=0.01,
                  head_width=0.1, head_length=0.1, fc='tab:red',
                  ec='tab:red')):
    """Plot the circular mean and vector length of a set of angles
    on the unit circle.

    .. versionadded:: 0.3.3

    Parameters
    ----------
    angles : array or list
        Angles (expressed in radians). Only 1D array are supported here.
    figsize : tuple
        Figsize in inches. Default is (4, 4).
    dpi : int
        Resolution of the figure in dots per inches.
    ax : matplotlib axes
        Axis on which to draw the plot.
    kwargs_markers : dict
        Optional keywords arguments that are passed to
        :obj:`matplotlib.axes.Axes.plot`
        to control the markers aesthetics.
    kwargs_arrow : dict
        Optional keywords arguments that are passed to
        :obj:`matplotlib.axes.Axes.arrow`
        to control the arrow aesthetics.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Examples
    --------
    Default plot

    .. plot::

        >>> import pingouin as pg
        >>> ax = pg.plot_circmean([0.05, -0.8, 1.2, 0.8, 0.5, -0.3, 0.3, 0.7])

    Changing some aesthetics parameters

    .. plot::

        >>> import pingouin as pg
        >>> ax = pg.plot_circmean([0.05, -0.8, 1.2, 0.8, 0.5, -0.3, 0.3, 0.7],
        ...                       kwargs_markers=dict(color='k', mfc='k'),
        ...                       kwargs_arrow=dict(ec='k', fc='k'))

    .. plot::

        >>> import pingouin as pg
        >>> import seaborn as sns
        >>> sns.set(font_scale=1.5, style='white')
        >>> ax = pg.plot_circmean([0.8, 1.5, 3.14, 5.2, 6.1, 2.8, 2.6, 3.2],
        ...                       kwargs_markers=dict(marker="None"))
    """
    from matplotlib.patches import Circle
    from .circular import circ_r, circ_mean

    # Sanity checks
    angles = np.asarray(angles)
    assert angles.ndim == 1, "angles must be a one-dimensional array."
    assert angles.size > 1, "angles must have at least 2 values."

    assert isinstance(kwargs_markers, dict), "kwargs_markers must be a dict."
    assert isinstance(kwargs_arrow, dict), "kwargs_arrow must be a dict."

    # Fill missing values in dict
    if 'color' not in kwargs_markers.keys():
        kwargs_markers['color'] = 'tab:blue'
    if 'marker' not in kwargs_markers.keys():
        kwargs_markers['marker'] = 'o'
    if 'mfc' not in kwargs_markers.keys():
        kwargs_markers['mfc'] = 'none'
    if 'ms' not in kwargs_markers.keys():
        kwargs_markers['ms'] = 10

    if 'width' not in kwargs_arrow.keys():
        kwargs_arrow['width'] = 0.01
    if 'head_width' not in kwargs_arrow.keys():
        kwargs_arrow['head_width'] = 0.1
    if 'head_length' not in kwargs_arrow.keys():
        kwargs_arrow['head_length'] = 0.1
    if 'fc' not in kwargs_arrow.keys():
        kwargs_arrow['fc'] = 'tab:red'
    if 'ec' not in kwargs_arrow.keys():
        kwargs_arrow['ec'] = 'tab:red'

    # Convert angles to unit vector
    z = np.exp(1j * angles)
    r = circ_r(angles)  # Resulting vector length
    phi = circ_mean(angles)  # Circular mean
    zm = r * np.exp(1j * phi)

    # Plot unit circle
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    circle = Circle((0, 0), 1, edgecolor='k', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.axvline(0, lw=1, ls=':', color='slategrey')
    ax.axhline(0, lw=1, ls=':', color='slategrey')
    ax.plot(np.real(z), np.imag(z), ls="None", **kwargs_markers)

    # Plot mean resultant vector
    ax.arrow(0, 0, np.real(zm), np.imag(zm), **kwargs_arrow)

    # X and Y ticks in radians
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.text(1.2, 0, '0', verticalalignment='center')
    ax.text(-1.3, 0, r'$\pi$', verticalalignment='center')
    ax.text(0, 1.2, r'$+\pi/2$', horizontalalignment='center')
    ax.text(0, -1.3, r'$-\pi/2$', horizontalalignment='center')
    return ax
