"""Plotting functions.

Authors
- Raphael Vallat <raphaelvallat9@gmail.com>
- Nicolas Legrand <legrand@cyceron.fr>
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85
# sns.set(style='ticks', context='notebook')

__all__ = [
    "plot_blandaltman",
    "qqplot",
    "plot_paired",
    "plot_rm_corr",
    "plot_circmean",
]


def plot_blandaltman(
    x, y, agreement=1.96, xaxis="mean", confidence=0.95, annotate=True, ax=None, **kwargs
):
    """
    Generate a Bland-Altman plot to compare two sets of measurements.

    Parameters
    ----------
    x, y : pd.Series, np.array, or list
        First and second measurements.
    agreement : float
        Multiple of the standard deviation to plot agreement limits.
        The defaults is 1.96, which corresponds to 95% confidence interval if
        the differences are normally distributed.
    xaxis : str
        Define which measurements should be used as the reference (x-axis).
        Default is to use the average of x and y ("mean"). Accepted values are
        "mean", "x" or "y".
    confidence : float
        If not None, plot the specified percentage confidence interval of
        the mean and limits of agreement. The CIs of the mean difference and
        agreement limits describe a possible error in the
        estimate due to a sampling error. The greater the sample size,
        the narrower the CIs will be.
    annotate : bool
        If True (default), annotate the values for the mean difference
        and agreement limits.
    ax : matplotlib axes
        Axis on which to draw the plot.
    **kwargs : optional
        Optional argument(s) passed to :py:func:`matplotlib.pyplot.scatter`.

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

    The mean difference (= x - y) is the estimated bias, and the SD of the
    differences measures the random fluctuations around this mean.
    If the mean value of the difference differs significantly from 0 on the
    basis of a 1-sample t-test, this indicates the presence of fixed bias.
    If there is a consistent bias, it can be adjusted for by subtracting the
    mean difference from the new method.

    It is common to compute 95% limits of agreement for each comparison
    (average difference ± 1.96 standard deviation of the difference), which
    tells us how far apart measurements by 2 methods were more likely to be
    for most individuals. If the differences within mean ± 1.96 SD are not
    clinically important, the two methods may be used interchangeably.
    The 95% limits of agreement can be unreliable estimates of the population
    parameters especially for small sample sizes so, when comparing methods
    or assessing repeatability, it is important to calculate confidence
    intervals for the 95% limits of agreement.

    The code is an adaptation of the
    `PyCompare <https://github.com/jaketmp/pyCompare>`_ package. The present
    implementation is a simplified version; please refer to the original
    package for more advanced functionalities.

    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for assessing
           agreement between two methods of clinical measurement. The lancet,
           327(8476), 307-310.

    .. [2] Giavarina, D. (2015). Understanding bland altman analysis.
           Biochemia medica, 25(2), 141-151.

    Examples
    --------
    Bland-Altman plot (example data from [2]_)

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("blandaltman")
        >>> ax = pg.plot_blandaltman(df["A"], df["B"])
        >>> plt.tight_layout()
    """
    # Safety check
    assert xaxis in ["mean", "x", "y"]
    # Get names before converting to NumPy array
    xname = x.name if isinstance(x, pd.Series) else "x"
    yname = y.name if isinstance(y, pd.Series) else "y"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    assert not np.isnan(x).any(), "Missing values in x or y are not supported."
    assert not np.isnan(y).any(), "Missing values in x or y are not supported."

    # Update default kwargs with specified inputs
    _scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
    _scatter_kwargs.update(kwargs)

    # Calculate mean, STD and SEM of x - y
    n = x.size
    dof = n - 1
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # Define x-axis
    if xaxis == "mean":
        xval = np.vstack((x, y)).mean(0)
        xlabel = f"Mean of {xname} and {yname}"
    elif xaxis == "x":
        xval = x
        xlabel = xname
    else:
        xval = y
        xlabel = yname

    # Start the plot
    if ax is None:
        ax = plt.gca()

    # Plot the mean diff, limits of agreement and scatter
    ax.scatter(xval, diff, **_scatter_kwargs)
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    if annotate:
        loa_range = high - low
        offset = (loa_range / 100.0) * 1.5
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        xloc = 0.98
        ax.text(xloc, mean_diff + offset, "Mean", ha="right", va="bottom", transform=trans)
        ax.text(xloc, mean_diff - offset, "%.2f" % mean_diff, ha="right", va="top", transform=trans)
        ax.text(
            xloc, high + offset, "+%.2f SD" % agreement, ha="right", va="bottom", transform=trans
        )
        ax.text(xloc, high - offset, "%.2f" % high, ha="right", va="top", transform=trans)
        ax.text(xloc, low - offset, "-%.2f SD" % agreement, ha="right", va="top", transform=trans)
        ax.text(xloc, low + offset, "%.2f" % low, ha="right", va="bottom", transform=trans)

    # Add 95% confidence intervals for mean bias and limits of agreement
    if confidence is not None:
        assert 0 < confidence < 1
        ci = dict()
        ci["mean"] = stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se)
        ci["high"] = stats.t.interval(confidence, dof, loc=high, scale=high_low_se)
        ci["low"] = stats.t.interval(confidence, dof, loc=low, scale=high_low_se)
        ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
        ax.axhspan(ci["high"][0], ci["high"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)
        ax.axhspan(ci["low"][0], ci["low"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)

    # Labels
    ax.set_ylabel(f"{xname} - {yname}")
    ax.set_xlabel(xlabel)
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


def qqplot(x, dist="norm", sparams=(), confidence=0.95, square=True, ax=None, **kwargs):
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
    square: bool
        If True (default), ensure equal aspect ratio between X and Y axes.
    ax : matplotlib axes
        Axis on which to draw the plot
    **kwargs : optional
        Optional argument(s) passed to :py:func:`matplotlib.pyplot.scatter`.

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
        parameters. Always double-check your results with another
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
        >>> ax = pg.qqplot(x, dist="norm")

    Two Q-Q plots using two separate axes:

    .. plot::

        >>> import numpy as np
        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=50)
        >>> x_exp = np.random.exponential(size=50)
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        >>> ax1 = pg.qqplot(x, dist="norm", ax=ax1, confidence=False)
        >>> ax2 = pg.qqplot(x_exp, dist="expon", ax=ax2)

    Using custom location / scale parameters as well as another Seaborn style

    .. plot::

        >>> import numpy as np
        >>> import seaborn as sns
        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(123)
        >>> x = np.random.normal(size=50)
        >>> mean, std = 0, 0.8
        >>> sns.set_style("darkgrid")
        >>> ax = pg.qqplot(x, dist="norm", sparams=(mean, std))
    """
    # Update default kwargs with specified inputs
    _scatter_kwargs = {"marker": "o", "color": "blue"}
    _scatter_kwargs.update(kwargs)

    if isinstance(dist, str):
        dist = getattr(stats, dist)

    x = np.asarray(x)
    x = x[~np.isnan(x)]  # NaN are automatically removed

    # Check sparams: if single parameter, tuple becomes int
    if not isinstance(sparams, (tuple, list)):
        sparams = (sparams,)
    # For fancier distributions, check that the required parameters are passed
    if len(sparams) < dist.numargs:
        raise ValueError(
            "The following sparams are required for this "
            "distribution: %s. See scipy.stats.%s for details." % (dist.shapes, dist.name)
        )

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
        ax = plt.gca()

    ax.scatter(theor, observed, **_scatter_kwargs)

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered quantiles")

    # Add diagonal line
    end_pts = [ax.get_xlim(), ax.get_ylim()]
    end_pts[0] = min(end_pts[0])
    end_pts[1] = max(end_pts[1])
    ax.plot(end_pts, end_pts, color="slategrey", lw=1.5)
    ax.set_xlim(end_pts)
    ax.set_ylim(end_pts)

    # Add regression line and annotate R2
    fit_val = slope * theor + intercept
    ax.plot(theor, fit_val, "r-", lw=2)
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
        ax.plot(theor, upper, "r--", lw=1.25)
        ax.plot(theor, lower, "r--", lw=1.25)

    # Make square
    if square:
        ax.set_aspect("equal")

    return ax


def plot_paired(
    data=None,
    dv=None,
    within=None,
    subject=None,
    order=None,
    boxplot=True,
    boxplot_in_front=False,
    orient="v",
    ax=None,
    colors=["green", "grey", "indianred"],
    pointplot_kwargs={"scale": 0.6, "marker": "."},
    boxplot_kwargs={"color": "lightslategrey", "width": 0.2},
):
    """
    Paired plot.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataFrame.
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor.
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
    orient : string
        Plot the boxplots vertically and the subjects on the x-axis if
        ``orient='v'`` (default). Set to ``orient='h'`` to rotate the plot by
        by 90 degrees.

        .. versionadded:: 0.3.9
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
    Data must be a long-format pandas DataFrame. Missing values are automatically removed using a
    strict listwise approach (= complete-case analysis).

    Examples
    --------
    Default paired plot:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> ax = pg.plot_paired(data=df, dv="Scores", within="Time", subject="Subject")

    Paired plot on an existing axis (no boxplot and uniform color):

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        >>> pg.plot_paired(
        ...     data=df,
        ...     dv="Scores",
        ...     within="Time",
        ...     subject="Subject",
        ...     ax=ax1,
        ...     boxplot=False,
        ...     colors=["grey", "grey", "grey"],
        ... )  # doctest: +SKIP

    Horizontal paired plot with three unique within-levels:

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Group == 'Meditation'")
        >>> # df = df.query("Group == 'Meditation' and Subject > 40")
        >>> pg.plot_paired(
        ...     data=df, dv="Scores", within="Time", subject="Subject", orient="h"
        ... )  # doctest: +SKIP

    With the boxplot on the foreground:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Control'")
        >>> ax = pg.plot_paired(
        ...     data=df, dv="Scores", within="Time", subject="Subject", boxplot_in_front=True
        ... )
    """
    from pingouin.utils import _check_dataframe

    # Update default kwargs with specified inputs
    _pointplot_kwargs = {"scale": 0.6, "marker": "."}
    _pointplot_kwargs.update(pointplot_kwargs)
    _boxplot_kwargs = {"color": "lightslategrey", "width": 0.2}
    _boxplot_kwargs.update(boxplot_kwargs)
    # Extract pointplot alpha, if set
    pp_alpha = _pointplot_kwargs.pop("alpha", 1.0)

    # Calculate size of the plot elements by scale as in Seaborn pointplot
    scale = _pointplot_kwargs.pop("scale")
    lw = plt.rcParams["lines.linewidth"] * 1.8 * scale  # get the linewidth
    mew = lw * 0.75  # get the markeredgewidth
    markersize = np.pi * np.square(lw) * 2  # get the markersize

    # Set boxplot in front of Line2D plot (zorder=2 for both) and add alpha
    if boxplot_in_front:
        _boxplot_kwargs.update(
            {
                "boxprops": {"zorder": 3},  # Boxplot on top
                "whiskerprops": {"zorder": 3},
                "zorder": 3,
            }
        )
    else:
        _boxplot_kwargs.update(
            {
                "boxprops": {"zorder": 1},  # Boxplot behind
                "whiskerprops": {"zorder": 1},
                "zorder": 1,
            }
        )

    # Validate args
    data = _check_dataframe(data=data, dv=dv, within=within, subject=subject, effects="within")

    # Pivot and melt the table. This has several effects:
    # 1) Force missing values to be explicit (a NaN cell is created)
    # 2) Automatic collapsing to the mean if multiple within factors are present
    # 3) If using dropna, remove rows with missing values (listwise deletion).
    # The latter is the same behavior as JASP (= strict complete-case analysis).
    data_piv = data.pivot_table(index=subject, columns=within, values=dv, observed=True)
    data_piv = data_piv.dropna()
    data = data_piv.melt(ignore_index=False, value_name=dv).reset_index()

    # Extract within-subject level (alphabetical order)
    x_cat = np.unique(data[within])

    if order is None:
        order = x_cat
    else:
        assert len(order) == len(x_cat), (
            "Order must have the same number of elements as the number of levels in `within`."
        )

    # Substitue within by integer order of the ordered columns to allow for
    # changing the order of numeric withins.
    data["wthn"] = data[within].replace({_ordr: i for i, _ordr in enumerate(order)})
    order_num = range(len(order))  # Make numeric order

    # Start the plot
    if ax is None:
        ax = plt.gca()

    # Set x and y depending on orientation using the num. replacement within
    _x = "wthn" if orient == "v" else dv
    _y = dv if orient == "v" else "wthn"

    for cat in range(len(x_cat) - 1):
        _order = (order_num[cat], order_num[cat + 1])
        # Extract data of the current subject-combination
        data_now = data.loc[data["wthn"].isin(_order), [dv, "wthn", subject]]
        # Select colors for all lines between the current subjects
        y1 = data_now.loc[data_now["wthn"] == _order[0], dv].to_numpy()
        y2 = data_now.loc[data_now["wthn"] == _order[1], dv].to_numpy()
        # Line and scatter colors depending on subject dv trend
        _colors = np.where(y1 < y2, colors[0], np.where(y1 > y2, colors[2], colors[1]))
        # Line and scatter colors as hue-indexed dictionary
        _colors = {subj: clr for subj, clr in zip(data_now[subject].unique(), _colors)}
        # Plot individual lines using Seaborn
        sns.lineplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            ls="-",
            lw=lw,
            legend=False,
            ax=ax,
        )
        # Plot individual markers using Seaborn
        sns.scatterplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            edgecolor="face",
            lw=mew,
            sizes=[markersize] * data_now.shape[0],
            legend=False,
            ax=ax,
            **_pointplot_kwargs,
        )

    # Set zorder and alpha of pointplot markers and lines
    _ = plt.setp(ax.collections, alpha=pp_alpha, zorder=2)  # Set marker alpha
    _ = plt.setp(ax.lines, alpha=pp_alpha, zorder=2)  # Set line alpha

    if boxplot:
        # Set boxplot x and y depending on orientation
        _xbp = within if orient == "v" else dv
        _ybp = dv if orient == "v" else within
        sns.boxplot(data=data, x=_xbp, y=_ybp, order=order, ax=ax, orient=orient, **_boxplot_kwargs)

        # Set alpha to patch of boxplot but not to whiskers
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.75))
    else:
        # If no boxplot, axis needs manual styling as in Seaborn pointplot
        if orient == "v":
            xlabel, ylabel = within, dv
            ax.set_xticks(np.arange(len(x_cat)))
            ax.set_xticklabels(order)
            ax.xaxis.grid(False)
            ax.set_xlim(-0.5, len(x_cat) - 0.5, auto=None)
        else:
            xlabel, ylabel = dv, within
            ax.set_yticks(np.arange(len(x_cat)))
            ax.set_yticklabels(order)
            ax.yaxis.grid(False)
            ax.set_ylim(-0.5, len(x_cat) - 0.5, auto=None)
            ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    return ax


def plot_rm_corr(
    data=None,
    x=None,
    y=None,
    subject=None,
    legend=False,
    kwargs_facetgrid=dict(height=4, aspect=1),
    kwargs_line=dict(ls="solid"),
    kwargs_scatter=dict(marker="o"),
):
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
        Optional keyword arguments passed to :py:class:`seaborn.FacetGrid`
    kwargs_line : dict
        Optional keyword arguments passed to :py:class:`matplotlib.pyplot.plot`
    kwargs_scatter : dict
        Optional keyword arguments passed to :py:class:`matplotlib.pyplot.scatter`

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
        >>> df = pg.read_dataset("rm_corr")
        >>> g = pg.plot_rm_corr(data=df, x="pH", y="PacO2", subject="Subject")

    With some tweakings

    .. plot::

        >>> import pingouin as pg
        >>> import seaborn as sns
        >>> df = pg.read_dataset("rm_corr")
        >>> sns.set_theme(style="darkgrid", font_scale=1.2)
        >>> g = pg.plot_rm_corr(
        ...     data=df,
        ...     x="pH",
        ...     y="PacO2",
        ...     subject="Subject",
        ...     legend=True,
        ...     kwargs_facetgrid=dict(height=4.5, aspect=1.5, palette="Spectral"),
        ... )
    """
    # Check that stasmodels is installed
    from pingouin.utils import _is_statsmodels_installed

    _is_statsmodels_installed(raise_error=True)
    from statsmodels.formula.api import ols

    # Safety check (duplicated from pingouin.rm_corr)
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
    assert x in data.columns, "The %s column is not in data." % x
    assert y in data.columns, "The %s column is not in data." % y
    assert data[x].dtype.kind in "bfiu", "%s must be numeric." % x
    assert data[y].dtype.kind in "bfiu", "%s must be numeric." % y
    assert subject in data.columns, "The %s column is not in data." % subject
    if data[subject].nunique() < 3:
        raise ValueError("rm_corr requires at least 3 unique subjects.")

    # Remove missing values
    data = data[[x, y, subject]].dropna(axis=0)

    # Calculate rm_corr
    # rmc = pg.rm_corr(data=data, x=x, y=y, subject=subject)

    # Fit ANCOVA model
    # https://patsy.readthedocs.io/en/latest/builtins-reference.html
    # C marks the data as categorical
    # Q allows to quote variable that do not meet Python variable name rule
    # e.g. if variable is "weight.in.kg" or "2A"
    assert x not in ["C", "Q"], "`x` must not be 'C' or 'Q'."
    assert y not in ["C", "Q"], "`y` must not be 'C' or 'Q'."
    assert subject not in ["C", "Q"], "`subject` must not be 'C' or 'Q'."
    formula = f"Q('{y}') ~ C(Q('{subject}')) + Q('{x}')"
    model = ols(formula, data=data).fit()

    # Fitted values
    data["pred"] = model.fittedvalues

    # Define color palette
    if "palette" not in kwargs_facetgrid:
        kwargs_facetgrid["palette"] = sns.hls_palette(data[subject].nunique())

    # Start plot
    g = sns.FacetGrid(data, hue=subject, **kwargs_facetgrid)
    g = g.map(sns.regplot, x, "pred", scatter=False, ci=None, truncate=True, line_kws=kwargs_line)
    g = g.map(sns.scatterplot, x, y, **kwargs_scatter)

    if legend:
        g.add_legend()

    return g


def plot_circmean(
    angles,
    square=True,
    ax=None,
    kwargs_markers=dict(color="tab:blue", marker="o", mfc="none", ms=10),
    kwargs_arrow=dict(width=0.01, head_width=0.1, head_length=0.1, fc="tab:red", ec="tab:red"),
):
    """Plot the circular mean and vector length of a set of angles
    on the unit circle.

    .. versionadded:: 0.3.3

    Parameters
    ----------
    angles : array or list
        Angles (expressed in radians). Only 1D array are supported here.
    square: bool
        If True (default), ensure equal aspect ratio between X and Y axes.
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
        >>> import matplotlib.pyplot as plt
        >>> _, ax = plt.subplots(1, 1, figsize=(3, 3))
        >>> ax = pg.plot_circmean(
        ...     [0.05, -0.8, 1.2, 0.8, 0.5, -0.3, 0.3, 0.7],
        ...     kwargs_markers=dict(color="k", mfc="k"),
        ...     kwargs_arrow=dict(ec="k", fc="k"),
        ...     ax=ax,
        ... )

    .. plot::

        >>> import pingouin as pg
        >>> import seaborn as sns
        >>> sns.set_theme(font_scale=1.5, style="white")
        >>> ax = pg.plot_circmean(
        ...     [0.8, 1.5, 3.14, 5.2, 6.1, 2.8, 2.6, 3.2], kwargs_markers=dict(marker="None")
        ... )
    """
    from matplotlib.patches import Circle

    from .circular import circ_mean, circ_r

    # Sanity checks
    angles = np.asarray(angles)
    assert angles.ndim == 1, "angles must be a one-dimensional array."
    assert angles.size > 1, "angles must have at least 2 values."

    assert isinstance(kwargs_markers, dict), "kwargs_markers must be a dict."
    assert isinstance(kwargs_arrow, dict), "kwargs_arrow must be a dict."

    # Fill missing values in dict
    if "color" not in kwargs_markers.keys():
        kwargs_markers["color"] = "tab:blue"
    if "marker" not in kwargs_markers.keys():
        kwargs_markers["marker"] = "o"
    if "mfc" not in kwargs_markers.keys():
        kwargs_markers["mfc"] = "none"
    if "ms" not in kwargs_markers.keys():
        kwargs_markers["ms"] = 10

    if "width" not in kwargs_arrow.keys():
        kwargs_arrow["width"] = 0.01
    if "head_width" not in kwargs_arrow.keys():
        kwargs_arrow["head_width"] = 0.1
    if "head_length" not in kwargs_arrow.keys():
        kwargs_arrow["head_length"] = 0.1
    if "fc" not in kwargs_arrow.keys():
        kwargs_arrow["fc"] = "tab:red"
    if "ec" not in kwargs_arrow.keys():
        kwargs_arrow["ec"] = "tab:red"

    # Convert angles to unit vector
    z = np.exp(1j * angles)
    r = circ_r(angles)  # Resulting vector length
    phi = circ_mean(angles)  # Circular mean
    zm = r * np.exp(1j * phi)

    # Plot unit circle
    if ax is None:
        ax = plt.gca()
    circle = Circle((0, 0), 1, edgecolor="k", facecolor="none", linewidth=2)
    ax.add_patch(circle)
    ax.axvline(0, lw=1, ls=":", color="slategrey")
    ax.axhline(0, lw=1, ls=":", color="slategrey")
    ax.plot(np.real(z), np.imag(z), ls="None", **kwargs_markers)

    # Plot mean resultant vector
    ax.arrow(0, 0, np.real(zm), np.imag(zm), **kwargs_arrow)

    # X and Y ticks in radians
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(1.2, 0, "0", verticalalignment="center")
    ax.text(-1.3, 0, r"$\pi$", verticalalignment="center")
    ax.text(0, 1.2, r"$+\pi/2$", horizontalalignment="center")
    ax.text(0, -1.3, r"$-\pi/2$", horizontalalignment="center")

    # Make square
    if square:
        ax.set_aspect("equal")
    return ax
