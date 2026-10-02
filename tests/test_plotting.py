import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from pingouin import read_dataset
from pingouin.plotting import (
    _ppoints,
    plot_blandaltman,
    plot_circmean,
    plot_paired,
    plot_rm_corr,
    qqplot,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all the figures after each test, so that they do not accumulate."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def df_paired():
    """Two within-subject levels of the mixed_anova dataset, with some unchanged values."""
    df = read_dataset("mixed_anova")
    df = df.query("Group == 'Meditation' and Subject > 40").copy()
    df.loc[[101, 161], "Scores"] = 6
    return df


def _hlines(ax):
    """Return the y-value of the horizontal lines (axhline) of an axis."""
    return [line.get_ydata()[0] for line in ax.lines]


def test_plot_blandaltman():
    """Test plot_blandaltman()"""
    # With random data
    rng = np.random.default_rng(123)
    mean, cov = [10, 11], [[1, 0.8], [0.8, 1]]
    x, y = rng.multivariate_normal(mean, cov, 30).T
    ax = plot_blandaltman(x, y)
    assert isinstance(ax, matplotlib.axes.Axes)
    # Scatter of the differences against the means
    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], (x + y) / 2)
    np.testing.assert_allclose(offsets[:, 1], x - y)
    # Zero line, mean difference and limits of agreement
    diff = x - y
    md, sd = diff.mean(), diff.std(ddof=1)
    np.testing.assert_allclose(_hlines(ax), [0, md, md + 1.96 * sd, md - 1.96 * sd])
    assert len(ax.patches) == 3  # Confidence intervals
    assert len(ax.texts) == 6  # Annotations
    assert ax.get_xlabel() == "Mean of x and y"
    assert ax.get_ylabel() == "x − y"

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    assert plot_blandaltman(x, y, agreement=2, confidence=None, ax=ax1) is ax1
    np.testing.assert_allclose(_hlines(ax1)[2:], [md + 2 * sd, md - 2 * sd])
    assert len(ax1.patches) == 0
    assert plot_blandaltman(x, y, agreement=2, confidence=0.68, ax=ax2) is ax2
    # The CI of the mean difference is a T-interval
    low, high = stats.t.interval(0.68, 29, loc=md, scale=sd / np.sqrt(30))
    np.testing.assert_allclose(ax2.patches[0].get_y(), low)
    np.testing.assert_allclose(ax2.patches[0].get_y() + ax2.patches[0].get_height(), high)
    # Narrower CI than with confidence=0.95
    assert ax2.patches[0].get_height() < ax.patches[0].get_height()

    # With Pingouin's dataset
    df_ba = read_dataset("blandaltman")
    x, y = df_ba["A"], df_ba["B"]
    _, axes = plt.subplots(2, 3)
    axes = axes.ravel()
    plot_blandaltman(x, y, ax=axes[0])
    assert axes[0].get_ylabel() == "A − B"  # Names of the pd.Series
    plot_blandaltman(x, y, annotate=False, ax=axes[1])
    assert len(axes[1].texts) == 0
    plot_blandaltman(x, y, xaxis="x", confidence=None, ax=axes[2])
    np.testing.assert_allclose(axes[2].collections[0].get_offsets()[:, 0], x)
    assert axes[2].get_xlabel() == "A"
    plot_blandaltman(x, y, xaxis="y", color="green", s=10, ax=axes[3])
    np.testing.assert_allclose(axes[3].collections[0].get_offsets()[:, 0], y)
    assert axes[3].get_xlabel() == "B"
    np.testing.assert_allclose(axes[3].collections[0].get_sizes(), 10)
    np.testing.assert_allclose(
        axes[3].collections[0].get_facecolor()[0], matplotlib.colors.to_rgba("green", 0.8)
    )
    plot_blandaltman(x, y, percentage=True, ax=axes[4])
    pct = (x - y) / ((x + y) / 2) * 100
    np.testing.assert_allclose(axes[4].collections[0].get_offsets()[:, 1], pct)
    assert axes[4].get_ylabel() == "A − B [%]"
    plot_blandaltman(x, y, percentage=True, confidence=None, annotate=False, ax=axes[5])
    assert len(axes[5].patches) == 0 and len(axes[5].texts) == 0
    # percentage=True raises ValueError when mean(x, y) == 0 for any pair
    with pytest.raises(ValueError, match="zero"):
        plot_blandaltman(np.array([1.0, -1.0]), np.array([-1.0, 1.0]), percentage=True)
    # percentage=True must preserve the sign of the difference when the data is negative
    _, ax3 = plt.subplots()
    plot_blandaltman(
        np.array([-10.0, -20.0, -30.0]),
        np.array([-11.0, -22.0, -33.0]),
        percentage=True,
        confidence=None,  # The differences have no variance
        ax=ax3,
    )
    assert (ax3.collections[0].get_offsets()[:, 1] > 0).all()
    # The y-axis is only symmetric around zero when explicitly requested
    _, (ax4, ax5) = plt.subplots(1, 2, figsize=(9, 4))
    plot_blandaltman(x, y, ax=ax4)
    low, high = ax4.get_ylim()
    assert abs(low) != pytest.approx(abs(high))
    plot_blandaltman(x, y, symmetric_ylim=True, ax=ax5)
    low, high = ax5.get_ylim()
    assert low == -high


def test_ppoints():
    """Test _ppoints()"""
    R_test_5 = [0.1190476, 0.3095238, 0.5, 0.6904762, 0.8809524]
    R_test_15 = [
        0.03333333,
        0.10000000,
        0.16666667,
        0.23333333,
        0.30000000,
        0.36666667,
        0.43333333,
        0.50000000,
        0.56666667,
        0.63333333,
        0.70000000,
        0.76666667,
        0.83333333,
        0.90000000,
        0.96666667,
    ]

    np.testing.assert_array_almost_equal(_ppoints(5), R_test_5)
    np.testing.assert_array_almost_equal(_ppoints(15), R_test_15)


def test_qqplot():
    """Test qqplot()"""
    rng = np.random.RandomState(123)
    x = rng.normal(size=50)
    x_ln = rng.lognormal(size=50)
    x_exp = rng.exponential(size=50)
    ax = qqplot(x, dist="norm")
    assert isinstance(ax, matplotlib.axes.Axes)
    # Scatter of the theoretical quantiles against the standardized ordered values
    theor, observed = stats.probplot(x, fit=False)
    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], theor)
    np.testing.assert_allclose(offsets[:, 1], (observed - x.mean()) / x.std())
    # Identity line, fitted line and the two confidence bands
    assert len(ax.lines) == 4
    r2 = stats.linregress(offsets[:, 0], offsets[:, 1]).rvalue ** 2
    assert ax.texts[0].get_text() == f"$R^2={r2:.3f}$"
    assert ax.get_aspect() == 1  # square=True
    upper, lower = ax.lines[2].get_ydata(), ax.lines[3].get_ydata()
    assert (upper > ax.lines[1].get_ydata()).all() and (lower < ax.lines[1].get_ydata()).all()

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    assert qqplot(x_exp, dist="expon", ax=ax2, color="black", marker="+") is ax2
    np.testing.assert_allclose(ax2.collections[0].get_facecolor()[0], (0, 0, 0, 1))
    mean, std = 0, 0.8
    ax = qqplot(x, dist=stats.norm, sparams=(mean, std), confidence=False, ax=ax1)
    assert len(ax.lines) == 2  # No confidence bands
    # For lognormal distribution, the shape parameter must be specified
    _, ax = plt.subplots()
    ax = qqplot(x_ln, dist="lognorm", sparams=(1), ax=ax)
    theor, _ = stats.probplot(x_ln, sparams=(1,), dist=stats.lognorm, fit=False)
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], theor)
    # Error: required parameters are not specified
    with pytest.raises(ValueError):
        qqplot(x_ln, dist="lognorm", sparams=())
    # Custom line and CI kwargs
    _, ax = plt.subplots()
    qqplot(x, line_kwargs={"color": "k", "lw": 1}, ci_kwargs={"color": "k", "ls": ":"}, ax=ax)
    fit_line, ci_line = ax.lines[1], ax.lines[2]
    assert fit_line.get_color() == "k" and fit_line.get_linewidth() == 1
    assert ci_line.get_color() == "k" and ci_line.get_linestyle() == ":"
    # Canonical Matplotlib names must override the aliases used in the defaults
    _, ax = plt.subplots()
    qqplot(x, line_kwargs={"linewidth": 1}, ci_kwargs={"linestyle": ":", "linewidth": 1}, ax=ax)
    assert ax.lines[1].get_linewidth() == 1 and ax.lines[1].get_color() == "r"
    assert ax.lines[2].get_linestyle() == ":" and ax.lines[2].get_linewidth() == 1
    # square=False
    _, ax = plt.subplots()
    assert qqplot(x, square=False, ax=ax).get_aspect() == "auto"
    # An exact identity fit (loc = 0, scale = 1) skips the standardization
    assert isinstance(qqplot(np.array([-1.0, 1.0])), matplotlib.axes.Axes)
    # Zero-variance input raises ValueError instead of dividing by zero
    with pytest.raises(ValueError, match="identical"):
        qqplot(np.zeros(20))


def test_plot_paired(df_paired):
    """Test plot_paired()"""
    df = df_paired.query("Time == 'August' or Time == 'June'")
    n_subj = df["Subject"].nunique()
    ax = plot_paired(data=df, dv="Scores", within="Time", subject="Subject")
    assert isinstance(ax, matplotlib.axes.Axes)
    # One line per subject, and one box per level
    assert len(ax.lines) >= n_subj
    assert [t.get_text() for t in ax.get_xticklabels()] == ["August", "June"]
    # Colors: green when the value increases, indianred when it decreases, grey otherwise
    wide = df.pivot_table(index="Subject", columns="Time", values="Scores")
    n_same = (wide["August"] == wide["June"]).sum()
    assert n_same > 0
    grey = matplotlib.colors.to_rgba("grey")
    colors = [matplotlib.colors.to_rgba(line.get_color()) for line in ax.lines[:n_subj]]
    assert sum(np.allclose(c, grey) for c in colors) == n_same

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    plot_paired(data=df, dv="Scores", within="Time", subject="Subject", boxplot=False, ax=ax1)
    assert len(ax1.patches) == 0  # No boxplot
    assert ax1.get_xlim() == (-0.5, 1.5)
    assert ax1.get_xlabel() == "Time" and ax1.get_ylabel() == "Scores"
    plot_paired(
        data=df, dv="Scores", within="Time", subject="Subject", order=["June", "August"], ax=ax2
    )
    assert [t.get_text() for t in ax2.get_xticklabels()] == ["June", "August"]
    # Explicit colors (exercises the colors-is-not-None branch)
    _, ax = plt.subplots()
    plot_paired(
        data=df,
        dv="Scores",
        within="Time",
        subject="Subject",
        colors=["blue", "black", "red"],
        boxplot=False,
        ax=ax,
    )
    line_colors = {matplotlib.colors.to_hex(line.get_color()) for line in ax.lines}
    assert line_colors == {"#0000ff", "#000000", "#ff0000"}
    # Patches already present on a user-supplied axis must not be restyled
    _, ax3 = plt.subplots()
    span = ax3.axvspan(-0.5, 0.5, facecolor="orange")
    facecolor = span.get_facecolor()
    plot_paired(data=df, dv="Scores", within="Time", subject="Subject", ax=ax3)
    assert span.get_facecolor() == facecolor
    # Mismatched order length raises ValueError
    with pytest.raises(ValueError):
        plot_paired(
            data=df,
            dv="Scores",
            within="Time",
            subject="Subject",
            order=["June", "August", "Extra"],
        )
    # Boxplot in front of the lines
    _, ax = plt.subplots()
    plot_paired(
        data=df,
        dv="Scores",
        within="Time",
        subject="Subject",
        order=["June", "August"],
        boxplot_in_front=True,
        ax=ax,
    )
    assert max(p.get_zorder() for p in ax.patches) == 3
    # Test with more than two within levels
    order = ["January", "June", "August"]
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_paired(data=df_paired, dv="Scores", within="Time", subject="Subject", order=order, ax=ax1)
    assert [t.get_text() for t in ax1.get_xticklabels()] == order
    plot_paired(
        data=df_paired,
        dv="Scores",
        within="Time",
        subject="Subject",
        order=order,
        orient="h",
        ax=ax2,
    )
    assert [t.get_text() for t in ax2.get_yticklabels()] == order
    plot_paired(
        data=df_paired,
        dv="Scores",
        within="Time",
        subject="Subject",
        orient="h",
        boxplot=False,
        ax=ax3,
    )
    assert ax3.get_ylim() == (2.5, -0.5)  # Inverted y-axis
    assert ax3.get_xlabel() == "Scores" and ax3.get_ylabel() == "Time"


def test_plot_rm_corr():
    """Test plot_rm_corr()."""
    df = read_dataset("rm_corr")
    n_subj = df["Subject"].nunique()
    g = plot_rm_corr(data=df, x="pH", y="PacO2", subject="Subject", legend=False)
    assert isinstance(g, sns.FacetGrid)
    assert g.legend is None
    # One fitted line per subject, all with the same slope
    assert len(g.ax.lines) == n_subj
    slopes = [np.polyfit(*line.get_xydata().T, 1)[0] for line in g.ax.lines]
    np.testing.assert_allclose(slopes, slopes[0])
    # legend=True exercises g.add_legend()
    g = plot_rm_corr(data=df, x="pH", y="PacO2", subject="Subject", legend=True)
    assert isinstance(g, sns.FacetGrid)
    assert len(g.legend.get_texts()) == n_subj
    # Passing a palette in kwargs_facetgrid exercises the palette-already-set branch
    g = plot_rm_corr(
        data=df,
        x="pH",
        y="PacO2",
        subject="Subject",
        kwargs_facetgrid={"height": 4, "aspect": 1, "palette": "Set2"},
    )
    assert isinstance(g, sns.FacetGrid)
    color = matplotlib.colors.to_rgb(g.ax.lines[0].get_color())
    np.testing.assert_allclose(color, sns.color_palette("Set2")[0])
    # Fitted lines are the same as the ANCOVA fitted values, with any column name
    df_quote = df.rename(columns={"pH": "patient's pH", "PacO2": "C", "Subject": "Q"})
    g = plot_rm_corr(data=df_quote, x="patient's pH", y="C", subject="Q")
    ols = sm.OLS(df["PacO2"], pd.get_dummies(df["Subject"], dtype=float).assign(pH=df["pH"]))
    pred = ols.fit().fittedvalues
    first = df["Subject"] == df["Subject"].min()  # First hue level
    line = g.ax.lines[0].get_xydata()
    slope, intercept = np.polyfit(line[:, 0], line[:, 1], 1)
    np.testing.assert_allclose(intercept + slope * df.loc[first, "pH"], pred[first])
    # Fewer than 3 subjects raises ValueError
    with pytest.raises(ValueError):
        plot_rm_corr(
            data=df.query("Subject in [1, 2]"),
            x="pH",
            y="PacO2",
            subject="Subject",
        )


def test_plot_circmean():
    """Test plot_circmean.

    The MATLAB equivalent is:
    circ_plot(alpha,'pretty','ro',true,'linewidth',2,'color','r')
    """
    angles = np.array([0.02, 0.07, -0.12, 0.14, 1.2, -1.3])
    ax = plot_circmean(angles)
    assert isinstance(ax, matplotlib.axes.Axes)
    # The angles are plotted on the unit circle
    markers = ax.lines[-1]
    np.testing.assert_allclose(markers.get_xdata(), np.cos(angles))
    np.testing.assert_allclose(markers.get_ydata(), np.sin(angles))
    assert markers.get_marker() == "o" and markers.get_markersize() == 10
    assert ax.get_aspect() == 1  # square=True
    _, ax = plt.subplots()
    ax = plot_circmean(angles, kwargs_markers={}, kwargs_arrow={}, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    # Canonical Matplotlib names must override the aliases used in the defaults
    _, ax = plt.subplots()
    ax = plot_circmean(
        angles, kwargs_markers={"markersize": 5}, kwargs_arrow={"facecolor": "k"}, ax=ax
    )
    assert ax.lines[-1].get_markersize() == 5
    arrow = ax.patches[-1]
    np.testing.assert_allclose(arrow.get_facecolor(), (0, 0, 0, 1))
    np.testing.assert_allclose(arrow.get_edgecolor(), matplotlib.colors.to_rgba("tab:red"))
    # Non-dict kwargs raise TypeError
    with pytest.raises(TypeError):
        plot_circmean(angles, kwargs_markers="red")
    with pytest.raises(TypeError):
        plot_circmean(angles, kwargs_arrow="red")
    # Explicit ax exercises the ax-is-not-None branch; square=False skips set_aspect
    _, ax2 = plt.subplots(1, 1)
    ax = plot_circmean(angles, ax=ax2, square=False)
    assert ax is ax2
    assert ax.get_aspect() == "auto"
