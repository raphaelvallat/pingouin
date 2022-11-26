import pytest
import matplotlib
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from unittest import TestCase
from pingouin import read_dataset
from pingouin.plotting import (
    plot_blandaltman,
    _ppoints,
    qqplot,
    plot_paired,
    plot_shift,
    plot_rm_corr,
    plot_circmean,
)

# Disable open figure warning
plt.close("all")  # Close all opened windows
plt.rcParams.update({"figure.max_open_warning": 0})


class TestPlotting(TestCase):
    """Test plotting.py."""

    def test_plot_blandaltman(self):
        """Test plot_blandaltman()"""
        # With random data
        np.random.seed(123)
        mean, cov = [10, 11], [[1, 0.8], [0.8, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        ax = plot_blandaltman(x, y)
        assert isinstance(ax, matplotlib.axes.Axes)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        plot_blandaltman(x, y, agreement=2, confidence=None, ax=ax1)
        plot_blandaltman(x, y, agreement=2, confidence=0.68, ax=ax2)
        plt.close("all")
        # With Pingouin's dataset
        df_ba = read_dataset("blandaltman")
        x, y = df_ba["A"], df_ba["B"]
        plot_blandaltman(x, y)
        plot_blandaltman(x, y, annotate=False)
        plot_blandaltman(x, y, xaxis="x", confidence=None)
        plot_blandaltman(x, y, xaxis="y", color="green", s=10)
        plt.close("all")

    def test_ppoints(self):
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

    def test_qqplot(self):
        """Test qqplot()"""
        np.random.seed(123)
        x = np.random.normal(size=50)
        x_ln = np.random.lognormal(size=50)
        x_exp = np.random.exponential(size=50)
        ax = qqplot(x, dist="norm")
        assert isinstance(ax, matplotlib.axes.Axes)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        qqplot(x_exp, dist="expon", ax=ax2, color="black", marker="+")
        mean, std = 0, 0.8
        qqplot(x, dist=stats.norm, sparams=(mean, std), confidence=False)
        # For lognormal distribution, the shape parameter must be specified
        ax = qqplot(x_ln, dist="lognorm", sparams=(1))
        assert isinstance(ax, matplotlib.axes.Axes)
        # Error: required parameters are not specified
        with pytest.raises(ValueError):
            qqplot(x_ln, dist="lognorm", sparams=())
        plt.close("all")

    def test_plot_paired(self):
        """Test plot_paired()"""
        df = read_dataset("mixed_anova")
        df = df.query(
            "Group == 'Meditation' and Subject > 40 and " "(Time == 'August' or Time == 'June')"
        ).copy()
        df.loc[[101, 161], "Scores"] = 6
        ax = plot_paired(data=df, dv="Scores", within="Time", subject="Subject")
        assert isinstance(ax, matplotlib.axes.Axes)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        plot_paired(data=df, dv="Scores", within="Time", subject="Subject", boxplot=False, ax=ax1)
        plot_paired(
            data=df, dv="Scores", within="Time", subject="Subject", order=["June", "August"], ax=ax2
        )
        plot_paired(
            data=df,
            dv="Scores",
            within="Time",
            subject="Subject",
            order=["June", "August"],
            boxplot_in_front=True,
            ax=ax2,
        )
        # Test with more than two within levels
        df = read_dataset("mixed_anova")
        df = df.query("Group == 'Meditation' and Subject > 40").copy()
        df.loc[[101, 161], "Scores"] = 6
        plot_paired(
            data=df,
            dv="Scores",
            within="Time",
            subject="Subject",
            order=["January", "June", "August"],
        )
        plot_paired(
            data=df,
            dv="Scores",
            within="Time",
            subject="Subject",
            order=["January", "June", "August"],
            orient="h",
        )
        plot_paired(
            data=df, dv="Scores", within="Time", subject="Subject", orient="h", boxplot=False
        )
        plt.close("all")

    def test_plot_shift(self):
        """Test plot_shift()."""
        x = np.random.normal(5.5, 2, 50)
        y = np.random.normal(6, 1.5, 50)
        plot_shift(x, y)
        plot_shift(
            x, y, n_boot=100, percentiles=[5, 55, 95], show_median=False, seed=456, violin=False
        )
        plot_shift(x, y, paired=True, n_boot=100, percentiles=[25, 75], confidence=0.90)
        plt.close("all")

    def test_plot_rm_corr(self):
        """Test plot_shift()."""
        df = read_dataset("rm_corr")
        g = plot_rm_corr(data=df, x="pH", y="PacO2", subject="Subject")
        g = plot_rm_corr(data=df, x="pH", y="PacO2", subject="Subject", legend=False)
        assert isinstance(g, sns.FacetGrid)
        plt.close("all")

    def test_plot_circmean(self):
        """Test plot_circmean.

        The MATLAB equivalent is:
        circ_plot(alpha,'pretty','ro',true,'linewidth',2,'color','r')
        """
        angles = np.array([0.02, 0.07, -0.12, 0.14, 1.2, -1.3])
        ax = plot_circmean(angles)
        assert isinstance(ax, matplotlib.axes.Axes)
        ax = plot_circmean(angles, kwargs_markers={}, kwargs_arrow={})
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")
