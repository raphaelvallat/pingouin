import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.plotting import plot_skipped_corr


class TestPlotting(_TestPingouin):
    """Test plotting.py."""

    def test_ttest(self):
        """Test plot_skipped_corr()"""
        # Data for correlation
        np.random.seed(123)
        x, y = np.random.multivariate_normal([170, 70], [[20, 10], [10, 20]], 30).T
        # Introduce two outliers
        x[10], y[10] = 160, 100
        x[8], y[8] = 165, 90
        fig, ax1, ax2, ax3 = plot_skipped_corr(x, y)
        fig, ax1, ax2, ax3 = plot_skipped_corr(x, y, n_boot=10)
        fig, ax1, ax2, ax3 = plot_skipped_corr(x, y, seed=456)
