import matplotlib
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.plotting import plot_skipped_corr, _ppoints, qqplot


class TestPlotting(_TestPingouin):
    """Test plotting.py."""

    def test_plot_skipped_corr(self):
        """Test plot_skipped_corr()"""
        # Data for correlation
        np.random.seed(123)
        x, y = np.random.multivariate_normal([170, 70], [[20, 10],
                                                         [10, 20]], 30).T
        # Introduce two outliers
        x[10], y[10] = 160, 100
        x[8], y[8] = 165, 90
        plot_skipped_corr(x, y, xlabel='Height', ylabel='Weight')
        plot_skipped_corr(x, y, n_boot=10)
        fig = plot_skipped_corr(x, y, seed=456)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_ppoints(self):
        """Test _ppoints()"""
        R_test_5 = [0.1190476, 0.3095238, 0.5, 0.6904762, 0.8809524]
        R_test_15 = [0.03333333, 0.10000000, 0.16666667, 0.23333333,
                     0.30000000, 0.36666667, 0.43333333, 0.50000000,
                     0.56666667, 0.63333333, 0.70000000, 0.76666667,
                     0.83333333, 0.90000000, 0.96666667]

        np.testing.assert_array_almost_equal(_ppoints(5), R_test_5)
        np.testing.assert_array_almost_equal(_ppoints(15), R_test_15)

    def test_qqplot(self):
        """Test qqplot()"""
        np.random.seed(123)
        x = np.random.normal(size=50)
        x_exp = np.random.exponential(size=50)
        ax = qqplot(x, dist='norm')
        assert isinstance(ax, matplotlib.axes.Axes)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        qqplot(x_exp, dist='expon', ax=ax2)
        mean, std = 0, 0.8
        qqplot(x, dist=stats.norm, sparams=(mean, std), confidence=False)
