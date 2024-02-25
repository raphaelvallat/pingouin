import numpy as np
import pandas as pd
from sklearn import datasets
from unittest import TestCase
from pingouin import read_dataset
from pingouin.multivariate import multivariate_normality, multivariate_ttest, box_m

data = read_dataset("multivariate")
dvs = ["Fever", "Pressure", "Aches"]
X = data[data["Condition"] == "Drug"][dvs]
Y = data[data["Condition"] == "Placebo"][dvs]
# With missing values
X_na = X.copy()
X_na.iloc[4, 2] = np.nan
# Rank deficient
X_rd = X.copy()
X_rd["Bad"] = 1.08 * X_rd["Fever"] - 0.5 * X_rd["Pressure"]


class TestMultivariate(TestCase):
    """Test multivariate.py.
    Tested against the R package MVN.
    """

    def test_multivariate_normality(self):
        """Test function multivariate_normality."""
        np.random.seed(123)
        # With 2 variables
        mean, cov, n = [4, 6], [[1, 0.5], [0.5, 1]], 30
        Z = np.random.multivariate_normal(mean, cov, n)
        hz, pval, normal = multivariate_normality(Z, alpha=0.05)
        assert round(hz, 7) == 0.3243965
        assert round(pval, 7) == 0.7523511
        assert normal is True
        # With 3 variables
        hz, pval, normal = multivariate_normality(data[dvs], alpha=0.01)
        assert round(hz, 7) == 0.5400861
        assert round(pval, 7) == 0.7173687
        assert normal is True
        # With missing values
        hz, pval, normal = multivariate_normality(X_na, alpha=0.05)
        assert round(hz, 7) == 0.7288211
        assert round(pval, 7) == 0.1120792
        assert normal is True
        # Rank deficient
        # Return a LAPACK singular error in R so we cannot compare
        hz, pval, normal = multivariate_normality(X_rd)

    def test_multivariate_ttest(self):
        """Test function multivariate_ttest.
        Tested against the R package Hotelling and real-statistics.com.
        """
        np.random.seed(123)
        # With 2 variables
        mean, cov, n = [4, 6], [[1, 0.5], [0.5, 1]], 30
        Z = np.random.multivariate_normal(mean, cov, n)
        # One-sample test
        multivariate_ttest(Z, Y=None, paired=False)
        multivariate_ttest(Z, Y=[4, 5], paired=False)
        # With 3 variables
        # Two-sample independent
        stats = multivariate_ttest(X, Y)
        assert round(stats.at["hotelling", "F"], 3) == 1.327
        assert stats.at["hotelling", "df1"] == 3
        assert stats.at["hotelling", "df2"] == 32
        assert round(stats.loc["hotelling", "pval"], 3) == 0.283
        # Paired test with NaN values
        stats = multivariate_ttest(X_na, Y, paired=True)
        assert stats.at["hotelling", "df1"] == 3
        assert stats.at["hotelling", "df2"] == X.shape[0] - 1 - X.shape[1]

    def test_box_m(self):
        """Test function box_m.

        Tested against the R package biotools (iris dataset).
        """
        # Test 1: Iris dataset
        iris = datasets.load_iris()
        df = pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
        )
        stats = box_m(
            df,
            dvs=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            group="target",
        )
        assert round(stats.at["box", "Chi2"], 3) == 140.943
        assert stats.at["box", "df"] == 20
        assert np.isclose(stats.at["box", "pval"], 3.352034e-20)

        # Test 2: Multivariate normal dist with balanced sample size
        # In R:
        # >>> library(biotools)
        # >>> data < - read.csv("data.csv")
        # >>> boxM(data[, c('A', 'B', 'C')], grouping=data[, c('group')])
        from scipy.stats import multivariate_normal as mvn

        data = pd.DataFrame(mvn.rvs(size=(100, 3), random_state=42), columns=["A", "B", "C"])
        data["group"] = [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25
        stats = box_m(data, dvs=["A", "B", "C"], group="group")
        assert round(stats.at["box", "Chi2"], 5) == 11.63419
        assert stats.at["box", "df"] == 18
        assert round(stats.at["box", "pval"], 7) == 0.8655372

        # Test 3: Multivariate normal dist with unbalanced sample size
        data = pd.DataFrame(mvn.rvs(size=(30, 2), random_state=42), columns=["A", "B"])
        data["group"] = [1] * 20 + [2] * 10
        stats = box_m(data, dvs=["A", "B"], group="group")
        assert round(stats.at["box", "Chi2"], 5) == 0.70671
        assert stats.at["box", "df"] == 3
        assert round(stats.at["box", "pval"], 7) == 0.8716249

        # Test 4: with missing values
        # In R, biotools return NA for Chi2 and p-value
        data.loc[[1, 3], ["A", "B"]] = np.nan
        stats = box_m(data, dvs=["A", "B"], group="group")
        assert stats.at["box", "df"] == 3

    def test_multivariate_normality_numerical_stability(self):
        """Test numerical stability of multivariate_normality."""
        np.random.seed(123)
        # Test that large datasets do not produce nan
        n, p = 1000, 100
        Z = np.random.normal(size=(n, p))
        hz, pval, normal = multivariate_normality(Z, alpha=0.05)
        assert np.isfinite(pval)
