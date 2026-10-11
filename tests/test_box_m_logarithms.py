import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from pingouin import box_m


@pytest.mark.parametrize("scale", [1e-50, 1e50])
def test_box_m_covariance_scale(scale):
    iris = load_iris()
    data = pd.DataFrame(iris.data)
    data["group"] = iris.target
    reference = box_m(data, dvs=[0, 1, 2, 3], group="group")
    data[[0, 1, 2, 3]] *= scale
    result = box_m(data, dvs=[0, 1, 2, 3], group="group")
    np.testing.assert_allclose(result[["Chi2", "pval"]], reference[["Chi2", "pval"]], rtol=1e-10)


def test_box_m_likelihood_underflow():
    iris = load_iris()
    data = pd.DataFrame(np.tile(iris.data, (10, 1)))
    data["group"] = np.tile(iris.target, 10)
    result = box_m(data, dvs=[0, 1, 2, 3], group="group")
    # R biotools::boxM for this same repeated dataset.
    np.testing.assert_allclose(result.at["box", "Chi2"], 1487.8504406506481, rtol=1e-12)
    np.testing.assert_allclose(
        result.at["box", "pval"], 1.6093899533089083e-303, rtol=1e-10, atol=0
    )


@pytest.mark.parametrize("other_scale", [1, 10])
def test_box_m_negative_covariance_determinants(other_scale):
    # Pairwise missing-value removal can produce a non-positive covariance matrix.
    incomplete = np.array([[0, 0], [10, 10]] + [[5, np.nan]] * 4 + [[np.nan, 5]] * 4)
    complete = np.random.default_rng(12).normal(size=(10, 2)) * other_scale
    data = pd.DataFrame(np.vstack((incomplete, complete)), columns=["a", "b"])
    data["group"] = [0] * 10 + [1] * 10
    # Exercise negative determinants in either the pooled or a group covariance.
    result = box_m(data, dvs=["a", "b"], group="group")
    assert np.isnan(result.at["box", "Chi2"])
    assert np.isnan(result.at["box", "pval"])
