import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from scipy.stats import chi2_contingency
import pingouin as pg


class TestContingency(TestCase):
    """Test contingency.py."""

    def test_chi2(self):
        """Test function chi2."""
        # Setup
        np.random.seed(42)
        mean, cov = [0.5, 0.5], [(1, .6), (.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        data = pd.DataFrame({'x': x, 'y': y})
        mask_class_1 = data > 0.5
        data[mask_class_1] = 1
        data[~mask_class_1] = 0

        # Comparing results with SciPy
        _, _, stats = pg.chi2(data, x='x', y='y')
        contingency_table = pd.crosstab(data['x'], data['y'])
        for i in stats.index:
            lambda_ = stats.at[i, 'lambda']
            dof = stats.at[i, 'dof']
            chi2 = stats.at[i, 'chi2']
            p = round(stats.at[i, 'p'], 6)
            sp_chi2, sp_p, sp_dof, _ = chi2_contingency(contingency_table,
                                                        lambda_=lambda_)
            assert (chi2, p, dof) == (round(sp_chi2, 3), round(sp_p, 6),
                                      sp_dof)

        # Testing resilience to NaN
        mask_nan = np.random.random(data.shape) > 0.8  # ~20% NaN values
        data[mask_nan] = np.nan
        pg.chi2(data, x='x', y='y')

        # Testing validations
        def expect_assertion_error(*params):
            with pytest.raises(AssertionError):
                pg.chi2(*params)
        expect_assertion_error(1, 'x', 'y')  # Not a pd.DataFrame
        expect_assertion_error(data, x, 'y')  # Not a string
        expect_assertion_error(data, 'x', y)  # Not a string
        expect_assertion_error(data, 'x', 'z')  # Not a column of data

        # Testing "no data" ValueError
        data['x'] = np.nan
        with pytest.raises(ValueError):
            pg.chi2(data, x='x', y='y')

        # Testing degenerated case (observed == expected)
        data['x'] = 1
        data['y'] = 1
        expected, observed, stats = pg.chi2(data, 'x', 'y')
        assert expected.iloc[0, 0] == observed.iloc[0, 0]
        assert stats.at[0, 'dof'] == 0
        for i in stats.index:
            chi2 = stats.at[i, 'chi2']
            p = stats.at[i, 'p']
            assert (chi2, p) == (0.0, 1.0)

        # Testing warning on low count
        data.iloc[0, 0] = 0
        with pytest.warns(UserWarning):
            pg.chi2(data, 'x', 'y')

    def test_chi2_mcnemar(self):
        """Test function chi2_mcnemar."""
        # Setup
        np.random.seed(42)
        mean, cov = [0.5, 0.5], [(1, .6), (.6, 1)]
        x, y = np.random.multivariate_normal(mean, cov, 30).T
        data = pd.DataFrame({'x': x, 'y': y})
        mask_class_1 = data > 0.5
        data[mask_class_1] = 1
        data[~mask_class_1] = 0

        # Testing validations
        def expect_assertion_error(*params):
            with pytest.raises(AssertionError):
                pg.chi2_mcnemar(*params)
        expect_assertion_error(1, 'x', 'y')  # Not a pd.DataFrame
        expect_assertion_error(data, x, 'y')  # Not a string
        expect_assertion_error(data, 'x', y)  # Not a string
        expect_assertion_error(data, 'x', 'z')  # Not a column of data

        # Testing NaN incompatibility
        data.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            pg.chi2_mcnemar(data, 'x', 'y')

        # Testing invalid dichotomous value
        data.iloc[0, 0] = 3
        with pytest.raises(ValueError):
            pg.chi2_mcnemar(data, 'x', 'y')
