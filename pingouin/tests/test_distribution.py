import numpy as np
import pytest

from unittest import TestCase
from pingouin.distribution import (gzscore, normality, anderson, epsilon,
                                   homoscedasticity, sphericity)
from pingouin import read_dataset

# Generate random dataframe
df = read_dataset('mixed_anova.csv')
df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan

# Create random normal variables
np.random.seed(1234)
x = np.random.normal(scale=1., size=100)
y = np.random.normal(scale=0.8, size=100)
z = np.random.normal(scale=0.9, size=100)


class TestDistribution(TestCase):
    """Test distribution.py."""

    def test_gzscore(self):
        """Test function gzscore."""
        raw = np.random.lognormal(size=100)
        gzscore(raw)

    def test_normality(self):
        """Test function test_normality."""
        normality(x, alpha=.05)
        normality(x, y, alpha=.05)

    def test_homoscedasticity(self):
        """Test function test_homoscedasticity."""
        homoscedasticity(x, y, alpha=.05)
        with pytest.raises(ValueError):
            homoscedasticity(x)

    def test_epsilon(self):
        """Test function epsilon."""
        df_pivot = df.pivot(index='Subject', columns='Time',
                            values='Scores').reset_index(drop=True)
        eps_gg = epsilon(df_pivot)
        eps_hf = epsilon(df_pivot, correction='hf')
        eps_lb = epsilon(df_pivot, correction='lb')
        # Compare with ezANOVA
        assert np.allclose([eps_gg, eps_hf, eps_lb], [0.9987509, 1, 0.5])

    def test_sphericity(self):
        """Test function test_sphericity."""
        df_pivot = df.pivot(index='Subject', columns='Time',
                            values='Scores').reset_index(drop=True)
        _, W, _, _, p = sphericity(df_pivot, method='mauchly')
        # Compare with ezANOVA
        assert np.round(W, 3) == 0.999
        assert np.round(p, 3) == 0.964
        # JNS
        sphericity(df_pivot, method='jns')

    def test_anderson(self):
        """Test function test_anderson."""
        anderson(x)
        anderson(x, y)
        anderson(x, dist='expon')
