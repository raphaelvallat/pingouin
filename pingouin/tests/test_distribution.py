import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.distribution import (gzscore, normality, anderson, epsilon,
                                   homoscedasticity, sphericity)
from pingouin import read_dataset

# Generate random dataframe
df = read_dataset('mixed_anova.csv')
df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan
df_pivot = df.pivot(index='Subject', columns='Time',
                    values='Scores').reset_index(drop=True)

# Create random normal variables
np.random.seed(1234)
x = np.random.normal(scale=1., size=100)
y = np.random.normal(scale=0.8, size=100)
z = np.random.normal(scale=0.9, size=100)

# Two-way repeated measures (epsilon and sphericity)
data = read_dataset('rm_anova2')
idx, dv = 'Subject', 'Performance'
within = ['Time', 'Metric']
pa = data.pivot_table(index=idx, columns=within[0], values=dv)
pb = data.pivot_table(index=idx, columns=within[1], values=dv)
pab = data.pivot_table(index=idx, columns=within, values=dv)
# pab_single is a multilevel dataframe with shape columns.levshape = (1, 3)
pab_single = pab.xs('Pre', level=0, drop_level=False, axis=1)
# Create a 3-level mutlilevel columns (to test ValueError)
pab_3fac = pab_single.copy()
old_idx = pab_single.columns.to_frame()
old_idx.insert(0, 'new_level_name', [0, 0, 0])
pab_3fac.columns = pd.MultiIndex.from_frame(old_idx)

# Two-way repeated measures (3, 4)
np.random.seed(123)
dv = np.random.normal(scale=3, size=600)
w1 = np.repeat(['P1', 'P2', 'P3'], 200)
w2 = np.tile(np.repeat(['A', 'B', 'C', 'D'], 50), 3)
subj = np.tile(np.tile(np.arange(50), 4), 3)
df3 = pd.DataFrame({'dv': dv, 'within1': w1, 'within2': w2, 'subj': subj})
idx, dv = 'subj', 'dv'
within = ['within1', 'within2']
pa1 = df3.pivot_table(index=idx, columns=within[0], values=dv)
pb1 = df3.pivot_table(index=idx, columns=within[1], values=dv)
pab1 = df3.pivot_table(index=idx, columns=within, values=dv)


class TestDistribution(TestCase):
    """Test distribution.py."""

    def test_gzscore(self):
        """Test function gzscore."""
        raw = np.random.lognormal(size=100)
        gzscore(raw)

    def test_normality(self):
        """Test function test_normality."""
        # List / 1D array
        normality(x, alpha=.05)
        normality(x.tolist(), method='normaltest', alpha=.05)
        # Pandas DataFrame
        df_nan_piv = df_nan.pivot(index='Subject', columns='Time',
                                  values='Scores')
        normality(df_nan_piv)  # Wide-format dataframe
        normality(df_nan_piv['August'])  # pandas Series
        # The line below is disabled because test fails on python 3.5
        # assert stats_piv.equals(normality(df_nan, group='Time', dv='Scores'))
        normality(df_nan, group='Group', dv='Scores', method='normaltest')

    def test_homoscedasticity(self):
        """Test function test_homoscedasticity."""
        hl = homoscedasticity(data=[x, y], alpha=.05)
        homoscedasticity(data=[x, y], method='bartlett', alpha=.05)
        hd = homoscedasticity(data={'x': x, 'y': y}, alpha=.05)
        assert hl.equals(hd)
        # Wide-format DataFrame
        homoscedasticity(df_pivot)
        # Long-format
        homoscedasticity(df, dv='Scores', group='Time')

    def test_epsilon(self):
        """Test function epsilon."""
        df_pivot = df.pivot(index='Subject', columns='Time',
                            values='Scores').reset_index(drop=True)
        eps_gg = epsilon(df_pivot)
        eps_hf = epsilon(df_pivot, correction='hf')
        eps_lb = epsilon(df_pivot, correction='lb')
        # In long-format
        eps_gg_rm = epsilon(df, subject='Subject', within='Time', dv='Scores')
        eps_hf_rm = epsilon(df, subject='Subject', within='Time', dv='Scores',
                            correction='hf')
        assert np.isclose(eps_gg_rm, eps_gg)
        assert np.isclose(eps_hf_rm, eps_hf)
        # Compare with ezANOVA
        assert np.allclose([eps_gg, eps_hf, eps_lb], [0.9987509, 1, 0.5])

        # Time has only two values so epsilon is one.
        assert epsilon(pa, correction='lb') == epsilon(pa, correction='gg')
        assert epsilon(pa, correction='gg') == epsilon(pa, correction='hf')
        # Lower bound <= Greenhouse-Geisser <= Huynh-Feldt
        assert epsilon(pb, correction='lb') <= epsilon(pb, correction='gg')
        assert epsilon(pb, correction='gg') <= epsilon(pb, correction='hf')
        assert epsilon(pab, correction='lb') <= epsilon(pab, correction='gg')
        assert epsilon(pab, correction='gg') <= epsilon(pab, correction='hf')
        # Lower bound == 0.5 for pb and pab
        assert epsilon(pb, correction='lb') == epsilon(pab, correction='lb')
        assert np.allclose(epsilon(pb), 0.9691030)  # ez
        assert np.allclose(epsilon(pb, correction='hf'), 1.0)  # ez
        # Epsilon for the interaction (shape = (2, N))
        assert np.allclose(epsilon(pab), 0.7271664)
        assert np.allclose(epsilon(pab, correction='hf'), 0.831161)
        assert epsilon(pab) == epsilon(pab.swaplevel(axis=1))
        assert epsilon(pab_single) == epsilon(pab_single.swaplevel(axis=1))
        eps_gg_rm = epsilon(data, subject='Subject', dv='Performance',
                            within=['Time', 'Metric'])
        assert eps_gg_rm == epsilon(pab)
        # Now with a (3, 4) two-way design
        assert np.allclose(epsilon(pa1), 0.9963275)
        assert np.allclose(epsilon(pa1, correction='hf'), 1.)
        assert np.allclose(epsilon(pb1), 0.9716288)
        assert np.allclose(epsilon(pb1, correction='hf'), 1.)
        assert 0.8 < epsilon(pab1) < .90  # Pingouin = .822, ez = .856
        eps_gg_rm = epsilon(df3, subject='subj', dv='dv',
                            within=['within1', 'within2'])
        assert eps_gg_rm == epsilon(pab1)
        # With missing values
        eps_gg_rm = epsilon(df_nan, subject='Subject', within='Time',
                            dv='Scores')
        # 3 repeated measures factor
        with pytest.raises(ValueError):
            epsilon(pab_3fac)

    def test_sphericity(self):
        """Test function test_sphericity.
        Compare with ezANOVA."""
        _, W, _, _, p = sphericity(df_pivot, method='mauchly')
        assert W == 0.999
        assert np.round(p, 3) == 0.964
        _, W, _, _, p = sphericity(df, dv='Scores', subject='Subject',
                                   within='Time')  # Long-format
        assert W == 0.999
        assert np.round(p, 3) == 0.964
        assert sphericity(pa)[0]  # Only two levels so sphericity = True
        spher = sphericity(pb)
        assert spher[0]
        assert spher[1] == 0.968  # W
        assert spher[3] == 2  # dof
        assert np.isclose(spher[4], 0.8784418)  # P-value
        # JNS
        sphericity(df_pivot, method='jns')
        sphericity(df, dv='Scores', subject='Subject', within=['Time'],
                   method='jns')
        # Two-way design of shape (2, N)
        spher = sphericity(pab)
        assert spher[1] == 0.625
        assert spher[3] == 2
        assert np.isclose(spher[4], 0.1523917)
        assert sphericity(pab)[1] == sphericity(pab.swaplevel(axis=1))[1]
        spher_long = sphericity(data, subject='Subject', dv='Performance',
                                within=['Time', 'Metric'])
        assert spher_long[1] == 0.625
        assert np.isclose(spher[4], spher_long[4])
        sphericity(pab_single)  # For coverage
        # Now with a (3, 4) two-way design
        # First, main effect
        spher = sphericity(pb1)
        assert spher[0]
        assert spher[1] == 0.958  # W
        assert round(spher[4], 4) == 0.8436  # P-value
        spher2 = sphericity(df3, subject='subj', dv='dv', within=['within2'])
        assert spher[1] == spher2[1]
        assert spher[4] == spher2[4]
        # And then interaction (ValueError)
        with pytest.raises(ValueError):
            sphericity(pab1)
        # Same with long-format
        with pytest.raises(ValueError):
            sphericity(df3, subject='subj', dv='dv',
                       within=['within1', 'within2'])
        # 3 repeated measures factor
        with pytest.raises(ValueError):
            sphericity(pab_3fac)
        # With missing values
        sphericity(df_nan, subject='Subject', within='Time', dv='Scores')

    def test_anderson(self):
        """Test function test_anderson."""
        anderson(x)
        anderson(x, y)
        anderson(x, dist='expon')
