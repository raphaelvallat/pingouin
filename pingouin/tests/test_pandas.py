"""Test pingouin to Pandas methods.

Authors
- Nicolas Legrand <nicolaslegrand21@gmail.com>
- Raphael Vallat <raphaelvallat9@gmail.com>
"""
import numpy as np
import pingouin as pg
from unittest import TestCase

df = pg.read_dataset('mixed_anova')
data = pg.read_dataset('mediation')


class TestParametric(TestCase):
    """Test parametric.py."""

    def test_pandas(self):
        """Test pandas method.
        """
        # Test the ANOVA (Pandas)
        aov = df.anova(dv='Scores', between='Group', detailed=True)
        assert 'F' in aov.columns

        # Test the Welch ANOVA (Pandas)
        aov = df.welch_anova(dv='Scores', between='Group')
        assert 'F' in aov.columns

        # Test the repeated measures ANOVA (Pandas)
        aov = df.rm_anova(dv='Scores', within='Time', subject='Subject',
                          detailed=True)
        assert 'F' in aov.columns

        # FDR-corrected post hocs with Hedges'g effect size
        ttests = df.pairwise_ttests(dv='Scores', within='Time',
                                    subject='Subject', padjust='fdr_bh',
                                    effsize='hedges')
        assert 'p-corr' in ttests.columns

        # Test two-way mixed ANOVA
        aov = df.mixed_anova(dv='Scores', between='Group', within='Time',
                             subject='Subject', correction=False)
        assert 'F' in aov.columns

        # Test parwise correlations
        corrs = data.pairwise_corr(columns=['X', 'M', 'Y'], method='spearman')
        assert 'r2' in corrs.columns

        # Test partial correlation
        corrs = data.partial_corr(x='X', y='Y', covar='M', method='spearman')
        assert 'r2' in corrs.columns

        # Test partial correlation matrix (compare with the ppcor package)
        corrs = data.pcorr().round(3)
        np.testing.assert_array_equal(corrs.iloc[0, :].values,
                                      [1, 0.392, 0.06, -0.014, -0.149])

        # Test mediation analysis
        med = data.mediation_analysis(x='X', m='M', y='Y', seed=42, n_boot=500)
        assert 'coef' in med.columns
