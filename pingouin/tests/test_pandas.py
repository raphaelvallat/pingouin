"""Test pingouin to Pandas methods.

Authors
- Nicolas Legrand <nicolaslegrand21@gmail.com>
- Raphael Vallat <raphaelvallat9@gmail.com>
"""
import numpy as np
import pingouin as pg
from unittest import TestCase

df = pg.read_dataset('mixed_anova')
df_aov3 = pg.read_dataset('anova3_unbalanced')
df_anc = pg.read_dataset('ancova')
df_corr = pg.read_dataset('pairwise_corr').iloc[:, 1:]
data = pg.read_dataset('mediation')


class TestParametric(TestCase):
    """Test pandas methods"""

    def test_pandas(self):
        """Test pandas method.
        """
        # Test the ANOVA (Pandas)
        aov = df.anova(dv='Scores', between='Group', detailed=True)
        assert aov.equals(pg.anova(dv='Scores', between='Group', detailed=True,
                                   data=df))
        aov3_ss1 = df_aov3.anova(dv='Cholesterol', between=['Sex', 'Drug'],
                                 ss_type=1)
        aov3_ss2 = df_aov3.anova(dv='Cholesterol', between=['Sex', 'Drug'],
                                 ss_type=2)
        aov3_ss2_pg = pg.anova(dv='Cholesterol', between=['Sex', 'Drug'],
                               data=df_aov3, ss_type=2)
        assert not aov3_ss1.equals(aov3_ss2)
        assert aov3_ss2.round(3).equals(aov3_ss2_pg.round(3))

        # Test the Welch ANOVA (Pandas)
        aov = df.welch_anova(dv='Scores', between='Group')
        assert aov.equals(pg.welch_anova(dv='Scores', between='Group',
                                         data=df))

        # Test the ANCOVA
        aov = df_anc.ancova(dv='Scores', covar='Income',
                            between='Method').round(3)
        assert aov.equals(pg.ancova(data=df_anc, dv='Scores', covar='Income',
                          between='Method').round(3))

        # Test the repeated measures ANOVA (Pandas)
        aov = df.rm_anova(dv='Scores', within='Time', subject='Subject',
                          detailed=True)
        assert aov.equals(pg.rm_anova(dv='Scores', within='Time',
                                      subject='Subject',
                                      detailed=True, data=df))

        # FDR-corrected post hocs with Hedges'g effect size
        ttests = df.pairwise_ttests(dv='Scores', within='Time',
                                    subject='Subject', padjust='fdr_bh',
                                    effsize='hedges')
        assert ttests.equals(pg.pairwise_ttests(dv='Scores', within='Time',
                                                subject='Subject',
                                                padjust='fdr_bh',
                                                effsize='hedges', data=df))

        # Pairwise Tukey
        tukey = df.pairwise_tukey(dv='Scores', between='Group')
        assert tukey.equals(pg.pairwise_tukey(data=df, dv='Scores',
                                              between='Group'))

        # Test two-way mixed ANOVA
        aov = df.mixed_anova(dv='Scores', between='Group', within='Time',
                             subject='Subject', correction=False)
        assert aov.equals(pg.mixed_anova(dv='Scores', between='Group',
                                         within='Time',
                                         subject='Subject', correction=False,
                                         data=df))

        # Test parwise correlations
        corrs = data.pairwise_corr(columns=['X', 'M', 'Y'], method='spearman')
        corrs2 = pg.pairwise_corr(data=data, columns=['X', 'M', 'Y'],
                                  method='spearman')
        assert corrs['r'].equals(corrs2['r'])

        # Test partial correlation
        corrs = data.partial_corr(x='X', y='Y', covar='M', method='spearman')
        corrs2 = pg.partial_corr(x='X', y='Y', covar='M', method='spearman',
                                 data=data)
        assert corrs['r'].equals(corrs2['r'])

        # Test partial correlation matrix (compare with the ppcor package)
        corrs = data.pcorr().round(3)
        np.testing.assert_array_equal(corrs.iloc[0, :].values,
                                      [1, 0.392, 0.06, -0.014, -0.149])
        # Now compare against Pingouin's own partial_corr function
        corrs = data[['X', 'Y', 'M']].pcorr()
        corrs2 = data.partial_corr(x='X', y='Y', covar='M')
        assert round(corrs.loc['X', 'Y'], 3) == corrs2.loc['pearson', 'r']

        # Test rcorr (correlation matrix with p-values)
        # We compare against Pingouin pairwise_corr function
        corrs = df_corr.rcorr(padjust='holm')
        corrs2 = df_corr.pairwise_corr(padjust='holm')
        assert corrs.loc['Neuroticism', 'Agreeableness'] == '*'
        assert (corrs.loc['Agreeableness', 'Neuroticism'] ==
                str(corrs2.loc[2, 'r']))
        corrs = df_corr.rcorr(padjust='holm', stars=False, decimals=4)
        assert (corrs.loc['Neuroticism', 'Agreeableness'] ==
                str(corrs2.loc[2, 'p-corr'].round(4)))
        corrs = df_corr.rcorr(upper='n')
        corrs2 = df_corr.pairwise_corr()
        assert corrs.loc['Extraversion', 'Openness'] == corrs2.loc[4, 'n']
        assert corrs.loc['Openness', 'Extraversion'] == str(corrs2.loc[4, 'r'])
        # Method = spearman does not work with Python 3.5 on Travis?
        # Instead it seems to return the Pearson correlation!
        df_corr.rcorr(method='spearman')
        df_corr.rcorr()

        # Test mediation analysis
        med = data.mediation_analysis(x='X', m='M', y='Y', seed=42, n_boot=500)
        np.testing.assert_array_equal(med.loc[:, 'coef'].values,
                                      [0.5610, 0.6542, 0.3961, 0.0396, 0.3565])
