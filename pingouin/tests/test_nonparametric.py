import scipy
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin.nonparametric import (mad, madmedianrule, mwu, wilcoxon,
                                    kruskal, friedman, cochran, harrelldavis)

np.random.seed(1234)
x = np.random.normal(size=100)
y = np.random.normal(size=100)
z = np.random.normal(size=100)
w = np.random.normal(size=(5, 10))


class TestNonParametric(TestCase):
    """Test nonparametric.py."""

    def test_mad(self):
        """Test function mad."""
        from scipy.stats import median_absolute_deviation as mad_scp
        a = [1.2, 3, 4.5, 2.4, 5, 6.7, 0.4]
        # Compare to Matlab
        assert mad(a, normalize=False) == 1.8
        assert np.round(mad(a), 3) == np.round(1.8 * 1.4826, 3)
        # Axes handling -- Compare to SciPy
        assert np.allclose(mad_scp(w), mad(w))  # Axis = 0
        assert np.allclose(mad_scp(w, axis=1), mad(w, axis=1))
        assert np.allclose(mad_scp(w, axis=None), mad(w, axis=None))
        # Missing values
        # Note that in Scipy 1.3.0, mad(axis=0/1) does not work properly
        # if data contains NaN, even when passing (nan_policy='omit')
        wnan = w.copy()
        wnan[3, 2] = np.nan
        assert np.allclose(mad_scp(wnan, axis=None, nan_policy='omit'),
                           mad(wnan, axis=None))
        assert mad(wnan, axis=0).size == wnan.shape[1]
        assert mad(wnan, axis=1).size == wnan.shape[0]
        # Now we make sure that `w` and `wnan` returns almost the same results,
        # i.e. except for the row/column with missing values
        assert np.allclose(mad(w, axis=None), mad(wnan, axis=None), atol=1e-02)
        assert sum(mad(w, axis=0) == mad(wnan, axis=0)) == 9
        assert sum(mad(w, axis=1) == mad(wnan, axis=1)) == 4

    def test_madmedianrule(self):
        """Test function madmedianrule."""
        a = [1.2, 3, 4.5, 2.4, 5, 12.7, 0.4]
        assert np.alltrue(madmedianrule(a) == [False, False, False,
                                               False, False, True, False])

    def test_mwu(self):
        """Test function mwu"""
        mwu_scp = scipy.stats.mannwhitneyu(x, y, use_continuity=True,
                                           alternative='two-sided')
        mwu_pg = mwu(x, y, tail='two-sided')
        # Similar to R: wilcox.test(df$x, df$y, paired = FALSE, exact = FALSE)
        # Note that the RBC value are compared to JASP in test_pairwise.py
        assert mwu_scp[0] == mwu_pg.at['MWU', 'U-val']
        assert mwu_scp[1] == mwu_pg.at['MWU', 'p-val']
        # One-sided
        assert np.median(x) > np.median(y)  # Tail = greater, x > y
        assert (mwu(x, y, tail='one-sided').at['MWU', 'p-val'] ==
                mwu(x, y, tail='greater').at['MWU', 'p-val'])
        assert (mwu(x, y, tail='less').at['MWU', 'p-val'] ==
                scipy.stats.mannwhitneyu(x, y, use_continuity=True,
                                         alternative='less')[1])

    def test_wilcoxon(self):
        """Test function wilcoxon"""
        # R: wilcox.test(df$x, df$y, paired = TRUE, exact = FALSE)
        # The V value is slightly different between SciPy and R
        # The p-value, however, is almost identical
        wc_scp = scipy.stats.wilcoxon(x, y, correction=True)
        wc_pg = wilcoxon(x, y, tail='two-sided')
        wc_pg_1 = wilcoxon(x, y, tail='one-sided')
        assert wc_scp[0] == wc_pg.at['Wilcoxon', 'W-val']
        assert wc_scp[1] == wc_pg.at['Wilcoxon', 'p-val']
        # Compare to R canprot::CLES
        # Note that the RBC value are compared to JASP in test_pairwise.py
        assert wc_pg.at['Wilcoxon', 'CLES'] == 0.536
        assert (wc_pg.at['Wilcoxon', 'p-val'] / 2) == wc_pg_1.at['Wilcoxon',
                                                                 'p-val']

    def test_friedman(self):
        """Test function friedman"""
        df = pd.DataFrame({'DV': np.r_[x, y, z],
                           'Time': np.repeat(['A', 'B', 'C'], 100),
                           'Subject': np.tile(np.arange(100), 3)})
        friedman(data=df, dv='DV', subject='Subject', within='Time')
        summary = friedman(data=df, dv='DV', within='Time', subject='Subject',
                           export_filename='test_export.csv')
        # Compare with SciPy built-in function
        from scipy import stats
        Q, p = stats.friedmanchisquare(x, y, z)
        assert np.allclose(np.round(Q, 3), summary['Q']['Friedman'])
        assert np.allclose(p, summary['p-unc']['Friedman'])
        # Test with NaN
        df.loc[10, 'DV'] = np.nan
        friedman(data=df, dv='DV', subject='Subject', within='Time')

    def test_kruskal(self):
        """Test function kruskal"""
        x_nan = x.copy()
        x_nan[10] = np.nan
        df = pd.DataFrame({'DV': np.r_[x_nan, y, z],
                           'Group': np.repeat(['A', 'B', 'C'], 100)})
        kruskal(data=df, dv='DV', between='Group')
        summary = kruskal(data=df, dv='DV', between='Group',
                          export_filename='test_export.csv')
        # Compare with SciPy built-in function
        H, p = scipy.stats.kruskal(x_nan, y, z, nan_policy='omit')
        assert np.allclose(np.round(H, 3), summary['H']['Kruskal'])
        assert np.allclose(p, summary['p-unc']['Kruskal'])

    def test_cochran(self):
        """Test function cochran
        http://www.real-statistics.com/anova-repeated-measures/cochrans-q-test/
        """
        from pingouin import read_dataset
        df = read_dataset('cochran')
        st = cochran(dv='Energetic', within='Time', subject='Subject', data=df)
        assert st.loc['cochran', 'Q'] == 6.706
        assert np.allclose(st.loc['cochran', 'p-unc'], 0.034981)
        cochran(dv='Energetic', within='Time', subject='Subject', data=df,
                export_filename='test_export.csv')
        # With a NaN value
        df.loc[2, 'Energetic'] = np.nan
        cochran(dv='Energetic', within='Time', subject='Subject', data=df)

    def test_harrelldavis(self):
        """Test Harrel-Davis estimation of :math:`q^{th}` quantile.
        """
        a = [77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299,
             306, 376, 428, 515, 666, 1310, 2611]
        assert harrelldavis(a, quantile=0.5) == 271.72120054908913
        harrelldavis(x=x, quantile=np.arange(0.1, 1, 0.1))
        assert harrelldavis(a, [0.25, 0.5, 0.75])[1] == 271.72120054908913
