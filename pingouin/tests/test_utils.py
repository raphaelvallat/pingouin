import pandas as pd
import numpy as np
import pytest

from unittest import TestCase
from pingouin import read_dataset
from pingouin.utils import (print_table, _perm_pval, _export_table,
                            remove_rm_na, _check_eftype, _check_dataframe,
                            remove_na, _flatten_list, _is_sklearn_installed,
                            _is_statsmodels_installed, _is_mpmath_installed)

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4],
                   'Subject': [1, 1, 2, 2]})


class TestUtils(TestCase):
    """Test utils.py."""

    def test_print_table(self):
        """Test function print_table."""
        df2, df3 = df.copy(), df.copy()
        df2['F'] = 0
        print_table(df2)
        df3['A'] = 0
        print_table(df3, tablefmt='html', floatfmt='.3f')

    def test_flatten_list(self):
        """Test function _flatten_list."""
        x = ['X1', ['M1', 'M2'], 'Y1', ['Y2']]
        fl = _flatten_list(x)
        np.testing.assert_array_equal(fl, ['X1', 'M1', 'M2', 'Y1', 'Y2'])
        x = ['Xaa', 'Xbb', 'Xcc']
        np.testing.assert_array_equal(_flatten_list(x), x)
        # With tuples
        xt = ['Xaa', ('Xbb', 'Xcc')]
        fl = _flatten_list(xt)
        assert fl == xt
        np.testing.assert_array_equal(_flatten_list(xt, include_tuple=True), x)

    def test_perm_pval(self):
        """Test function _perm_pval.
        """
        np.random.seed(123)
        bootstat = np.random.normal(size=1000)
        x = -2
        up = _perm_pval(bootstat, x, tail='greater')
        low = _perm_pval(bootstat, x, tail='less')
        two = _perm_pval(bootstat, x, tail='two-sided')
        assert up > low
        assert up + low == 1
        assert low < two < up
        x = 2.5
        up = _perm_pval(bootstat, x, tail='greater')
        low = _perm_pval(bootstat, x, tail='less')
        two = _perm_pval(bootstat, x, tail='two-sided')
        assert low > up
        assert up + low == 1
        assert up < two < low

    def test_export_table(self):
        """Test function export_table."""
        _export_table(df, fname='test_export')

    def test_remove_na(self):
        """Test function remove_na."""
        x = [6.4, 3.2, 4.5, np.nan]
        y = [3.5, 7.2, 8.4, 3.2]
        z = [2.3, np.nan, 5.2, 4.6]
        remove_na(x, y, paired=True)
        remove_na(x, y, paired=False)
        remove_na(y, x, paired=False)
        x_out, _ = remove_na(x, z, paired=True)
        assert np.allclose(x_out, [6.4, 4.5])
        # When y is None
        remove_na(x, None)
        remove_na(x, 4)
        # With 2D arrays
        x = np.array([[4, 2], [4, np.nan], [7, 6]])
        y = np.array([[6, np.nan], [3, 2], [2, 2]])
        x_nan, y_nan = remove_na(x, y, paired=False)
        assert np.allclose(x_nan, [[4., 2.], [7., 6.]])
        assert np.allclose(y_nan, [[3., 2.], [2., 2.]])
        x_nan, y_nan = remove_na(x, y, paired=True)
        assert np.allclose(x_nan, [[7., 6.]])
        assert np.allclose(y_nan, [[2., 2.]])
        x_nan, y_nan = remove_na(x, y, paired=False, axis='columns')
        assert np.allclose(x_nan, [[4.], [4.], [7.]])
        assert np.allclose(y_nan, [[6.], [3.], [2.]])
        # When y is None
        remove_na(x, None, paired=False)

    def test_remove_rm_na(self):
        """Test function remove_rm_na."""
        # With one within factor
        df = pd.DataFrame({'Time': ['A', 'A', 'B', 'B'],
                           'Values': [1.52, np.nan, 8.2, 3.4],
                           'Ss': [0, 1, 0, 1]})
        df = remove_rm_na(dv='Values', within='Time', subject='Ss', data=df)
        assert df['Ss'].nunique() == 1
        # With multiple factor
        df = read_dataset('rm_missing')
        stats = remove_rm_na(data=df, dv='BOLD', within=['Session', 'Time'],
                             subject='Subj')
        assert stats['BOLD'].isnull().sum() == 0
        assert stats['Memory'].isnull().sum() == 5
        # Multiple factors
        stats = remove_rm_na(data=df, within=['Time', 'Session'],
                             subject='Subj')
        assert stats['BOLD'].isnull().sum() == 0
        assert stats['Memory'].isnull().sum() == 0
        # Aggregation
        remove_rm_na(data=df, dv='BOLD', within='Session', subject='Subj')
        remove_rm_na(data=df, within='Session', subject='Subj',
                     aggregate='sum')
        remove_rm_na(data=df, within='Session', subject='Subj',
                     aggregate='first')
        df.loc['Subj', 1] = np.nan
        with pytest.raises(ValueError):
            remove_rm_na(data=df, within='Session', subject='Subj')

    def test_check_eftype(self):
        """Test function _check_eftype."""
        eftype = 'cohen'
        _check_eftype(eftype)
        eftype = 'fake'
        _check_eftype(eftype)

    def test_check_dataframe(self):
        """Test function _check_dataframe."""
        _check_dataframe(dv='Values', between='Group', effects='between',
                         data=df)
        _check_dataframe(dv='Values', within='Time', subject='Subject',
                         effects='within', data=df)
        _check_dataframe(dv='Values', within='Time', subject='Subject',
                         between='Group', effects='interaction', data=df)
        # Wrond and or missing arguments
        with pytest.raises(ValueError):
            _check_dataframe(dv='Group', between='Group', effects='between',
                             data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', effects='between')
        with pytest.raises(ValueError):
            _check_dataframe(between='Group', effects='between', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', effects='wrong',
                             data=df)
        with pytest.raises(ValueError):
            _check_dataframe(effects='within', dv='Values', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(effects='between', dv='Values', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(between='Group', effects='interaction',
                             dv='Values', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', within='Time',
                             effects='within', data=df)

    def _is_statsmodels_installed(self):
        """Test function _is_statsmodels_installed."""
        assert isinstance(_is_statsmodels_installed(), bool)

    def _is_sklearn_installed(self):
        """Test function _is_statsmodels_installed."""
        assert isinstance(_is_sklearn_installed(), bool)

    def _is_mpmath_installed(self):
        """Test function _is_mpmath_installed."""
        assert isinstance(_is_mpmath_installed(), bool)
