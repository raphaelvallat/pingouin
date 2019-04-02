import pandas as pd
import numpy as np
import pytest

from unittest import TestCase
from pingouin.utils import (print_table, _perm_pval, _export_table,
                            _remove_rm_na, _check_eftype, _check_dataframe,
                            _remove_na, _is_sklearn_installed,
                            _is_statsmodels_installed)

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

    def test_perm_pval(self):
        """Test function _perm_pval.
        """
        np.random.seed(123)
        bootstat = np.random.normal(size=1000)
        x = -2
        up = _perm_pval(bootstat, x, tail='upper')
        low = _perm_pval(bootstat, x, tail='lower')
        two = _perm_pval(bootstat, x, tail='two-sided')
        assert up > low
        assert up + low == 1
        assert low < two < up
        x = 2.5
        up = _perm_pval(bootstat, x, tail='upper')
        low = _perm_pval(bootstat, x, tail='lower')
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
        _remove_na(x, y, paired=True)
        _remove_na(x, y, paired=False)
        _remove_na(y, x, paired=False)
        x_out, _ = _remove_na(x, z, paired=True)
        assert np.allclose(x_out, [6.4, 4.5])

    def test_remove_rm_na(self):
        """Test function _remove_rm_na."""
        df = pd.DataFrame({'Time': ['A', 'A', 'B', 'B'],
                           'Values': [1.52, np.nan, 8.2, 3.4],
                           'Ss': [0, 1, 0, 1]})
        _remove_rm_na(dv='Values', within='Time', data=df)
        df = _remove_rm_na(dv='Values', within='Time', subject='Ss', data=df)
        assert df['Ss'].nunique() == 1

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
