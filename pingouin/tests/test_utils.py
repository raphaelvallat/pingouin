import pandas as pd
import numpy as np
import pytest

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.utils import (print_table, _export_table, _remove_rm_na,
                            _check_eftype, _check_dataframe, _remove_na,
                            reshape_data, is_sklearn_installed,
                            is_statsmodels_installed, mad, madmedianrule,
                            mahal)

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4],
                   'Subject': [1, 1, 2, 2]})


class TestUtils(_TestPingouin):
    """Test utils.py."""

    def test_print_table(self):
        """Test function print_table."""
        df2, df3 = df.copy(), df.copy()
        df2['F'] = 0
        print_table(df2)
        df3['A'] = 0
        print_table(df3, tablefmt='html', floatfmt='.3f')

    def test_export_table(self):
        """Test function export_table."""
        _export_table(df, fname='test_export')

    def test_reshape_data(self):
        """Test function reshape_data."""
        data = {'Ss': [1, 2, 3],
                '10am': [12, 6, 5],
                '2pm': [10, 6, 11],
                '6pm': [8, 5, 7]}
        df = pd.DataFrame(data, columns=['Ss', '10am', '2pm', '6pm'])
        reshape_data(df, 'Ss', dv="Score", rm="Time")

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
                           'Values': [1.52, np.nan, 8.2, 3.4]})
        df = _remove_rm_na(dv='Values', within='Time', data=df)

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
        # Missing arguments
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', effects='between')
        with pytest.raises(ValueError):
            _check_dataframe(between='Group', effects='between', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', effects='wrong',
                             data=df)
        with pytest.raises(ValueError):
            _check_dataframe(within=None, effects='within', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(between=None, effects='between', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(between='Group', effects='interaction', data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv='Values', between='Group', within='Time',
                             effects='within', data=df)

    def is_statsmodels_installed(self):
        """Test function is_statsmodels_installed."""
        assert isinstance(is_statsmodels_installed(), bool)

    def is_sklearn_installed(self):
        """Test function is_statsmodels_installed."""
        assert isinstance(is_sklearn_installed(), bool)

    def test_mad(self):
        """Test function mad."""
        a = [1.2, 3, 4.5, 2.4, 5, 6.7, 0.4]
        # Compare to Matlab
        assert mad(a, normalize=False) == 1.8
        assert np.round(mad(a), 3) == np.round(1.8 * 1.4826, 3)

    def test_madmedianrule(self):
        """Test function madmedianrule."""
        a = [1.2, 3, 4.5, 2.4, 5, 12.7, 0.4]
        assert np.alltrue(madmedianrule(a) == [False, False, False,
                                               False, False, True, False])

    def test_mahal(self):
        """Test function mahal."""
        # Compare to Matlab mahal function
        x = np.array([[-1.06, 1.60, 1.23, -0.23, -1.51],
                      [-1.15, 1.38, 1.23, -0.32, -1.16]]).T
        y = np.array([[1, 1, -1], [1, -1, 1]]).T
        np.allclose(np.round(mahal(y, x), 4), [0.9758, 131.4219, 134.05044])
