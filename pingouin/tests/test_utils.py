import pandas as pd
import numpy as np

from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.utils import *

# Dataset
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Time': ['Mon', 'Thur', 'Mon', 'Thur'],
                   'Values': [1.52, 5.8, 8.2, 3.4]})

class TestUtils(_TestPingouin):
    """Test utils.py."""

    def test_print_table(self):
        """Test function print_table."""
        df2, df3 = df.copy(), df.copy()
        df['F'] = 0
        print_table(df2)
        df3['A'] = 0
        print_table(df3, tablefmt='html', floatfmt='.3f')


    def test_export_table(self):
        """Test function export_table."""
        _export_table(df, fname=None)


    def test_reshape_data(self):
        """Test function reshape_data."""
        data = {'Ss': [1, 2, 3],
                '10am': [12, 6, 5],
                '2pm': [10, 6, 11],
                '6pm': [8, 5, 7]}
        df = pd.DataFrame(data, columns=['Ss', '10am', '2pm', '6pm'])
        reshaped = reshape_data(df, 'Ss', dv="Score", rm="Time")


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


    def test_check_data(self):
        """Test function _check_data."""
        _check_data(dv='Values', group='Group', data=df)
        x = [1, 5, 7, 8]
        y = np.array([4, 4, 7, 5])
        _check_data(x=x, y=y)


    def test_check_dataframe(self):
        """Test function _check_dataframe."""
        _check_dataframe(dv='Values', between='Group', effects='between',
                         data=df)
        _check_dataframe(dv='Values', within='Time', effects='within',
                         data=df)
        _check_dataframe(dv='Values', within='Time', between='Group',
                         effects='interaction', data=df)


    def test_extract_effects(self):
        """Test function _extract_effects."""
        _extract_effects(dv='Values', between='Group', effects='all',
                         data=df)
        _extract_effects(dv='Values', within='Time', between='Group',
                         effects='interaction', data=df)
        _extract_effects(dv='Values', within='Time',
                         effects='within', data=df)
