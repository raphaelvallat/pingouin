import pandas as pd
import numpy as np
import pytest

import pingouin

from unittest import TestCase
from pingouin.utils import (
    print_table,
    _postprocess_dataframe,
    _get_round_setting_for,
    _perm_pval,
    _check_eftype,
    _check_dataframe,
    remove_na,
    _flatten_list,
    _is_sklearn_installed,
    _is_statsmodels_installed,
    _is_mpmath_installed,
)

# Dataset
df = pd.DataFrame(
    {
        "Group": ["A", "A", "B", "B"],
        "Time": ["Mon", "Thur", "Mon", "Thur"],
        "Values": [1.52, 5.8, 8.2, 3.4],
        "Subject": [1, 1, 2, 2],
    }
)


class TestUtils(TestCase):
    """Test utils.py."""

    def test_print_table(self):
        """Test function print_table."""
        df2, df3 = df.copy(), df.copy()
        df2["F"] = 0
        print_table(df2)
        df3["A"] = 0
        print_table(df3, tablefmt="html", floatfmt=".3f")

    def test__postprocess_dataframe(self):
        """Test function _postprocess_dataframe."""
        df2 = df.copy()
        # add some more values and give a stringy index
        df2.Values = [1.54321, 5.87654, 8.23456, 3.45678]
        df2 = df2.assign(Values2=[1.54321, 5.87654, 8.23456, 3.45678])
        df2.index = ["row" + str(x) for x in df.index]

        # set rounding options (keeping original options dict to restore after)
        old_opts = pingouin.options.copy()
        pingouin.options.clear()
        pingouin.options["round"] = 4
        pingouin.options["round.cell.[row0]x[Values]"] = None
        pingouin.options["round.column.Values"] = 3
        pingouin.options["round.row.row1"] = 2
        pingouin.options["round.cell.[row3]x[Values2]"] = 0

        df_expected = df2.copy()
        df_expected.Values = [1.54321, 5.877, 8.235, 3.457]
        df_expected.Values2 = [1.5432, 5.88, 8.2346, 3.0]

        df2 = _postprocess_dataframe(df2)
        pd.testing.assert_frame_equal(df2, df_expected)

        # restore old options
        pingouin.options.update(old_opts)

    def test_get_round_setting_for(self):
        """Test function _get_round_setting_for."""
        # set rounding options (keeping original options dict to restore after)
        old_opts = pingouin.options.copy()
        pingouin.options.clear()
        pingouin.options["round"] = 4
        pingouin.options["round.cell.[row0]x[Values]"] = None
        pingouin.options["round.column.Values"] = 3
        pingouin.options["round.row.row1"] = 2
        pingouin.options["round.cell.[row3]x[Values2]"] = 0

        assert _get_round_setting_for("row0", "Values") is None
        assert _get_round_setting_for("row1", "Values") == 3
        assert _get_round_setting_for("row1", "Values2") == 2
        assert _get_round_setting_for("row3", "Values2") == 0
        assert _get_round_setting_for("row2", "Values2") == 4  # default

        # restore old options
        pingouin.options.update(old_opts)

    def test_flatten_list(self):
        """Test function _flatten_list."""
        x = ["X1", ["M1", "M2"], "Y1", ["Y2"]]
        fl = _flatten_list(x)
        np.testing.assert_array_equal(fl, ["X1", "M1", "M2", "Y1", "Y2"])
        x = ["Xaa", "Xbb", "Xcc"]
        np.testing.assert_array_equal(_flatten_list(x), x)
        # With tuples
        xt = ["Xaa", ("Xbb", "Xcc")]
        fl = _flatten_list(xt)
        assert fl == xt
        np.testing.assert_array_equal(_flatten_list(xt, include_tuple=True), x)
        assert _flatten_list(1) == 1  # x is not iterable
        assert _flatten_list([(1), (2)]) == [1, 2]  # (1) is an int and not tup

    def test_perm_pval(self):
        """Test function _perm_pval."""
        np.random.seed(123)
        bootstat = np.random.normal(size=1000)
        x = -2
        up = _perm_pval(bootstat, x, alternative="greater")
        low = _perm_pval(bootstat, x, alternative="less")
        two = _perm_pval(bootstat, x, alternative="two-sided")
        assert up > low
        assert up + low == 1
        assert low < two < up
        x = 2.5
        up = _perm_pval(bootstat, x, alternative="greater")
        low = _perm_pval(bootstat, x, alternative="less")
        two = _perm_pval(bootstat, x, alternative="two-sided")
        assert low > up
        assert up + low == 1
        assert up < two < low

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
        assert np.allclose(x_nan, [[4.0, 2.0], [7.0, 6.0]])
        assert np.allclose(y_nan, [[3.0, 2.0], [2.0, 2.0]])
        x_nan, y_nan = remove_na(x, y, paired=True)
        assert np.allclose(x_nan, [[7.0, 6.0]])
        assert np.allclose(y_nan, [[2.0, 2.0]])
        x_nan, y_nan = remove_na(x, y, paired=False, axis="columns")
        assert np.allclose(x_nan, [[4.0], [4.0], [7.0]])
        assert np.allclose(y_nan, [[6.0], [3.0], [2.0]])
        # When y is None
        remove_na(x, None, paired=False)
        # When y is an empty list
        # See https://github.com/raphaelvallat/pingouin/issues/222
        with pytest.raises(AssertionError):
            remove_na(x, y=[])

    def test_check_eftype(self):
        """Test function _check_eftype."""
        eftype = "cohen"
        _check_eftype(eftype)
        eftype = "fake"
        _check_eftype(eftype)

    def test_check_dataframe(self):
        """Test function _check_dataframe."""
        _check_dataframe(dv="Values", between="Group", effects="between", data=df)
        _check_dataframe(dv="Values", within="Time", subject="Subject", effects="within", data=df)
        _check_dataframe(
            dv="Values",
            within="Time",
            subject="Subject",
            between="Group",
            effects="interaction",
            data=df,
        )
        # Wrond and or missing arguments
        with pytest.raises(ValueError):
            _check_dataframe(dv="Group", between="Group", effects="between", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv="Values", between="Group", effects="between")
        with pytest.raises(ValueError):
            _check_dataframe(between="Group", effects="between", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv="Values", between="Group", effects="wrong", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(effects="within", dv="Values", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(effects="between", dv="Values", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(between="Group", effects="interaction", dv="Values", data=df)
        with pytest.raises(ValueError):
            _check_dataframe(dv="Values", between="Group", within="Time", effects="within", data=df)

    def _is_statsmodels_installed(self):
        """Test function _is_statsmodels_installed."""
        assert isinstance(_is_statsmodels_installed(), bool)

    def _is_sklearn_installed(self):
        """Test function _is_statsmodels_installed."""
        assert isinstance(_is_sklearn_installed(), bool)

    def _is_mpmath_installed(self):
        """Test function _is_mpmath_installed."""
        assert isinstance(_is_mpmath_installed(), bool)
