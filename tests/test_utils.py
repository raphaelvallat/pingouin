import sys
import types

import numpy as np
import pandas as pd
import pytest

import pingouin
from pingouin.utils import (
    _check_dataframe,
    _check_eftype,
    _flatten_list,
    _get_round_setting_for,
    _is_installed,
    _perm_pval,
    _postprocess_dataframe,
    print_table,
    remove_na,
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


def test_print_table(capsys):
    """Test function print_table."""
    # ANOVA table
    print_table(df.assign(F=0.123456))
    out = capsys.readouterr().out
    assert "ANOVA SUMMARY" in out
    assert "POST HOC TESTS" not in out
    assert "0.123" in out and "0.1235" not in out  # floatfmt=".3f"
    for col in ["Group", "Time", "Values", "Subject", "F"]:
        assert col in out
    # Post-hoc table, HTML format
    print_table(df.assign(A=0), tablefmt="html", floatfmt=".1f")
    out = capsys.readouterr().out
    assert "POST HOC TESTS" in out
    assert "ANOVA SUMMARY" not in out
    assert "<table>" in out and "<td" in out
    assert "1.5" in out and "1.52" not in out
    # Other tables have no title, and the index is not printed
    print_table(df.set_index("Group"), tablefmt="plain")
    out = capsys.readouterr().out
    assert "SUMMARY" not in out and "POST HOC" not in out
    assert "Group" not in out


def test__postprocess_dataframe():
    """Test function _postprocess_dataframe."""
    df2 = df.copy()
    # add some more values and give a stringy index
    df2.Values = [1.54321, 5.87654, 8.23456, 3.45678]
    df2 = df2.assign(Values2=[1.54321, 5.87654, 8.23456, 3.45678])
    df2.index = ["row" + str(x) for x in df.index]

    # set rounding options (restored after the test by the autouse fixture in conftest.py)
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

    # Column-only options (fast path): booleans and str are never rounded, arrays are
    df3 = pd.DataFrame(
        {
            "flag": [True, False],
            "name": ["a", "b"],
            "CI95": [np.array([0.123, 1.987])] * 2,
            "BF10": [1.23456, 0.0000123],
        }
    )
    pingouin.options.clear()
    pingouin.options["round"] = 1
    pingouin.options["round.column.BF10"] = pingouin.config._format_bf
    df3 = _postprocess_dataframe(df3)
    assert df3["flag"].tolist() == [True, False]
    assert df3["name"].tolist() == ["a", "b"]
    np.testing.assert_array_equal(df3.at[0, "CI95"], [0.1, 2.0])
    assert df3["BF10"].tolist() == ["1.235", "1.23e-05"]

    # Missing default rounding option: no rounding
    pingouin.options.clear()
    pd.testing.assert_frame_equal(_postprocess_dataframe(df), df)

    # Invalid rounding options
    for opt in [1.5, "2", True]:
        pingouin.options["round"] = opt
        with pytest.raises(TypeError, match="Rounding options"):
            _postprocess_dataframe(df)


def test_get_round_setting_for():
    """Test function _get_round_setting_for."""
    # set rounding options (restored after the test by the autouse fixture in conftest.py)
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
    del pingouin.options["round"]
    assert _get_round_setting_for("row2", "Values2") is None


def test_flatten_list():
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


def test_perm_pval():
    """Test function _perm_pval."""
    bootstat = np.random.default_rng(123).normal(size=1000)
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
    # Values within floating-point error of the estimate are counted as ties
    x = 0.1 + 0.2  # 0.30000000000000004
    bootstat = np.array([0.3, 0.3, 0.3, 0.1])
    assert _perm_pval(bootstat, x, alternative="greater") == 0.75
    assert _perm_pval(bootstat, -x, alternative="less") == 0
    assert _perm_pval(-bootstat, -x, alternative="less") == 0.75
    assert _perm_pval(-bootstat, x, alternative="two-sided") == 0.75


def test_remove_na():
    """Test function remove_na."""
    x = [6.4, 3.2, 4.5, np.nan]
    y = [3.5, 7.2, 8.4, 3.2]
    z = [2.3, np.nan, 5.2, 4.6]
    x_out, y_out = remove_na(x, y, paired=True)
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert np.allclose(y_out, [3.5, 7.2, 8.4])
    x_out, y_out = remove_na(x, y, paired=False)
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert np.allclose(y_out, y)
    y_out, x_out = remove_na(y, x, paired=False)
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert np.allclose(y_out, y)
    x_out, z_out = remove_na(x, z, paired=True)
    assert np.allclose(x_out, [6.4, 4.5])
    assert np.allclose(z_out, [2.3, 5.2])
    x_out, z_out = remove_na(x, z, paired=False)
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert np.allclose(z_out, [2.3, 5.2, 4.6])
    # When y is None or a scalar
    assert np.allclose(remove_na(x, None), [6.4, 3.2, 4.5])
    x_out, y_out = remove_na(x, 4)
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert y_out == 4
    # When y has a single element, it is passed through
    x_out, y_out = remove_na(x, [np.nan])
    assert np.allclose(x_out, [6.4, 3.2, 4.5])
    assert np.isnan(y_out).all()
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
    assert np.allclose(remove_na(x, None, paired=False), [[4.0, 2.0], [7.0, 6.0]])
    assert np.allclose(remove_na(x, None, axis="columns"), [[4.0], [4.0], [7.0]])
    # When y is an empty list
    # See https://github.com/raphaelvallat/pingouin/issues/222
    with pytest.raises(AssertionError):
        remove_na(x, y=[])


def test_check_eftype():
    """Test function _check_eftype."""
    for eftype in ["none", "hedges", "cohen", "Cohen", "cohen_dz", "r", "eta_square", "cles"]:
        assert _check_eftype(eftype)
    for eftype in ["fake", "cohen-d", ""]:
        assert not _check_eftype(eftype)


def test_check_dataframe():
    """Test function _check_dataframe."""

    # DataMatrix objects are converted to pandas DataFrame
    class DataMatrix:
        pass

    fake_datamatrix = types.ModuleType("datamatrix")
    fake_datamatrix.DataMatrix = DataMatrix
    fake_datamatrix.convert = types.SimpleNamespace(to_pandas=lambda dm: df)
    with pytest.MonkeyPatch.context() as mp:
        # DataMatrix not installed
        mp.setitem(sys.modules, "datamatrix", None)
        with pytest.raises(ValueError, match="DataMatrix not available"):
            _check_dataframe(dv="Values", between="Group", effects="between", data=DataMatrix())
        mp.setitem(sys.modules, "datamatrix", fake_datamatrix)
        out = _check_dataframe(dv="Values", between="Group", effects="between", data=DataMatrix())
        pd.testing.assert_frame_equal(out, df)
        # An object that is only named DataMatrix
        other = type("DataMatrix", (), {})()
        with pytest.raises(ValueError, match="compatible object"):
            _check_dataframe(dv="Values", between="Group", effects="between", data=other)
    # Valid inputs: the dataframe is returned unchanged
    out = _check_dataframe(dv="Values", between="Group", effects="between", data=df)
    assert out is df
    out = _check_dataframe(dv="Values", within="Time", subject="Subject", effects="within", data=df)
    assert out is df
    out = _check_dataframe(
        dv="Values",
        within="Time",
        subject="Subject",
        between="Group",
        effects="interaction",
        data=df,
    )
    assert out is df
    # Unsigned integer DV
    df_uint = df.astype({"Values": "uint8"})
    out = _check_dataframe(dv="Values", between="Group", effects="between", data=df_uint)
    assert out is df_uint
    # Wrond and or missing arguments
    with pytest.raises(ValueError, match="numeric"):
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


def test_is_installed():
    """Test function _is_installed."""
    assert _is_installed("numpy")
    assert not _is_installed("a_module_that_does_not_exist")
    with pytest.raises(ImportError, match="pingouin\\[extras\\]"):
        _is_installed("a_module_that_does_not_exist", raise_error=True)
