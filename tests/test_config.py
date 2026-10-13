import pytest

import pingouin
from pingouin.config import _no_rounding, set_default_options

expected_default_options = pingouin.options.copy()


def test_set_default_options():
    """Test function set_default_options."""
    pingouin.options.clear()
    set_default_options()
    assert pingouin.options == expected_default_options
    # Custom options are reset
    pingouin.options["round"] = 2
    pingouin.options["round.column.T"] = 1
    set_default_options()
    assert pingouin.options == expected_default_options


def test_no_rounding():
    """Test the context manager _no_rounding."""
    pingouin.options["round"] = 3
    pingouin.options["round.column.T"] = 1
    old_options = pingouin.options.copy()
    with _no_rounding():
        assert pingouin.options == {"round": None}
    assert pingouin.options == old_options
    # The options are restored even if an exception is raised
    with pytest.raises(RuntimeError), _no_rounding():
        raise RuntimeError
    assert pingouin.options == old_options


def test_format_bf():
    """Test function _format_bf."""
    assert pingouin.config._format_bf(1.23456) == "1.235"
    assert pingouin.config._format_bf(3.0) == "3.0"
    assert pingouin.config._format_bf(12345.6) == "1.235e+04"
    assert pingouin.config._format_bf(0.0000123) == "1.23e-05"
