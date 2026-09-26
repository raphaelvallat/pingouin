"""Pingouin global configuration."""

from contextlib import contextmanager

import numpy as np

__all__ = ["options", "set_default_options"]

options = {}


def _format_bf(bf, precision=3, trim="0"):
    """Format BF10 to floating point or scientific notation."""
    if bf >= 1e4 or bf <= 1e-4:
        return np.format_float_scientific(bf, precision=precision, trim=trim)
    return np.format_float_positional(bf, precision=precision, trim=trim)


def set_default_options():
    """Reset Pingouin's default global options (e.g. rounding).

    .. versionadded:: 0.3.8
    """
    options.clear()

    # Rounding behavior
    options["round"] = None
    options["round.column.CI95"] = 2
    # default is to return Bayes factors inside DataFrames as formatted str
    options["round.column.BF10"] = _format_bf


@contextmanager
def _no_rounding():
    """Temporarily disable rounding, e.g. when calling public functions internally.

    Both the global ``round`` option and all the per-column/row/cell ``round.*`` overrides
    (including formatters such as the Bayes Factor one) are disabled, so that the internal results
    are kept at full precision. Rounding and formatting are then applied once, on the final
    output. The original options are always restored, even if an exception is raised.
    """
    old_options = options.copy()
    for key in [k for k in options if k.startswith("round.")]:
        del options[key]
    options["round"] = None
    try:
        yield
    finally:
        options.clear()
        options.update(old_options)
