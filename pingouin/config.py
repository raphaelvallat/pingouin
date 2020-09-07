from .bayesian import _format_bf

"""Pingouin global configuration."""

__all__ = ['options', 'set_default_options']

options = {}


def set_default_options():
    """Reset Pingouin's default global options (e.g. rounding).

    .. versionadded:: 0.3.8
    """
    options.clear()

    # Rounding behavior
    options['round'] = None
    options['round.column.CI95%'] = 2
    # default is to return Bayes factors inside DataFrames as formatted str
    options['round.column.BF10'] = _format_bf
