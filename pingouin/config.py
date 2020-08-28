"""Pingouin global configuration."""

__all__ = ['options', 'set_default_options']

options = {}


def set_default_options():
    """Reset Pingouin's default global options (e.g. rounding).

    Examples
    --------
    >>> import pingouin as pg
    >>> pg.options
    {'round': None, 'round.column.CI95%': 2}
    """
    options.clear()

    # Rounding behavior
    options['round'] = None
    options['round.column.CI95%'] = 2
