"""Pingouin global configuration."""

__all__ = ['options', '_set_default_options']

options = {}


def _set_default_options():
    """Set default options."""
    options['round'] = None
