from collections import namedtuple

__all__ = ['options', '_set_default_options']

options = {}


def _set_default_options():
    options['round'] = 4
