import pandas as pd
import numpy as np
import pytest

import pingouin

from unittest import TestCase
from pingouin.config import _set_default_options

expected_default_options = {
    'round': None
}

class TestConfig(TestCase):
    """Test config.py."""

    def test__set_default_options(self):
        """Test function _set_default_options."""
        old_opts = pingouin.options.copy()
        pingouin.options.clear()
        _set_default_options()
        assert pingouin.options == expected_default_options
        pingouin.options.update(old_opts)