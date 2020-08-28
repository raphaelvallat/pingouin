import pingouin
from unittest import TestCase
from pingouin.config import set_default_options

expected_default_options = pingouin.options.copy()


class TestConfig(TestCase):
    """Test config.py."""

    def test_set_default_options(self):
        """Test function set_default_options."""
        old_opts = pingouin.options.copy()
        pingouin.options.clear()
        set_default_options()
        assert pingouin.options == expected_default_options
        pingouin.options.update(old_opts)
