import matplotlib

matplotlib.use("Agg")

import pytest

import pingouin


@pytest.fixture(autouse=True)
def _restore_options():
    """Restore the global ``pingouin.options`` after each test, even if the test fails."""
    old_options = pingouin.options.copy()
    try:
        yield
    finally:
        pingouin.options.clear()
        pingouin.options.update(old_options)
