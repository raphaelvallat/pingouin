import numpy as np
import pytest
from _pytest.doctest import DoctestItem


@pytest.fixture(autouse=True)
def _doctest_legacy_numpy_repr(request):
    """Print NumPy scalars as ``1.0`` instead of ``np.float64(1.0)`` (NumPy >= 2) in doctests."""
    if isinstance(request.node, DoctestItem):
        with np.printoptions(legacy="1.25"):
            yield
    else:
        yield
