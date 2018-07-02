import pytest
import numpy as np
from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.circular import circ_corrcc


class TestCircular(_TestPingouin):
    """Test circular.py."""

    def test_circ_corrcc(self):
        """Test function circ_corrcc."""
        x = [0.785, 1.570, 3.141, 3.839, 5.934]
        y = [0.593, 1.291, 2.879, 3.892, 6.108]
        r, pval = circ_corrcc(x, y)
        # Compare with the CircStats MATLAB toolbox
        assert r == 0.942
        assert np.round(pval, 3) == 0.066
        _, pval2 = circ_corrcc(x, y, tail='one-sided')
        assert pval2 == pval / 2
        # Wrong argument
        with pytest.raises(ValueError):
            circ_corrcc(x, [0.52, 1.29, 2.87])
