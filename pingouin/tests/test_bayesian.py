from pingouin.tests._tests_pingouin import _TestPingouin
from pingouin.bayesian import bayesfactor_ttest


class TestBayesian(_TestPingouin):
    """Test bayesian.py."""

    def test_bayesfactor_ttest(self):
        """Test function bayesfactor_ttest."""
        bf = bayesfactor_ttest(3.5, 20, 20)
        assert bf == 26.743
        assert bayesfactor_ttest(3.5, 20) == 17.185
        assert bayesfactor_ttest(3.5, 20, 1) == 17.185
