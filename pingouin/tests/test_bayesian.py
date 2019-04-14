from unittest import TestCase
from pingouin.bayesian import bayesfactor_ttest, bayesfactor_pearson


class TestBayesian(TestCase):
    """Test bayesian.py."""

    def test_bayesfactor_ttest(self):
        """Test function bayesfactor_ttest."""
        bf = bayesfactor_ttest(3.5, 20, 20)
        assert float(bf) == 26.743
        assert float(bayesfactor_ttest(3.5, 20)) == 17.185
        assert float(bayesfactor_ttest(3.5, 20, 1)) == 17.185

    def test_bayesfactor_pearson(self):
        """Test function bayesfactor_pearson."""
        assert float(bayesfactor_pearson(0.6, 20)) == 8.221
        assert float(bayesfactor_pearson(-0.6, 20)) == 8.221
        assert float(bayesfactor_pearson(0.6, 10)) == 1.278
