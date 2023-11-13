import pytest
import numpy as np
from unittest import TestCase
from pingouin.power import (
    power_ttest,
    power_ttest2n,
    power_anova,
    power_rm_anova,
    power_corr,
    power_chi2,
)


class TestPower(TestCase):
    """Test power.py."""

    def test_power_ttest(self):
        """Test function power_ttest.
        Values are compared to the pwr R package.
        """
        # ONE-SAMPLE / PAIRED
        # Alternative = 'greater'
        assert np.isclose(
            power_ttest(d=0.5, n=20, contrast="one-sample", alternative="greater"), 0.6951493
        )
        assert np.isclose(
            power_ttest(d=0.5, n=20, contrast="paired", alternative="greater"), 0.6951493
        )
        assert np.isclose(
            power_ttest(d=0.5, power=0.80, contrast="one-sample", alternative="greater"), 26.13753
        )
        assert np.isclose(
            power_ttest(n=20, power=0.80, contrast="one-sample", alternative="greater"), 0.5769185
        )
        assert np.isclose(
            power_ttest(
                d=0.5, n=20, power=0.80, alpha=None, alternative="greater", contrast="one-sample"
            ),
            0.09004593,
            rtol=1e-03,
        )
        # Alternative = 'less'
        assert np.isclose(
            power_ttest(d=0.5, n=20, contrast="one-sample", alternative="less"), 7.083752e-05
        )
        assert np.isclose(
            power_ttest(d=-0.5, n=20, contrast="one-sample", alternative="less"), 0.6951493
        )
        assert np.isclose(
            power_ttest(d=0.5, n=20, contrast="paired", alternative="less"), 7.083752e-05
        )
        assert np.isclose(
            power_ttest(d=-0.5, power=0.80, contrast="one-sample", alternative="less"), 26.13753
        )
        assert np.isclose(
            power_ttest(n=20, power=0.80, contrast="one-sample", alternative="less"), -0.5769185
        )
        assert np.isclose(
            power_ttest(
                d=-0.5, n=20, power=0.80, alpha=None, alternative="less", contrast="one-sample"
            ),
            0.09004593,
            rtol=1e-03,
        )
        # Two-sided
        assert np.isclose(power_ttest(d=0.5, n=20, contrast="one-sample"), 0.5645044, rtol=1e-03)
        assert np.isclose(power_ttest(d=0.5, power=0.80, contrast="one-sample"), 33.36713)
        assert np.isclose(power_ttest(n=20, power=0.80, contrast="one-sample"), 0.6604413)
        assert np.isclose(
            power_ttest(d=0.5, n=20, power=0.80, alpha=None, contrast="one-sample"),
            0.1798043,
            rtol=1e-02,
        )

        # TWO-SAMPLES
        # Alternative = 'greater'
        assert np.isclose(power_ttest(d=0.5, n=20, alternative="greater"), 0.4633743)
        assert np.isclose(power_ttest(d=0.5, power=0.80, alternative="greater"), 50.1508)
        assert np.isclose(power_ttest(n=20, power=0.80, alternative="greater"), 0.8006879)
        assert np.isclose(
            power_ttest(d=0.5, n=20, power=0.80, alpha=None, alternative="greater"),
            0.2315111,
            rtol=1e-01,
        )
        # Alternative = 'less'
        assert np.isclose(power_ttest(d=-0.5, n=20, alternative="less"), 0.4633743)
        assert np.isclose(power_ttest(d=-0.5, power=0.80, alternative="less"), 50.1508)
        assert np.isclose(power_ttest(n=20, power=0.80, alternative="less"), -0.8006879)
        assert np.isclose(
            power_ttest(d=-0.5, n=20, power=0.80, alpha=None, alternative="less"),
            0.2315111,
            rtol=1e-01,
        )
        # Two-sided
        assert np.isclose(power_ttest(d=0.5, n=20), 0.337939, rtol=1e-03)
        assert np.isclose(power_ttest(d=0.5, power=0.80), 63.76561)
        assert np.isclose(power_ttest(n=20, power=0.80), 0.9091587, rtol=1e-03)
        assert np.isclose(power_ttest(d=0.5, n=20, power=0.80, alpha=None), 0.4430163, rtol=1e-01)
        # Error
        with pytest.raises(ValueError):
            power_ttest(d=0.5)

    def test_power_ttest2n(self):
        """Test function power_ttest2n.
        Values are compared to the pwr R package.
        """
        # TWO-SAMPLES
        # alternative = 'greater'
        assert np.isclose(power_ttest2n(nx=20, ny=18, d=0.5, alternative="greater"), 0.4463552)
        assert np.isclose(power_ttest2n(nx=20, ny=18, power=0.80, alternative="greater"), 0.8234684)
        assert np.isclose(
            power_ttest2n(nx=20, ny=18, d=0.5, power=0.80, alpha=None, alternative="greater"),
            0.2444025,
            rtol=1e-01,
        )
        # alternative = 'less'
        assert np.isclose(power_ttest2n(nx=20, ny=18, d=0.5, alternative="less"), 0.0008011104)
        assert np.isclose(power_ttest2n(nx=20, ny=18, d=-0.5, alternative="less"), 0.4463552)
        assert np.isclose(power_ttest2n(nx=20, ny=18, power=0.80, alternative="less"), -0.8234684)
        assert np.isclose(
            power_ttest2n(nx=20, ny=18, d=0.5, power=0.80, alpha=None, alternative="less"),
            0.989896,
            rtol=1e-01,
        )
        # Two-sided
        assert np.isclose(power_ttest2n(nx=20, ny=18, d=0.5), 0.3223224, rtol=1e-03)
        assert np.isclose(power_ttest2n(nx=20, ny=18, power=0.80), 0.9354168)
        assert np.isclose(
            power_ttest2n(nx=20, ny=18, d=0.5, power=0.80, alpha=None), 0.46372, rtol=1e-01
        )
        # Error
        with pytest.raises(ValueError):
            power_ttest2n(nx=20, ny=20)

    def test_power_anova(self):
        """Test function power_anova.
        Values are compared to the pwr R package.
        """
        eta = 0.0727003
        assert np.isclose(power_anova(eta_squared=eta, k=4, n=20), 0.5149793)
        assert np.isclose(power_anova(eta_squared=eta, n=20, power=0.80), 10.70313)
        assert np.isclose(power_anova(eta_squared=eta, k=4, power=0.80), 35.75789)
        assert np.isclose(power_anova(k=4, n=20, power=0.80, alpha=0.05), 0.1254838, rtol=1e-03)
        assert np.isclose(
            power_anova(eta_squared=eta, k=4, n=20, power=0.80, alpha=None), 0.2268337
        )
        # Error
        with pytest.raises(ValueError):
            power_anova(eta_squared=eta, k=2)

    def test_power_rm_anova(self):
        """Test function power_rm_anova.
        Values are compared to GPower.
        """
        eta = 0.20  # f = 0.5
        eta2 = 0.058823529411764705  # f = 0.25 (default in GPower)

        # Power
        assert np.isclose(
            power_rm_anova(eta_squared=eta, m=4, n=20, epsilon=1, corr=0.5), 0.9998070
        )
        assert np.isclose(
            power_rm_anova(eta_squared=eta, m=3, n=20, epsilon=0.9, corr=0.5), 0.9968277
        )
        assert np.isclose(
            power_rm_anova(eta_squared=eta2, m=3, n=10, epsilon=1, corr=0.7), 0.5271828
        )
        assert np.isclose(
            power_rm_anova(eta_squared=eta2, m=2, n=18, epsilon=1, corr=0.6), 0.6089353
        )
        assert np.isclose(
            power_rm_anova(eta_squared=eta2, m=3, n=10, corr=-0.5), 0.13818, rtol=1e-4
        )

        # Number of repeated measures
        assert np.isclose(power_rm_anova(eta_squared=eta2, n=30, power=0.9), 3.9274700428)
        assert np.isclose(power_rm_anova(eta_squared=eta2, n=20, power=0.8), 5.129210021)

        # Sample size
        assert np.ceil(power_rm_anova(eta_squared=eta, m=3, power=0.80)) == 9
        assert np.ceil(power_rm_anova(eta_squared=eta2, m=4, power=0.80)) == 24
        assert np.ceil(power_rm_anova(eta2, m=3, power=0.90, corr=0.8)) == 16
        assert np.ceil(power_rm_anova(eta2, 3, power=0.90, epsilon=0.9)) == 39

        # Effect size (eta-square)
        assert np.isclose(power_rm_anova(n=20, m=3, power=0.80), 0.0800112)
        assert np.isclose(power_rm_anova(n=20, m=2, power=0.90), 0.12747503)
        assert np.isclose(power_rm_anova(n=50, m=4, power=0.95, corr=0.7), 0.02576957)

        # Alpha
        assert np.isclose(
            power_rm_anova(eta_squared=eta, m=3, n=50, power=0.95, alpha=None), 3.652063e-9
        )
        assert np.isclose(
            power_rm_anova(eta_squared=eta2, m=2, n=100, power=0.95, corr=0.6, alpha=None),
            0.0001797,
            rtol=1e-4,
        )

        # Error
        with pytest.raises(ValueError):
            power_rm_anova(eta_squared=eta, m=2)

    def test_power_corr(self):
        """Test function power_corr.
        Values are compared to the pwr R package.
        """
        # Two-sided
        assert np.isclose(power_corr(r=0.5, n=20), 0.6378746)
        assert np.isclose(power_corr(r=0.5, power=0.80), 28.24841)
        assert np.isclose(power_corr(n=20, power=0.80), 0.5821478)
        assert np.isclose(power_corr(r=0.5, n=20, power=0.80, alpha=None), 0.1377332, rtol=1e-03)

        # Greater
        assert np.isclose(power_corr(r=0.5, n=20, alternative="greater"), 0.7509873)
        assert np.isclose(power_corr(r=-0.1, n=20, alternative="greater"), 0.01941224)
        assert np.isclose(power_corr(r=0.5, power=0.80, alternative="greater"), 22.60907)
        assert np.isclose(power_corr(n=20, power=0.80, alternative="greater"), 0.5286949)

        # Less
        assert np.isclose(power_corr(r=-0.5, n=20, alternative="less"), 0.7509873)
        assert np.isclose(power_corr(r=-0.1, n=20, alternative="less"), 0.1118106)
        assert np.isclose(power_corr(r=0.1, n=20, alternative="less"), 0.01941224)
        assert np.isclose(power_corr(r=-0.5, power=0.80, alternative="less"), 22.60907)
        assert np.isclose(power_corr(n=20, power=0.80, alternative="less"), -0.5286949)

        # Error & Warning
        with pytest.raises(ValueError):
            power_corr(r=0.5)
        with pytest.warns(UserWarning):
            power_corr(r=0.5, n=4)
            power_corr(power=0.80, n=4)

    def test_power_chi2(self):
        """Test function power_chi2.
        Values are compared to the pwr R package.
        """
        # Power
        assert np.isclose(power_chi2(dof=1, w=0.3, n=20), 0.2686618)
        assert np.isclose(power_chi2(dof=2, w=0.3, n=100), 0.7706831)

        # Sample size
        assert np.isclose(power_chi2(dof=1, w=0.3, power=0.80), 87.20954)
        assert np.isclose(power_chi2(dof=3, w=0.3, power=0.80), 121.1396)

        # Effect size
        assert np.isclose(power_chi2(dof=4, n=50, power=0.80), 0.4885751)
        assert np.isclose(power_chi2(dof=1, n=50, power=0.80), 0.3962023)

        # Alpha
        assert np.isclose(
            power_chi2(dof=1, w=0.3, n=100, power=0.80, alpha=None), 0.03089736, atol=1e-03
        )

        # Error
        with pytest.raises(ValueError):
            power_chi2(1, w=0.3)
