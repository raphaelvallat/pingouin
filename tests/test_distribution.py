from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from pingouin import read_dataset
from pingouin.distribution import (
    anderson,
    epsilon,
    homoscedasticity,
    normality,
    sphericity,
)

# Generate random dataframe
df = read_dataset("mixed_anova.csv")
df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan
df_pivot = df.pivot(index="Subject", columns="Time", values="Scores").reset_index(drop=True)

# Create random normal variables
np.random.seed(1234)
x = np.random.normal(scale=1.0, size=100)
y = np.random.normal(scale=0.8, size=100)
z = np.random.normal(scale=0.9, size=100)

# Two-way repeated measures (epsilon and sphericity)
data = read_dataset("rm_anova2")
idx, dv = "Subject", "Performance"
within = ["Time", "Metric"]
pa = data.pivot_table(index=idx, columns=within[0], values=dv)
pb = data.pivot_table(index=idx, columns=within[1], values=dv)
pab = data.pivot_table(index=idx, columns=within, values=dv)
# pab_single is a multilevel dataframe with shape columns.levshape = (1, 3)
pab_single = pab.xs("Pre", level=0, drop_level=False, axis=1)
# Create a 3-level mutlilevel columns (to test ValueError)
pab_3fac = pab_single.copy()
old_idx = pab_single.columns.to_frame()
old_idx.insert(0, "new_level_name", [0, 0, 0])
pab_3fac.columns = pd.MultiIndex.from_frame(old_idx)

# Two-way repeated measures (3, 4)
np.random.seed(123)
dv = np.random.normal(scale=3, size=600)
w1 = np.repeat(["P1", "P2", "P3"], 200)
w2 = np.tile(np.repeat(["A", "B", "C", "D"], 50), 3)
subj = np.tile(np.tile(np.arange(50), 4), 3)
df3 = pd.DataFrame({"dv": dv, "within1": w1, "within2": w2, "subj": subj})
idx, dv = "subj", "dv"
within = ["within1", "within2"]
pa1 = df3.pivot_table(index=idx, columns=within[0], values=dv)
pb1 = df3.pivot_table(index=idx, columns=within[1], values=dv)
pab1 = df3.pivot_table(index=idx, columns=within, values=dv)


class TestDistribution(TestCase):
    """Test distribution.py."""

    def test_normality(self):
        """Test function test_normality."""
        # List / 1D array
        normality(x, alpha=0.05)
        normality(x.tolist(), method="normaltest", alpha=0.05)
        # Pandas DataFrame
        df_nan_piv = df_nan.pivot(index="Subject", columns="Time", values="Scores")
        normality(df_nan_piv)  # Wide-format dataframe
        normality(df_nan_piv["August"])  # pandas Series
        # The line below is disabled because test fails on python 3.5
        # assert stats_piv.equals(normality(df_nan, group='Time', dv='Scores'))
        normality(df_nan, group="Group", dv="Scores", method="normaltest")
        normality(df_nan, group="Group", dv="Scores", method="jarque_bera")
        # Long-format but some groups are too small
        # https://github.com/raphaelvallat/pingouin/issues/319
        df_small = pd.DataFrame(
            {
                "group": [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3],
                "variable": [5, 6, 8, 9, 5, 5, 2, 3, 4, 5, np.nan, 8],
            }
        )
        stats = normality(df_small, dv="variable", group="group")
        assert np.isnan(stats.loc[[1, 3], "pval"]).all()
        assert not np.isnan(stats.loc[[0, 2], "pval"]).all()
        assert (~stats.loc[[1, 3], "normal"]).all()

    def test_homoscedasticity(self):
        """Test function test_homoscedasticity."""
        hl = homoscedasticity(data=[x, y], alpha=0.05)
        # Method name is case-insensitive
        assert hl.equals(homoscedasticity(data=[x, y], method="Levene", alpha=0.05))
        homoscedasticity(data=[x, y], method="bartlett", alpha=0.05)
        hd = homoscedasticity(data={"x": x, "y": y}, alpha=0.05)
        hd2 = homoscedasticity(data={"x": x, "y": y}, alpha=0.05, center="mean")
        assert hl.equals(hd)
        assert not hd.equals(hd2)
        # Wide-format DataFrame
        homoscedasticity(df_pivot)
        # Long-format
        homoscedasticity(df, dv="Scores", group="Time")
        # Integer inputs (scipy.stats.bartlett fails with integers in SciPy >= 1.17)
        a, b = [4, 8, 9, 20, 14], [5, 8, 15, 45, 12]
        for data in [[a, np.array(b)], {"a": a, "b": b}, pd.DataFrame({"a": a, "b": b})]:
            hb = homoscedasticity(data, method="bartlett")
            assert np.isclose(hb.at["bartlett", "T"], 2.873569)
            assert np.isclose(hb.at["bartlett", "pval"], 0.090045)
        # Missing values are removed separately in each sample
        a_nan, b_nan = [*a, np.nan], [np.nan, *b, np.nan]
        hb = homoscedasticity(data={"a": a, "b": b}, method="bartlett")
        for data in [[a_nan, b_nan], pd.DataFrame({"a": [*a_nan, np.nan], "b": b_nan})]:
            assert hb.equals(homoscedasticity(data, method="bartlett"))
        df_nan = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 7, "y": a_nan + b_nan})
        assert hb.equals(homoscedasticity(df_nan, dv="y", group="g", method="bartlett"))

    def test_epsilon(self):
        """Test function epsilon."""
        df_pivot = df.pivot(index="Subject", columns="Time", values="Scores").reset_index(drop=True)
        eps_gg = epsilon(df_pivot)
        eps_hf = epsilon(df_pivot, correction="hf")
        eps_lb = epsilon(df_pivot, correction="lb")
        # In long-format
        eps_gg_rm = epsilon(df, subject="Subject", within="Time", dv="Scores")
        eps_hf_rm = epsilon(df, subject="Subject", within="Time", dv="Scores", correction="hf")
        assert np.isclose(eps_gg_rm, eps_gg)
        assert np.isclose(eps_hf_rm, eps_hf)
        # Compare with ezANOVA
        assert np.allclose([eps_gg, eps_hf, eps_lb], [0.9987509, 1, 0.5])

        # Time has only two values so epsilon is one.
        assert epsilon(pa, correction="lb") == epsilon(pa, correction="gg")
        assert epsilon(pa, correction="gg") == epsilon(pa, correction="hf")
        # Lower bound <= Greenhouse-Geisser <= Huynh-Feldt
        assert epsilon(pb, correction="lb") <= epsilon(pb, correction="gg")
        assert epsilon(pb, correction="gg") <= epsilon(pb, correction="hf")
        assert epsilon(pab, correction="lb") <= epsilon(pab, correction="gg")
        assert epsilon(pab, correction="gg") <= epsilon(pab, correction="hf")
        # Lower bound == 0.5 for pb and pab
        assert epsilon(pb, correction="lb") == epsilon(pab, correction="lb")
        assert np.allclose(epsilon(pb), 0.9691030)  # ez
        assert np.allclose(epsilon(pb, correction="hf"), 1.0)  # ez
        # Epsilon for the interaction (shape = (2, N))
        assert np.allclose(epsilon(pab), 0.7271664)
        assert np.allclose(epsilon(pab, correction="hf"), 0.831161)
        assert epsilon(pab) == epsilon(pab.swaplevel(axis=1))
        assert epsilon(pab_single) == epsilon(pab_single.swaplevel(axis=1))
        eps_gg_rm = epsilon(data, subject="Subject", dv="Performance", within=["Time", "Metric"])
        assert eps_gg_rm == epsilon(pab)
        # Now with a (3, 4) two-way design
        assert np.allclose(epsilon(pa1), 0.9963275)
        assert np.allclose(epsilon(pa1, correction="hf"), 1.0)
        assert np.allclose(epsilon(pb1), 0.9716288)
        assert np.allclose(epsilon(pb1, correction="hf"), 1.0)
        # Interaction of a (3, 4) design: compare with R afex::aov_ez
        # See https://github.com/raphaelvallat/pingouin/issues/19
        assert np.isclose(epsilon(pab1), 0.8562148)
        assert np.isclose(epsilon(pab1, correction="hf"), 0.9684172)
        assert epsilon(pab1, correction="lb") == 1 / 6
        assert np.isclose(epsilon(pab1), epsilon(pab1.swaplevel(axis=1)))
        # The order of the columns does not matter
        assert np.isclose(epsilon(pab1), epsilon(pab1.iloc[:, ::-1]))
        eps_gg_rm = epsilon(df3, subject="subj", dv="dv", within=["within1", "within2"])
        assert eps_gg_rm == epsilon(pab1)
        # With missing values
        eps_gg_rm = epsilon(df_nan, subject="Subject", within="Time", dv="Scores")
        # 3 repeated measures factor
        with pytest.raises(ValueError):
            epsilon(pab_3fac)

    def test_sphericity(self):
        """Test function test_sphericity.
        Compare with ezANOVA.
        """
        _, W, _, _, p = sphericity(df_pivot, method="mauchly")
        assert round(W, 3) == 0.999
        assert np.round(p, 3) == 0.964
        _, W, _, _, p = sphericity(df, dv="Scores", subject="Subject", within="Time")  # Long-format
        assert round(W, 3) == 0.999
        assert np.round(p, 3) == 0.964
        assert sphericity(pa)[0]  # Only two levels so sphericity = True
        spher = sphericity(pb)
        assert spher[0]
        assert round(spher[1], 3) == 0.968  # W
        assert spher[3] == 2  # dof
        assert np.isclose(spher[4], 0.8784418)  # P-value
        # JNS
        sphericity(df_pivot, method="jns")
        sphericity(df, dv="Scores", subject="Subject", within=["Time"], method="jns")
        # Under the null hypothesis, JNS should reject sphericity at the nominal rate
        rng = np.random.default_rng(42)
        pvals = [
            sphericity(pd.DataFrame(rng.normal(size=(50, 4))), method="jns")[4] for _ in range(200)
        ]
        assert 0.01 < np.mean(np.array(pvals) < 0.05) < 0.1
        # Two-way design of shape (2, N)
        spher = sphericity(pab)
        assert round(spher[1], 3) == 0.625
        assert spher[3] == 2
        assert np.isclose(spher[4], 0.1523917)
        assert sphericity(pab)[1] == sphericity(pab.swaplevel(axis=1))[1]
        spher_long = sphericity(
            data, subject="Subject", dv="Performance", within=["Time", "Metric"]
        )
        assert round(spher_long[1], 3) == 0.625
        assert np.isclose(spher[4], spher_long[4])
        sphericity(pab_single)  # For coverage
        # Now with a (3, 4) two-way design
        # First, main effect
        spher = sphericity(pb1)
        assert spher[0]
        assert round(spher[1], 3) == 0.958  # W
        assert round(spher[4], 4) == 0.8436  # P-value
        spher2 = sphericity(df3, subject="subj", dv="dv", within=["within2"])
        assert spher[1] == spher2[1]
        assert spher[4] == spher2[4]
        # And then interaction: compare with R afex::aov_ez
        # See https://github.com/raphaelvallat/pingouin/issues/19
        spher = sphericity(pab1)
        assert spher[0]
        assert np.isclose(spher[1], 0.58944, atol=1e-5)  # W
        assert spher[3] == 20  # dof
        assert np.isclose(spher[4], 0.21311, atol=1e-5)  # P-value
        spher_long = sphericity(df3, subject="subj", dv="dv", within=["within1", "within2"])
        assert np.isclose(spher[4], spher_long[4])
        assert np.isclose(spher[4], sphericity(pab1.swaplevel(axis=1))[4])
        assert np.isclose(spher[4], sphericity(pab1.iloc[:, ::-1])[4])
        sphericity(pab1, method="jns")  # For coverage
        # Missing combination of the two within-subject factors
        with pytest.raises(ValueError, match="exactly once"):
            sphericity(pab1.iloc[:, 1:])
        # 3 repeated measures factor
        with pytest.raises(ValueError):
            sphericity(pab_3fac)
        # With missing values
        sphericity(df_nan, subject="Subject", within="Time", dv="Scores")

    def test_anderson(self):
        """Test function test_anderson."""
        assert not anderson(np.random.random(size=1000))[0]
        assert anderson(np.random.normal(size=10000))[0]
