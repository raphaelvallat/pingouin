"""Test pairwise.py.

pairwise_tests is tested against the pairwise.t.test R function, as well as JASP and JAMOVI.

Notes:
1) JAMOVI by default pool the error term for the within-subject factor in mixed design. Pingouin
does not pool the error term, which is the same behavior as JASP.

2) JASP does not return the uncorrected p-values, therefore only the corrected p-values are
compared.

3) JASP does not calculate the Bayes Factor for the interaction terms. For mixed design and
two-way design, in JASP, the Bayes Factor seems to be calculated without aggregating over repeated
measurements.

4) For factorial between-subject contrasts, both JASP and JAMOVI pool the error term. This option
is not yet implemented in Pingouin. Therefore, one cannot directly validate the T and p-values.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind, ttest_rel

from pingouin import read_dataset
from pingouin.correlation import corr, partial_corr
from pingouin.multicomp import multicomp
from pingouin.nonparametric import mwu, wilcoxon
from pingouin.pairwise import (
    pairwise_corr,
    pairwise_gameshowell,
    pairwise_tests,
    pairwise_ttests,
    pairwise_tukey,
)


@pytest.fixture(scope="module")
def df():
    """Simple and mixed design."""
    return read_dataset("mixed_anova")


@pytest.fixture(scope="module")
def df_sort(df):
    """Same as ``df`` but with a different order of the subjects."""
    return df.sort_values("Time")


@pytest.fixture(scope="module")
def df_unb():
    """Unbalanced mixed design."""
    return read_dataset("mixed_anova_unbalanced")


@pytest.fixture(scope="module")
def df_rm2():
    """Two-way repeated measures design."""
    return read_dataset("rm_anova2")


@pytest.fixture(scope="module")
def df_aov2():
    """Two-way factorial design."""
    return read_dataset("anova2")


@pytest.fixture(scope="module")
def df_drug():
    """One within (Drug) and one between (Gender) factor."""
    return read_dataset("pairwise_tests")


@pytest.fixture(scope="module")
def df_hair():
    """Hair color dataset (one-way between design)."""
    return read_dataset("anova")


@pytest.fixture(scope="module")
def df_penguins():
    """Palmer penguins dataset (unbalanced between design)."""
    return read_dataset("penguins")


@pytest.fixture(scope="module")
def df_bfi():
    """JASP Big 5 dataset, without the subject column."""
    return read_dataset("pairwise_corr").iloc[:, 1:]


def _is_integer(dof):
    return dof.apply(lambda x: x.is_integer())


def test_pairwise_ttests_deprecated(df):
    """The deprecated pairwise_ttests warns and returns the same as pairwise_tests."""
    kwargs = dict(
        dv="Scores", within="Time", subject="Subject", data=df, return_desc=True, padjust="holm"
    )
    with pytest.warns(UserWarning, match="deprecated"):
        pt = pairwise_ttests(**kwargs)
    pd.testing.assert_frame_equal(pt, pairwise_tests(**kwargs))


def test_pairwise_tests_within(df, df_sort):
    """Test function pairwise_tests with one within-subject factor."""
    # In R:
    # >>> pairwise.t.test(df$Scores, df$Time, pool.sd = FALSE,
    # ...                 p.adjust.method = 'holm', paired = TRUE)
    pt = pairwise_tests(
        dv="Scores", within="Time", subject="Subject", data=df, return_desc=True, padjust="holm"
    )
    np.testing.assert_array_equal(pt.loc[:, "p_corr"].round(3), [0.174, 0.024, 0.310])
    np.testing.assert_array_equal(pt.loc[:, "p_unc"].round(3), [0.087, 0.008, 0.310])

    # Non-parametric: same as a Wilcoxon test on each pair
    pt_np = pairwise_tests(
        dv="Scores", within="Time", subject="Subject", data=df, parametric=False, return_desc=True
    )
    assert not pt_np["Parametric"].any()
    np.testing.assert_allclose(pt_np[["mean_A", "mean_B"]], pt[["mean_A", "mean_B"]])
    wide = df.pivot(index="Subject", columns="Time", values="Scores")
    for _, row in pt_np.iterrows():
        wx = wilcoxon(wide[row["A"]], wide[row["B"]])
        assert row["W_val"] == wx.at["Wilcoxon", "W_val"]
        assert row["p_unc"] == pytest.approx(wx.at["Wilcoxon", "p_val"])

    # Same after random ordering of subject (issue 151)
    pt_sort = pairwise_tests(
        dv="Scores",
        within="Time",
        subject="Subject",
        data=df_sort,
        return_desc=True,
        padjust="holm",
    )
    assert pt_sort.equals(pt)


def test_pairwise_tests_between(df, df_sort):
    """Test function pairwise_tests with one between-subject factor."""
    # In R: >>> pairwise.t.test(df$Scores, df$Group, pool.sd = FALSE)
    pt = pairwise_tests(dv="Scores", between="Group", data=df).round(3)
    assert pt.loc[0, "p_unc"] == 0.023

    # Non-parametric: same as a Mann-Whitney U test. With only two groups, there is nothing to
    # correct for.
    pt_np = pairwise_tests(
        dv="Scores",
        between="Group",
        data=df,
        padjust="bonf",
        alternative="greater",
        effsize="cohen",
        parametric=False,
    )
    assert "p_corr" not in pt_np.columns
    assert "cohen" in pt_np.columns
    x = df.loc[df["Group"] == "Control", "Scores"]
    y = df.loc[df["Group"] == "Meditation", "Scores"]
    mw = mwu(x, y, alternative="greater")
    assert pt_np.at[0, "U_val"] == mw.at["MWU", "U_val"]
    assert pt_np.at[0, "p_unc"] == pytest.approx(mw.at["MWU", "p_val"])

    # Same after random ordering of subject (issue 151)
    pt_sort = pairwise_tests(dv="Scores", between="Group", data=df_sort).round(3)
    assert pt_sort.equals(pt)


def test_pairwise_tests_mixed(df, df_sort):
    """Test function pairwise_tests with a balanced mixed design."""
    # .Balanced data
    # ..With marginal means
    pt = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df,
        padjust="holm",
        interaction=False,
    )
    # ...Within main effect: OK with JASP
    assert np.array_equal(pt["Paired"], [True, True, True, False])
    assert np.array_equal(pt.loc[:2, "p_corr"].round(3), [0.174, 0.024, 0.310])
    assert np.array_equal(pt.loc[:2, "BF10"].astype(float), [0.582, 4.232, 0.232])
    # ..Between main effect: T and p-values OK with JASP
    #   but BF10 is only similar when marginal=False (see note in the
    #   2-way RM test below).
    assert pt.loc[3, "T"].round(3) == -2.248
    assert pt.loc[3, "p_unc"].round(3) == 0.028
    # ..Interaction: slightly different because JASP pool the error term
    #    across the between-subject groups. JASP does not compute the BF10
    #    for the interaction.
    pt = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df,
        padjust="holm",
        interaction=True,
    ).round(5)
    # Same after random ordering of subject (issue 151)
    pt_sort = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df_sort,
        padjust="holm",
        interaction=True,
    ).round(5)
    assert pt_sort.equals(pt)

    # ..Changing the order of the model with ``within_first=False``.
    #   output model is now between + within + between * within.
    # https://github.com/raphaelvallat/pingouin/issues/102
    pt = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df,
        padjust="holm",
        within_first=False,
    )
    # This should be equivalent to manually filtering dataframe to keep
    # only one level at a time of the between factor and then running
    # a within-subject pairwise T-tests.
    pt_con = pairwise_tests(
        dv="Scores",
        within="Time",
        subject="Subject",
        padjust="holm",
        data=df[df["Group"] == "Control"],
    )
    pt_med = pairwise_tests(
        dv="Scores",
        within="Time",
        subject="Subject",
        padjust="holm",
        data=df[df["Group"] == "Meditation"],
    )
    pt_merged = pd.concat([pt_con, pt_med], ignore_index=True, sort=False)
    # T, dof and p-values should be equal
    assert np.array_equal(pt_merged["T"], pt["T"].iloc[4:])
    assert np.array_equal(pt_merged["dof"], pt["dof"].iloc[4:])
    assert np.array_equal(pt_merged["p_unc"], pt["p_unc"].iloc[4:])
    # However adjusted p-values are not equal because they are calculated
    # separately on each dataframe.
    assert not np.array_equal(pt_merged["p_corr"], pt["p_corr"].iloc[4:])

    # Other options: lists of one factor, non-parametric, FDR correction
    pt_np = pairwise_tests(
        dv="Scores",
        within=["Time"],
        between=["Group"],
        subject="Subject",
        data=df,
        padjust="fdr_bh",
        alpha=0.01,
        return_desc=True,
        parametric=False,
    )
    assert pt_np["Contrast"].tolist() == ["Time"] * 3 + ["Group"] + ["Time * Group"] * 3
    # Wilcoxon for the paired contrasts, Mann-Whitney U for the others
    assert pt_np["W_val"].notna().tolist() == pt_np["Paired"].tolist()
    assert pt_np["U_val"].notna().tolist() == (~pt_np["Paired"]).tolist()
    # The p-values are only corrected within a contrast, and never decreased by the correction
    assert pd.isna(pt_np.loc[3, "p_corr"])  # A single between test: nothing to correct
    for contrast in ["Time", "Time * Group"]:
        sub = pt_np[pt_np["Contrast"] == contrast]
        np.testing.assert_allclose(sub["p_corr"], multicomp(sub["p_unc"], method="fdr_bh")[1])


def test_pairwise_tests_mixed_unbalanced(df_unb):
    """Test function pairwise_tests with an unbalanced mixed design."""
    # ..With marginal means
    pt1 = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df_unb,
        padjust="bonf",
    )
    # ...Within main effect: OK with JASP
    assert np.array_equal(
        pt1.loc[:5, "T"].round(3), [-0.777, -1.344, -2.039, -0.814, -1.492, -0.627]
    )
    assert np.array_equal(pt1.loc[:5, "p_corr"].round(3), [1.0, 1.0, 0.313, 1.0, 0.889, 1.0])
    assert np.array_equal(
        pt1.loc[:5, "BF10"].astype(float), [0.273, 0.463, 1.221, 0.280, 0.554, 0.248]
    )
    # ...Between main effect: slightly different from JASP (why?)
    #      True with or without the Welch correction...
    assert (pt1.loc[6:8, "p_corr"] > 0.20).all()
    # ...Interaction: slightly different because JASP pool the error term
    #    across the between-subject groups.
    # Below the interaction JASP bonferroni-correct p-values, which are
    # more conservative because JASP perform all possible pairwise tests
    # jasp_pbonf = [1., 1., 1., 1., 1., 1., 1., 0.886, 1., 1., 1., 1.]
    assert (pt1.loc[9:, "p_corr"] > 0.05).all()
    # Check that the Welch corection is applied by default
    assert not _is_integer(pt1["dof"]).all()

    # Same after random ordering of subject (issue 151)
    pt_sort = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df_unb.sample(frac=1, replace=False, random_state=42),
        padjust="bonf",
    )
    assert pt_sort.round(5).equals(pt1.round(5))

    # ..No marginal means
    pt2 = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df_unb,
        padjust="bonf",
        marginal=False,
    )

    # This only impacts the between-subject contrast
    assert np.array_equal(
        (pt1["T"] == pt2["T"]).astype(int),
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    assert (pt1.loc[6:8, "dof"] < pt2.loc[6:8, "dof"]).all()

    # Without the Welch correction, check that all the DF are integer
    pt3 = pairwise_tests(
        dv="Scores",
        within="Time",
        between="Group",
        subject="Subject",
        data=df_unb,
        correction=False,
    )
    assert _is_integer(pt3["dof"]).all()


def test_pairwise_tests_factorial(df_aov2):
    """Test function pairwise_tests with two between-subject factors."""
    pt = df_aov2.pairwise_tests(dv="Yield", between=["Blend", "Crop"], padjust="holm").round(3)

    # The T and p-values are close but not exactly the same as JASP /
    # JAMOVI, because they both pool the error term.
    # The dof are not available in JASP, but in JAMOVI they are 18
    # everywhere, which I'm not sure to understand why...
    assert np.array_equal(pt.loc[:3, "p_unc"] < 0.05, [False, False, False, True])

    # However, the Bayes Factor of the simple main effects are the same...!
    assert np.array_equal(pt.loc[:3, "BF10"].astype(float), [0.374, 0.533, 0.711, 2.287])

    # Using the Welch method (all df should be non-integer)
    pt_c = df_aov2.pairwise_tests(
        dv="Yield", between=["Blend", "Crop"], padjust="holm", correction=True
    )
    assert not _is_integer(pt_c["dof"]).any()

    # The ``marginal`` option has no impact here.
    assert pt.equals(
        df_aov2.pairwise_tests(
            dv="Yield", between=["Blend", "Crop"], padjust="holm", marginal=True
        ).round(3)
    )


def test_pairwise_tests_two_within(df_rm2):
    """Test function pairwise_tests with two within-subject factors."""
    # .Marginal = True
    ptw1 = pairwise_tests(
        data=df_rm2,
        dv="Performance",
        within=["Time", "Metric"],
        subject="Subject",
        padjust="bonf",
        marginal=True,
    ).round(3)
    # Compare the T values of the simple main effect against JASP
    # Note that the T-values of the interaction are slightly different
    # because JASP pool the error term.
    assert np.array_equal(ptw1.loc[0:3, "T"], [5.818, 1.559, 7.714, 5.110])

    # Random sorting of the dataframe (issue 151)
    pt_sort = pairwise_tests(
        data=df_rm2.sample(frac=1, random_state=42),
        dv="Performance",
        within=["Time", "Metric"],
        subject="Subject",
        padjust="bonf",
        marginal=True,
    ).round(3)
    assert pt_sort.equals(ptw1)

    # Non-parametric: same contrasts, with a Wilcoxon test on each pair
    ptw_np = pairwise_tests(
        data=df_rm2,
        dv="Performance",
        within=["Time", "Metric"],
        subject="Subject",
        parametric=False,
    )
    assert ptw_np[["Contrast", "A", "B"]].equals(ptw1[["Contrast", "A", "B"]])
    assert ptw_np["W_val"].notna().all()
    assert ptw_np["p_unc"].between(0, 1).all()


def test_pairwise_tests_errors(df):
    """Test the errors of function pairwise_tests."""
    # Both multiple between and multiple within
    with pytest.raises(ValueError):
        pairwise_tests(
            dv="Scores",
            between=["Time", "Group"],
            within=["Time", "Group"],
            subject="Subject",
            data=df,
        )

    # More than two between or within factors
    with pytest.raises(ValueError):
        pairwise_tests(dv="Scores", between=["Time", "Group", "Subject"], data=df)
    with pytest.raises(ValueError):
        pairwise_tests(dv="Scores", within=["Time", "Group", "Subject"], subject="Subject", data=df)

    # Wrong input argument: a between factor with a single level
    df_one = df.assign(Group="Control")
    with pytest.raises(ValueError):
        pairwise_tests(dv="Scores", between="Group", data=df_one)


def test_pairwise_tests_missing():
    """Test function pairwise_tests with missing values in repeated measurements."""
    df = read_dataset("pairwise_tests_missing")
    # 1. Parametric
    st = pairwise_tests(
        dv="Value", within="Condition", subject="Subject", data=df, nan_policy="listwise"
    )
    np.testing.assert_array_equal(st["dof"].to_numpy(), [7, 7, 7])
    st2 = pairwise_tests(
        dv="Value", within="Condition", data=df, subject="Subject", nan_policy="pairwise"
    )
    np.testing.assert_array_equal(st2["dof"].to_numpy(), [8, 7, 8])
    # 2. Non-parametric
    st = pairwise_tests(
        dv="Value",
        within="Condition",
        subject="Subject",
        data=df,
        parametric=False,
        nan_policy="listwise",
    )
    np.testing.assert_array_equal(st["W_val"].to_numpy(), [9, 3, 12])
    st2 = pairwise_tests(
        dv="Value",
        within="Condition",
        data=df,
        subject="Subject",
        nan_policy="pairwise",
        parametric=False,
    )
    # Tested against a simple for loop on combinations
    np.testing.assert_array_equal(st2["W_val"].to_numpy(), [9, 3, 21])

    # Two within factors from other datasets and with NaN values
    df2 = read_dataset("rm_anova")
    pt = pairwise_tests(
        dv="DesireToKill",
        within=["Disgustingness", "Frighteningness"],
        subject="Subject",
        padjust="holm",
        data=df2,
    )
    # The subjects with a missing value are removed (listwise): 88 complete subjects
    np.testing.assert_array_equal(pt["dof"], [87, 87, 87, 87])
    assert pt["Contrast"].tolist() == [
        "Disgustingness",
        "Frighteningness",
        "Disgustingness * Frighteningness",
        "Disgustingness * Frighteningness",
    ]
    # Only the interaction has more than one test to correct for
    assert pt.loc[:1, "p_corr"].isna().all()
    np.testing.assert_allclose(
        pt.loc[2:, "p_corr"], multicomp(pt.loc[2:, "p_unc"], method="holm")[1]
    )


def test_pairwise_tests_alternative(df_drug):
    """Test the alternative and parametric arguments of pairwise_tests (compare with JASP)."""
    df = df_drug
    # 1. Within
    # 1.1 Parametric
    # 1.1.1 Tail is greater
    pt = pairwise_tests(
        dv="Scores", within="Drug", subject="Subject", data=df, alternative="greater"
    )
    np.testing.assert_array_equal(pt.loc[:, "p_unc"].round(3), [0.907, 0.941, 0.405])
    # 1.1.2 Tail is less
    pt = pairwise_tests(dv="Scores", within="Drug", subject="Subject", data=df, alternative="less")
    np.testing.assert_array_equal(pt.loc[:, "p_unc"].round(3), [0.093, 0.059, 0.595])

    # 1.2 Non-parametric
    # The exact p-values are slightly different across SciPy versions (e.g. 1.13 versus 1.15),
    # hence the tolerance against JASP. They must however be the same as wilcoxon.
    wide = df.pivot(index="Subject", columns="Drug", values="Scores")
    jasp = {"greater": [0.910, 0.951, 0.483], "less": [0.108, 0.060, 0.551]}
    for alternative, jasp_pval in jasp.items():
        pt = pairwise_tests(
            dv="Scores",
            within="Drug",
            subject="Subject",
            parametric=False,
            data=df,
            alternative=alternative,
        )
        np.testing.assert_allclose(pt["p_unc"], jasp_pval, atol=0.025)
        for _, row in pt.iterrows():
            wx = wilcoxon(wide[row["A"]], wide[row["B"]], alternative=alternative)
            assert row["p_unc"] == pytest.approx(wx.at["Wilcoxon", "p_val"])

    # Compare the RBC value for wilcoxon
    x = df[df["Drug"] == "A"]["Scores"].to_numpy()
    y = df[df["Drug"] == "B"]["Scores"].to_numpy()
    assert -0.6 < wilcoxon(x, y).at["Wilcoxon", "RBC"] < -0.4
    x = df[df["Drug"] == "B"]["Scores"].to_numpy()
    y = df[df["Drug"] == "C"]["Scores"].to_numpy()
    assert wilcoxon(x, y).at["Wilcoxon", "RBC"].round(3) == 0.030

    # 2. Between
    # 2.1 Parametric
    # 2.1.1 Tail is greater
    pt = pairwise_tests(dv="Scores", between="Gender", data=df, alternative="greater")
    assert pt.loc[0, "p_unc"].round(3) == 0.932
    # 2.1.2 Tail is less
    pt = pairwise_tests(dv="Scores", between="Gender", data=df, alternative="less")
    assert pt.loc[0, "p_unc"].round(3) == 0.068

    # 2.2 Non-parametric
    # 2.2.1 Tail is greater
    pt = pairwise_tests(
        dv="Scores", between="Gender", parametric=False, data=df, alternative="greater"
    )
    assert pt.loc[0, "p_unc"].round(3) == 0.901
    # 2.2.2 Tail is less
    pt = pairwise_tests(
        dv="Scores", between="Gender", parametric=False, data=df, alternative="less"
    )
    assert pt.loc[0, "p_unc"].round(3) == 0.105

    # Compare the RBC value for MWU
    x = df[df["Gender"] == "M"]["Scores"].to_numpy()
    y = df[df["Gender"] == "F"]["Scores"].to_numpy()
    assert round(abs(mwu(x, y).at["MWU", "RBC"]), 3) == 0.252


def test_ptests(df_bfi):
    """Test function ptests."""
    df = df_bfi.iloc[:30].copy()
    df.columns = ["N", "E", "O", "A", "C"]
    # Add some missing values
    df.iloc[[2, 5, 20], 2] = np.nan
    df.iloc[[1, 4, 10], 3] = np.nan

    combs = list(combinations(df.columns, 2))

    # Independent T-test
    pt = df.ptests(decimals=7, stars=False)
    for a, b in combs:
        t, p = ttest_ind(df[a], df[b], nan_policy="omit")
        assert round(t, 7) == float(pt.at[b, a])
        assert round(p, 7) == float(pt.at[a, b])

    # Using custom parameter
    pt_welch = df.ptests(decimals=5, stars=False, equal_var=False)
    assert not pt.equals(pt_welch)

    # Using stars
    pt_stars = df.ptests(decimals=7)
    assert not pt.equals(pt_stars)

    # Using custom significance thresholds
    pt_custom = df.ptests(pval_stars={0.05: "sig"}).to_numpy()
    upper = pt_custom[np.triu_indices(df.shape[1], k=1)]
    assert set(upper) <= {"", "sig"}
    assert "sig" in upper

    # Paired T-test
    pt = df.ptests(decimals=7, paired=True, stars=False)
    for a, b in combs:
        t, p = ttest_rel(df[a], df[b], nan_policy="omit")
        assert round(t, 7) == float(pt.at[b, a])
        assert round(p, 7) == float(pt.at[a, b])

    # With Bonferroni correction
    pt = df.ptests(decimals=7, paired=True, stars=False, padjust="bonf")
    for a, b in combs:
        _, p = ttest_rel(df[a], df[b], nan_policy="omit")
        p = min(p * len(combs), 1)
        assert round(p, 7) == float(pt.at[a, b])

    # Other correction methods must be honored, not silently replaced by Bonferroni
    pvals = np.array([ttest_rel(df[a], df[b], nan_policy="omit")[1] for a, b in combs])
    for padjust in ["holm", "sidak", "fdr_bh", "fdr_by"]:
        pt = df.ptests(decimals=7, paired=True, stars=False, padjust=padjust)
        actual = np.array([float(pt.at[a, b]) for a, b in combs])
        np.testing.assert_allclose(actual, multicomp(pvals, method=padjust)[1], atol=1e-7)
    pt_holm = df.ptests(paired=True, stars=False, padjust="holm")
    pt_bonf = df.ptests(paired=True, stars=False, padjust="bonf")
    assert not pt_holm.equals(pt_bonf)
    with pytest.raises(ValueError):
        df.ptests(padjust="wrong")
    # axis and nan_policy are fixed by ptests and cannot be passed to scipy
    with pytest.raises(ValueError):
        df.ptests(axis=1)
    with pytest.raises(ValueError):
        df.ptests(nan_policy="raise")


def test_pairwise_tukey(df_hair, df_penguins):
    """Test function pairwise_tukey.

    The p-values are slightly different because of a different algorithm
    used to calculate the studentized range approximation, but
    significance should be the same.
    """
    # Compare with R package `userfriendlyscience` - Hair color dataset
    # Update Feb 2021: The userfriendlyscience package has been removed
    # from CRAN.
    stats = pairwise_tukey(dv="Pain threshold", between="Hair color", data=df_hair)
    # JASP: [0.0741, 0.4356, 0.4147, 0.0037, 0.7893, 0.0366]
    # Pingouin: [0.0742, 0.4369, 0.4160, 0.0037, 0.7697, 0.0367]
    assert np.allclose(
        [0.074, 0.435, 0.415, 0.004, 0.789, 0.037],
        stats.loc[:, "p_tukey"].to_numpy().round(3),
        atol=0.05,
    )
    # Compare with JASP in the Palmer Penguins dataset
    # The between factor (Species) is unbalanced.
    df = df_penguins
    stats = df.pairwise_tukey(dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["A"], ["Adelie", "Adelie", "Chinstrap"])
    assert np.array_equal(stats["B"], ["Chinstrap", "Gentoo", "Gentoo"])
    assert np.array_equal(stats["diff"], [-32.426, -1375.354, -1342.928])
    # SE is different for each group (Tukey-Kramer)
    assert np.array_equal(stats["se"], [67.5117, 56.1480, 69.8569])
    assert np.array_equal(stats["T"], [-0.4803, -24.4952, -19.2240])
    # P-values JASP: [0.8807, 0.0000, 0.0000]
    # P-values Pingouin: [0.8694, 0.0010, 0.0010]
    sig = stats["p_tukey"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
    assert np.array_equal(sig, ["No", "Yes", "Yes"])
    # Effect size should be the same as pairwise_tests
    stats_tests = df.pairwise_tests(dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["hedges"], stats_tests["hedges"])

    # Same but with balanced group
    df_balanced = df.groupby("species").head(20).copy()
    # To complicate things, let's encode between as a categorical
    df_balanced["species"] = df_balanced["species"].astype("category")
    stats = df_balanced.pairwise_tukey(dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["A"], ["Adelie", "Adelie", "Chinstrap"])
    assert np.array_equal(stats["B"], ["Chinstrap", "Gentoo", "Gentoo"])
    assert np.array_equal(stats["diff"], [-142.5, -1457.5, -1315.0])
    # SE is the same for all groups (Tukey HSD)
    assert np.array_equal(stats["se"], [142.9475, 142.9475, 142.9475])
    assert np.array_equal(stats["T"], [-0.9969, -10.1961, -9.1992])
    # P-values JASP: [0.5818, 0.0000, 0.0000]
    # P-values Pingouin: [0.5766, 0.0010, 0.0010]
    sig = stats["p_tukey"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
    assert np.array_equal(sig, ["No", "Yes", "Yes"])

    # Interaction of two factors: all the pairs of species x sex cells. Compare with R:
    # TukeyHSD(aov(body_mass_g ~ species * sex, data=df), which="species:sex")
    stats = df.pairwise_tukey(dv="body_mass_g", between=["species", "sex"])
    assert stats.shape[0] == 15
    assert stats.at[0, "A"] == ("Adelie", "female")
    assert stats.at[0, "B"] == ("Adelie", "male")
    # R: Adelie:male-Adelie:female, Chinstrap:female-Adelie:female, Chinstrap:male-Adelie:male,
    # Chinstrap:male-Chinstrap:female
    idx = [0, 1, 6, 9]
    np.testing.assert_allclose(
        stats.loc[idx, "diff"], [-674.6575342, -158.3702659, 104.5225624, -411.7647059]
    )
    np.testing.assert_allclose(
        stats.loc[idx, "p_tukey"], [0, 0.1376213087, 0.5812048336, 0.0000012196], atol=1e-8
    )
    # Same as a single factor with one level per cell
    df_cells = df.dropna(subset=["species", "sex"]).copy()
    df_cells["cell"] = df_cells["species"] + "_" + df_cells["sex"]
    stats_cells = df_cells.pairwise_tukey(dv="body_mass_g", between="cell")
    num = ["mean_A", "mean_B", "diff", "se", "T", "p_tukey", "hedges"]
    np.testing.assert_allclose(stats[num], stats_cells[num])
    # A list with a single factor is the same as a string
    stats = df.pairwise_tukey(dv="body_mass_g", between=["species"])
    assert stats.equals(df.pairwise_tukey(dv="body_mass_g", between="species"))


def test_pairwise_gameshowell(df_hair, df_penguins):
    """Test function pairwise_gameshowell.

    The p-values are slightly different because of a different algorithm
    used to calculate the studentized range approximation, but
    significance should be the same.
    """
    # Compare with R package `userfriendlyscience` - Hair color dataset
    # Update Feb 2021: The userfriendlyscience package has been removed
    # from CRAN.
    stats = pairwise_gameshowell(dv="Pain threshold", between="Hair color", data=df_hair)
    assert np.array_equal(np.abs(stats["T"].round(2)), [2.47, 1.42, 1.75, 4.09, 1.11, 3.56])
    assert np.array_equal(stats["df"].round(2), [7.91, 7.94, 6.56, 8.0, 6.82, 6.77])
    # JASP: [0.1401, 0.5228, 0.3715, 0.0148, 0.6980, 0.0378]
    # Pingouin: [0.1401, 0.5220, 0.3722, 0.0148, 0.6848, 0.0378]
    assert np.allclose(
        [0.1401, 0.5228, 0.3715, 0.0148, 0.6980, 0.0378],
        stats.loc[:, "pval"].to_numpy().round(3),
        atol=0.05,
    )
    sig = stats["pval"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
    assert np.array_equal(sig, ["No", "No", "No", "Yes", "No", "Yes"])
    # Compare with JASP in the Palmer Penguins dataset
    df = df_penguins
    stats = pairwise_gameshowell(data=df, dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["A"], ["Adelie", "Adelie", "Chinstrap"])
    assert np.array_equal(stats["B"], ["Chinstrap", "Gentoo", "Gentoo"])
    assert np.array_equal(stats["diff"], [-32.426, -1375.354, -1342.928])
    assert np.array_equal(stats["se"], [59.7064, 58.8109, 65.1028])
    assert np.array_equal(stats["df"], [152.4548, 249.6426, 170.4044])
    assert np.array_equal(stats["T"], [-0.5431, -23.3860, -20.6278])
    # P-values JASP: [0.8502, 0.0000, 0.0000]
    # P-values Pingouin: [0.8339, 0.0010, 0.0010]
    sig = stats["pval"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
    assert np.array_equal(sig, ["No", "Yes", "Yes"])
    # Effect size should be the same as pairwise_tests
    stats_tests = df.pairwise_tests(dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["hedges"], stats_tests["hedges"])

    # Same but with balanced group
    df_balanced = df.groupby("species").head(20).copy()
    # To complicate things, let's encode between as a categorical
    df_balanced["species"] = df_balanced["species"].astype("category")
    stats = pairwise_gameshowell(data=df_balanced, dv="body_mass_g", between="species").round(4)
    assert np.array_equal(stats["A"], ["Adelie", "Adelie", "Chinstrap"])
    assert np.array_equal(stats["B"], ["Chinstrap", "Gentoo", "Gentoo"])
    assert np.array_equal(stats["diff"], [-142.5, -1457.5, -1315.0])
    assert np.array_equal(stats["se"], [104.5589, 163.1546, 154.1104])
    assert np.array_equal(stats["df"], [35.5510, 30.8479, 26.4576])
    assert np.array_equal(stats["T"], [-1.3629, -8.9332, -8.5328])
    # P-values JASP: [0.3709, 0.0000, 0.0000]
    # P-values Pingouin: [0.3719, 0.0010, 0.0010]
    sig = stats["pval"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
    assert np.array_equal(sig, ["No", "Yes", "Yes"])

    # Interaction of two factors: same as a single factor with one level per cell
    stats = pairwise_gameshowell(data=df, dv="body_mass_g", between=["species", "sex"])
    assert stats.at[0, "A"] == ("Adelie", "female")
    df_cells = df.dropna(subset=["species", "sex"]).copy()
    df_cells["cell"] = df_cells["species"] + "_" + df_cells["sex"]
    stats_cells = pairwise_gameshowell(data=df_cells, dv="body_mass_g", between="cell")
    num = ["mean_A", "mean_B", "diff", "se", "T", "df", "pval", "hedges"]
    np.testing.assert_allclose(stats[num], stats_cells[num])


def _pairs(stats):
    """List of the (X, Y) pairs of a pairwise_corr output."""
    return list(zip(stats["X"], stats["Y"], strict=True))


def test_pairwise_corr(df_bfi):
    """Test function pairwise_corr"""
    data = df_bfi.copy()
    big5 = data.columns.tolist()
    # Compare with JASP
    stats = pairwise_corr(data=data, method="pearson", alternative="two-sided")
    jasp_rval = [-0.350, -0.01, -0.134, -0.368, 0.267, 0.055, 0.065, 0.159, -0.013, 0.159]
    assert np.allclose(stats["r"].round(3).to_numpy(), jasp_rval)
    assert stats["n"].to_numpy()[0] == 500
    assert _pairs(stats) == list(combinations(big5, 2))
    # Correct for multiple comparisons
    stats = pairwise_corr(data=data, method="spearman", alternative="greater", padjust="bonf")
    assert (stats["method"] == "spearman").all()
    sp = corr(data["Neuroticism"], data["Extraversion"], method="spearman", alternative="greater")
    assert stats.at[0, "r"] == pytest.approx(sp.at["spearman", "r"])
    np.testing.assert_allclose(stats["p_corr"], np.minimum(stats["p_unc"] * 10, 1))
    # Check with a subset of columns
    stats = pairwise_corr(data=data, columns=["Neuroticism", "Extraversion"])
    assert _pairs(stats) == [("Neuroticism", "Extraversion")]
    assert stats.at[0, "r"] == pytest.approx(-0.350, abs=5e-4)  # JASP
    with pytest.raises(ValueError):
        pairwise_corr(data=data, columns="wrong")
    # Non-numeric columns are ignored
    data["test"] = "test"
    stats = pairwise_corr(data=data, method="pearson")
    assert _pairs(stats) == list(combinations(big5, 2))
    # Check different variation of product / combination
    rng = np.random.default_rng(42)
    n = data.shape[0]
    data["Age"] = rng.integers(18, 65, n)
    data["IQ"] = rng.normal(105, 1, n)
    data["One"] = 1
    data["Gender"] = np.repeat(["M", "F"], int(n / 2))
    numeric = big5 + ["Age", "IQ"]  # "One" is constant, so it is ignored
    others = [c for c in numeric if c != "Neuroticism"]
    neuro_vs_all = [("Neuroticism", c) for c in others]
    stats = pairwise_corr(data, columns=["Neuroticism", "Gender"], method="shepherd")
    assert _pairs(stats) == neuro_vs_all
    assert (stats["method"] == "shepherd").all() and "outliers" in stats.columns
    stats = pairwise_corr(data, columns=["Neuroticism", "Extraversion", "Gender"])
    assert _pairs(stats) == [("Neuroticism", "Extraversion")]
    stats = pairwise_corr(data, columns=["Neuroticism"])
    assert _pairs(stats) == neuro_vs_all
    with pytest.warns(UserWarning, match="skipped correlation relies"):
        stats = pairwise_corr(data, columns="Neuroticism", method="skipped")
    assert _pairs(stats) == neuro_vs_all
    assert (stats["method"] == "skipped").all() and "outliers" in stats.columns
    stats = pairwise_corr(data, columns=[["Neuroticism"]], method="spearman")
    assert _pairs(stats) == neuro_vs_all
    stats_none = pairwise_corr(data, columns=[["Neuroticism"], None], method="percbend")
    assert _pairs(stats_none) == neuro_vs_all
    assert (stats_none["method"] == "percbend").all()
    stats = pairwise_corr(data, columns=[["Neuroticism", "Gender"], ["Age"]])
    assert _pairs(stats) == [("Neuroticism", "Age")]
    stats = pairwise_corr(data, columns=[["Neuroticism"], ["Age", "IQ"]])
    assert _pairs(stats) == [("Neuroticism", "Age"), ("Neuroticism", "IQ")]
    stats = pairwise_corr(data, columns=[["Age", "IQ"], []])
    assert _pairs(stats) == [(x, y) for x in ["Age", "IQ"] for y in big5]
    # Missing columns are ignored
    stats = pairwise_corr(data, columns=["Age", "Gender", "IQ", "Wrong"])
    assert _pairs(stats) == [("Age", "IQ")]
    stats = pairwise_corr(data, columns=["Age", "Gender", "Wrong"])
    assert _pairs(stats) == [("Age", c) for c in numeric if c != "Age"]
    # Test with no good combinations
    with pytest.raises(ValueError):
        pairwise_corr(data, columns=["Gender", "Gender"])
    # Test when one column has only one unique value
    stats = pairwise_corr(data=data, columns=["Age", "One", "Gender"])
    assert _pairs(stats) == [("Age", c) for c in numeric if c != "Age"]
    stats = pairwise_corr(data, columns=["Neuroticism", "IQ", "One"])
    assert stats.shape[0] == 1
    # Test with covariate: same as partial_corr
    stats = pairwise_corr(data, covar="Age")
    assert _pairs(stats) == list(combinations([c for c in numeric if c != "Age"], 2))
    pc = partial_corr(data, x="Neuroticism", y="Extraversion", covar="Age")
    assert stats.at[0, "r"] == pytest.approx(pc.at["pearson", "r"])
    stats = pairwise_corr(data, covar=["Age", "Neuroticism"])
    assert _pairs(stats) == list(combinations(others[:-2] + ["IQ"], 2))
    pc = partial_corr(data, x="Extraversion", y="Openness", covar=["Age", "Neuroticism"])
    assert stats.at[0, "r"] == pytest.approx(pc.at["pearson", "r"])
    pd.testing.assert_frame_equal(
        pairwise_corr(data, covar=pd.Index(["Age", "Neuroticism"])),
        stats,
    )
    with pytest.raises(AssertionError):
        pairwise_corr(data, covar=["Age", "Gender"])
    with pytest.raises(ValueError):
        pairwise_corr(data, columns=["Neuroticism", "Age"], covar="Age")
    # Partial pairwise with missing values
    data.loc[[4, 5, 8, 20, 22], "Age"] = np.nan
    data.loc[[10, 12], "Neuroticism"] = np.nan
    stats = pairwise_corr(data)
    n_obs = dict(zip(_pairs(stats), stats["n"], strict=True))
    assert n_obs[("Neuroticism", "Extraversion")] == 498
    assert n_obs[("Neuroticism", "Age")] == 493
    assert n_obs[("Extraversion", "Openness")] == 500
    stats = pairwise_corr(data, covar="Age")
    # Each pair is computed on the rows with no missing value in the pair or the covariate
    n_obs = dict(zip(_pairs(stats), stats["n"], strict=True))
    assert n_obs[("Neuroticism", "Extraversion")] == 493
    assert n_obs[("Extraversion", "Openness")] == 495
    # Listwise deletion
    assert pairwise_corr(data, covar="Age", nan_policy="listwise")["n"].nunique() == 1
    assert pairwise_corr(data, nan_policy="listwise")["n"].nunique() == 1


def test_pairwise_corr_multiindex():
    """Test function pairwise_corr with MultiIndex columns."""
    rng = np.random.default_rng(42)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Behavior", "Rating"),
            ("Behavior", "RT"),
            ("Physio", "BOLD"),
            ("Physio", "HR"),
            ("Psycho", "Anxiety"),
        ]
    )
    data = pd.DataFrame(rng.random(size=(10, 5)), columns=columns)
    stats = pairwise_corr(data, method="spearman")
    assert _pairs(stats) == list(combinations(columns, 2))
    stats = pairwise_corr(data, columns=[("Behavior", "Rating")])
    assert stats.shape[0] == data.shape[1] - 1
    stats = pairwise_corr(data, columns=[("Behavior", "Rating"), ("Behavior", "RT")])
    assert _pairs(stats) == [(("Behavior", "Rating"), ("Behavior", "RT"))]
    st1 = pairwise_corr(data, columns=[[("Behavior", "Rating"), ("Behavior", "RT")], None])
    st2 = pairwise_corr(data, columns=[[("Behavior", "Rating"), ("Behavior", "RT")]])
    assert st1["X"].equals(st2["X"])
    st3 = pairwise_corr(
        data, columns=[[("Behavior", "Rating")], [("Behavior", "RT"), ("Physio", "BOLD")]]
    )
    assert st3.shape[0] == 2
    # With covar
    stats = pairwise_corr(data, covar=[("Psycho", "Anxiety")])
    assert _pairs(stats) == list(combinations(columns[:4], 2))
    stats = pairwise_corr(data, columns=[("Behavior", "Rating")], covar=[("Psycho", "Anxiety")])
    assert _pairs(stats) == [(("Behavior", "Rating"), c) for c in columns[1:4]]
    # With missing values
    data.iloc[2, [2, 3]] = np.nan
    data.iloc[[1, 4], [1, 4]] = np.nan
    assert pairwise_corr(data, nan_policy="listwise")["n"].nunique() == 1
    assert pairwise_corr(data, nan_policy="pairwise")["n"].nunique() == 3
    assert (
        pairwise_corr(
            data,
            columns=[("Behavior", "Rating")],
            covar=[("Psycho", "Anxiety")],
            nan_policy="listwise",
        )["n"].nunique()
        == 1
    )
    assert (
        pairwise_corr(
            data,
            columns=[("Behavior", "Rating")],
            covar=[("Psycho", "Anxiety")],
            nan_policy="pairwise",
        )["n"].nunique()
        == 2
    )
