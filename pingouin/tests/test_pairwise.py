import pytest
import numpy as np
import pandas as pd
from unittest import TestCase
from pingouin import read_dataset
from pingouin.pairwise import (
    pairwise_ttests,
    pairwise_tests,
    pairwise_corr,
    pairwise_tukey,
    pairwise_gameshowell,
)


class TestPairwise(TestCase):
    """Test pairwise.py."""

    def test_pairwise_tests(self):
        """Test function pairwise_tests.

        Tested against the pairwise.t.test R function, as well as JASP and
        JAMOVI.

        Notes:
        1) JAMOVI by default pool the error term for the within-subject
        factor in mixed design. Pingouin does not pool the error term,
        which is the same behavior as JASP.

        2) JASP does not return the uncorrected p-values, therefore only the
        corrected p-values are compared.

        3) JASP does not calculate the Bayes Factor for the interaction terms.
        For mixed design and two-way design, in JASP, the Bayes Factor
        seems to be calculated without aggregating over repeated measurements.

        4) For factorial between-subject contrasts, both JASP and JAMOVI pool
        the error term. This option is not yet implemented in Pingouin.
        Therefore, one cannot directly validate the T and p-values.
        """
        df = read_dataset("mixed_anova.csv")  # Simple and mixed design
        df_sort = df.sort_values("Time")  # Same but random order of subject
        # df_sort = df.sample(frac=1)
        df_unb = read_dataset("mixed_anova_unbalanced")
        df_rm2 = read_dataset("rm_anova2")  # 2-way rm design
        df_aov2 = read_dataset("anova2")  # 2-way factorial design

        # -------------------------------------------------------------------
        # Warning for deprecated pairwise_ttests()
        # -------------------------------------------------------------------
        with pytest.warns(UserWarning):
            pt = pairwise_ttests(
                dv="Scores",
                within="Time",
                subject="Subject",
                data=df,
                return_desc=True,
                padjust="holm",
            )
            np.testing.assert_array_equal(pt.loc[:, "p-corr"].round(3), [0.174, 0.024, 0.310])
            np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.087, 0.008, 0.310])

        # -------------------------------------------------------------------
        # Simple within: EASY!
        # -------------------------------------------------------------------
        # In R:
        # >>> pairwise.t.test(df$Scores, df$Time, pool.sd = FALSE,
        # ...                 p.adjust.method = 'holm', paired = TRUE)
        pt = pairwise_tests(
            dv="Scores", within="Time", subject="Subject", data=df, return_desc=True, padjust="holm"
        )
        np.testing.assert_array_equal(pt.loc[:, "p-corr"].round(3), [0.174, 0.024, 0.310])
        np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.087, 0.008, 0.310])
        pairwise_tests(
            dv="Scores",
            within="Time",
            subject="Subject",
            data=df,
            parametric=False,
            return_desc=True,
        )

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

        # -------------------------------------------------------------------
        # Simple between: EASY!
        # -------------------------------------------------------------------
        # In R: >>> pairwise.t.test(df$Scores, df$Group, pool.sd = FALSE)
        pt = pairwise_tests(dv="Scores", between="Group", data=df).round(3)
        assert pt.loc[0, "p-unc"] == 0.023
        pairwise_tests(
            dv="Scores",
            between="Group",
            data=df,
            padjust="bonf",
            alternative="greater",
            effsize="cohen",
            parametric=False,
        )

        # Same after random ordering of subject (issue 151)
        pt_sort = pairwise_tests(dv="Scores", between="Group", data=df_sort).round(3)
        assert pt_sort.equals(pt)

        # -------------------------------------------------------------------
        # Mixed design: Within + Between + Within * Between
        # -------------------------------------------------------------------
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
        assert np.array_equal(pt.loc[:2, "p-corr"].round(3), [0.174, 0.024, 0.310])
        assert np.array_equal(pt.loc[:2, "BF10"].astype(float), [0.582, 4.232, 0.232])
        # ..Between main effect: T and p-values OK with JASP
        #   but BF10 is only similar when marginal=False (see note in the
        #   2-way RM test below).
        assert pt.loc[3, "T"].round(3) == -2.248
        assert pt.loc[3, "p-unc"].round(3) == 0.028
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
        assert np.array_equal(pt_merged["p-unc"], pt["p-unc"].iloc[4:])
        # However adjusted p-values are not equal because they are calculated
        # separately on each dataframe.
        assert not np.array_equal(pt_merged["p-corr"], pt["p-corr"].iloc[4:])

        # Other options
        pairwise_tests(
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

        # .Unbalanced data
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
        assert np.array_equal(pt1.loc[:5, "p-corr"].round(3), [1.0, 1.0, 0.313, 1.0, 0.889, 1.0])
        assert np.array_equal(
            pt1.loc[:5, "BF10"].astype(float), [0.273, 0.463, 1.221, 0.280, 0.554, 0.248]
        )
        # ...Between main effect: slightly different from JASP (why?)
        #      True with or without the Welch correction...
        assert (pt1.loc[6:8, "p-corr"] > 0.20).all()
        # ...Interaction: slightly different because JASP pool the error term
        #    across the between-subject groups.
        # Below the interaction JASP bonferroni-correct p-values, which are
        # more conservative because JASP perform all possible pairwise tests
        # jasp_pbonf = [1., 1., 1., 1., 1., 1., 1., 0.886, 1., 1., 1., 1.]
        assert (pt1.loc[9:, "p-corr"] > 0.05).all()
        # Check that the Welch corection is applied by default
        assert not pt1["dof"].apply(lambda x: x.is_integer()).all()

        # Same after random ordering of subject (issue 151)
        pt_sort = pairwise_tests(
            dv="Scores",
            within="Time",
            between="Group",
            subject="Subject",
            data=df_unb.sample(frac=1, replace=False),
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
        np.array_equal(
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
        assert pt3["dof"].apply(lambda x: x.is_integer()).all()

        # -------------------------------------------------------------------
        # Two between factors (FACTORIAL)
        # -------------------------------------------------------------------
        pt = df_aov2.pairwise_tests(dv="Yield", between=["Blend", "Crop"], padjust="holm").round(3)

        # The T and p-values are close but not exactly the same as JASP /
        # JAMOVI, because they both pool the error term.
        # The dof are not available in JASP, but in JAMOVI they are 18
        # everywhere, which I'm not sure to understand why...
        assert np.array_equal(pt.loc[:3, "p-unc"] < 0.05, [False, False, False, True])

        # However, the Bayes Factor of the simple main effects are the same...!
        np.array_equal(pt.loc[:3, "BF10"].astype(float), [0.374, 0.533, 0.711, 2.287])

        # Using the Welch method (all df should be non-integer)
        pt_c = df_aov2.pairwise_tests(
            dv="Yield", between=["Blend", "Crop"], padjust="holm", correction=True
        )
        assert not pt_c["dof"].apply(lambda x: x.is_integer()).any()

        # The ``marginal`` option has no impact here.
        assert pt.equals(
            df_aov2.pairwise_tests(
                dv="Yield", between=["Blend", "Crop"], padjust="holm", marginal=True
            ).round(3)
        )
        # -------------------------------------------------------------------
        # Two within subject factors
        # -------------------------------------------------------------------
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
            data=df_rm2.sample(frac=1),
            dv="Performance",
            within=["Time", "Metric"],
            subject="Subject",
            padjust="bonf",
            marginal=True,
        ).round(3)
        assert pt_sort.equals(ptw1)

        # Non-parametric (mostly for code coverage)
        pairwise_tests(
            data=df_rm2,
            dv="Performance",
            within=["Time", "Metric"],
            subject="Subject",
            parametric=False,
        )

        # -------------------------------------------------------------------
        # ERRORS
        # -------------------------------------------------------------------
        # Both multiple between and multiple within
        with pytest.raises(ValueError):
            pairwise_tests(
                dv="Scores",
                between=["Time", "Group"],
                within=["Time", "Group"],
                subject="Subject",
                data=df,
            )

        # Wrong input argument
        df["Group"] = "Control"
        with pytest.raises(ValueError):
            pairwise_tests(dv="Scores", between="Group", data=df)

        # -------------------------------------------------------------------
        # Missing values in repeated measurements
        # -------------------------------------------------------------------
        # 1. Parametric
        df = read_dataset("pairwise_tests_missing")
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
        np.testing.assert_array_equal(st["W-val"].to_numpy(), [9, 3, 12])
        st2 = pairwise_tests(
            dv="Value",
            within="Condition",
            data=df,
            subject="Subject",
            nan_policy="pairwise",
            parametric=False,
        )
        # Tested against a simple for loop on combinations
        np.testing.assert_array_equal(st2["W-val"].to_numpy(), [9, 3, 21])

        # Two within factors from other datasets and with NaN values
        df2 = read_dataset("rm_anova")
        pairwise_tests(
            dv="DesireToKill",
            within=["Disgustingness", "Frighteningness"],
            subject="Subject",
            padjust="holm",
            data=df2,
        )

        # -------------------------------------------------------------------
        # Test tail / parametric argument (compare with JASP)
        # -------------------------------------------------------------------
        df = read_dataset("pairwise_tests")
        # 1. Within
        # 1.1 Parametric
        # 1.1.1 Tail is greater
        pt = pairwise_tests(
            dv="Scores", within="Drug", subject="Subject", data=df, alternative="greater"
        )
        np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.907, 0.941, 0.405])
        assert all(pt.loc[:, "BF10"].astype(float) < 1)
        # 1.1.2 Tail is less
        pt = pairwise_tests(
            dv="Scores", within="Drug", subject="Subject", data=df, alternative="less"
        )
        np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.093, 0.059, 0.595])
        assert sum(pt.loc[:, "BF10"].astype(float) > 1) == 2

        # 1.2 Non-parametric
        # 1.2.1 Tail is greater
        pt = pairwise_tests(
            dv="Scores",
            within="Drug",
            subject="Subject",
            parametric=False,
            data=df,
            alternative="greater",
        )
        np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.910, 0.951, 0.483])
        # 1.2.2 Tail is less
        pt = pairwise_tests(
            dv="Scores",
            within="Drug",
            subject="Subject",
            parametric=False,
            data=df,
            alternative="less",
        )
        np.testing.assert_array_equal(pt.loc[:, "p-unc"].round(3), [0.108, 0.060, 0.551])

        # Compare the RBC value for wilcoxon
        from pingouin.nonparametric import wilcoxon

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
        assert pt.loc[0, "p-unc"].round(3) == 0.932
        assert float(pt.loc[0, "BF10"]) < 1
        # 2.1.2 Tail is less
        pt = pairwise_tests(dv="Scores", between="Gender", data=df, alternative="less")
        assert pt.loc[0, "p-unc"].round(3) == 0.068
        assert float(pt.loc[0, "BF10"]) > 1

        # 2.2 Non-parametric
        # 2.2.1 Tail is greater
        pt = pairwise_tests(
            dv="Scores", between="Gender", parametric=False, data=df, alternative="greater"
        )
        assert pt.loc[0, "p-unc"].round(3) == 0.901
        # 2.2.2 Tail is less
        pt = pairwise_tests(
            dv="Scores", between="Gender", parametric=False, data=df, alternative="less"
        )
        assert pt.loc[0, "p-unc"].round(3) == 0.105

        # Compare the RBC value for MWU
        from pingouin.nonparametric import mwu

        x = df[df["Gender"] == "M"]["Scores"].to_numpy()
        y = df[df["Gender"] == "F"]["Scores"].to_numpy()
        assert round(abs(mwu(x, y).at["MWU", "RBC"]), 3) == 0.252

    def test_ptests(self):
        """Test function ptests."""
        from itertools import combinations
        from scipy.stats import ttest_ind, ttest_rel

        # Load BFI dataset
        df = read_dataset("pairwise_corr").iloc[:30, 1:]
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

    def test_pairwise_tukey(self):
        """Test function pairwise_tukey.

        The p-values are slightly different because of a different algorithm
        used to calculate the studentized range approximation, but
        significance should be the same.
        """
        # Compare with R package `userfriendlyscience` - Hair color dataset
        # Update Feb 2021: The userfriendlyscience package has been removed
        # from CRAN.
        df = read_dataset("anova")
        stats = pairwise_tukey(dv="Pain threshold", between="Hair color", data=df)
        # JASP: [0.0741, 0.4356, 0.4147, 0.0037, 0.7893, 0.0366]
        # Pingouin: [0.0742, 0.4369, 0.4160, 0.0037, 0.7697, 0.0367]
        assert np.allclose(
            [0.074, 0.435, 0.415, 0.004, 0.789, 0.037],
            stats.loc[:, "p-tukey"].to_numpy().round(3),
            atol=0.05,
        )
        # Compare with JASP in the Palmer Penguins dataset
        # The between factor (Species) is unbalanced.
        df = read_dataset("penguins")
        stats = df.pairwise_tukey(dv="body_mass_g", between="species").round(4)
        assert np.array_equal(stats["A"], ["Adelie", "Adelie", "Chinstrap"])
        assert np.array_equal(stats["B"], ["Chinstrap", "Gentoo", "Gentoo"])
        assert np.array_equal(stats["diff"], [-32.426, -1375.354, -1342.928])
        # SE is different for each group (Tukey-Kramer)
        assert np.array_equal(stats["se"], [67.5117, 56.1480, 69.8569])
        assert np.array_equal(stats["T"], [-0.4803, -24.4952, -19.2240])
        # P-values JASP: [0.8807, 0.0000, 0.0000]
        # P-values Pingouin: [0.8694, 0.0010, 0.0010]
        sig = stats["p-tukey"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
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
        sig = stats["p-tukey"].apply(lambda x: "Yes" if x < 0.05 else "No").to_numpy()
        assert np.array_equal(sig, ["No", "Yes", "Yes"])

    def test_pairwise_gameshowell(self):
        """Test function pairwise_gameshowell.

        The p-values are slightly different because of a different algorithm
        used to calculate the studentized range approximation, but
        significance should be the same.
        """
        # Compare with R package `userfriendlyscience` - Hair color dataset
        # Update Feb 2021: The userfriendlyscience package has been removed
        # from CRAN.
        df = read_dataset("anova")
        stats = pairwise_gameshowell(dv="Pain threshold", between="Hair color", data=df)
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
        df = read_dataset("penguins")
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

    def test_pairwise_corr(self):
        """Test function pairwise_corr"""
        # Load JASP Big 5 DataSets (remove subject column)
        data = read_dataset("pairwise_corr").iloc[:, 1:]
        stats = pairwise_corr(data=data, method="pearson", alternative="two-sided")
        jasp_rval = [-0.350, -0.01, -0.134, -0.368, 0.267, 0.055, 0.065, 0.159, -0.013, 0.159]
        assert np.allclose(stats["r"].round(3).to_numpy(), jasp_rval)
        assert stats["n"].to_numpy()[0] == 500
        # Correct for multiple comparisons
        pairwise_corr(data=data, method="spearman", alternative="greater", padjust="bonf")
        # Check with a subset of columns
        pairwise_corr(data=data, columns=["Neuroticism", "Extraversion"])
        with pytest.raises(ValueError):
            pairwise_corr(data=data, columns="wrong")
        # Check with non-numeric columns
        data["test"] = "test"
        pairwise_corr(data=data, method="pearson")
        # Check different variation of product / combination
        n = data.shape[0]
        data["Age"] = np.random.randint(18, 65, n)
        data["IQ"] = np.random.normal(105, 1, n)
        data["One"] = 1
        data["Gender"] = np.repeat(["M", "F"], int(n / 2))
        pairwise_corr(data, columns=["Neuroticism", "Gender"], method="shepherd")
        pairwise_corr(data, columns=["Neuroticism", "Extraversion", "Gender"])
        pairwise_corr(data, columns=["Neuroticism"])
        pairwise_corr(data, columns="Neuroticism", method="skipped")
        pairwise_corr(data, columns=[["Neuroticism"]], method="spearman")
        pairwise_corr(data, columns=[["Neuroticism"], None], method="percbend")
        pairwise_corr(data, columns=[["Neuroticism", "Gender"], ["Age"]])
        pairwise_corr(data, columns=[["Neuroticism"], ["Age", "IQ"]])
        pairwise_corr(data, columns=[["Age", "IQ"], []])
        pairwise_corr(data, columns=["Age", "Gender", "IQ", "Wrong"])
        pairwise_corr(data, columns=["Age", "Gender", "Wrong"])
        # Test with no good combinations
        with pytest.raises(ValueError):
            pairwise_corr(data, columns=["Gender", "Gender"])
        # Test when one column has only one unique value
        pairwise_corr(data=data, columns=["Age", "One", "Gender"])
        stats = pairwise_corr(data, columns=["Neuroticism", "IQ", "One"])
        assert stats.shape[0] == 1
        # Test with covariate
        pairwise_corr(data, covar="Age")
        pairwise_corr(data, covar=["Age", "Neuroticism"])
        with pytest.raises(AssertionError):
            pairwise_corr(data, covar=["Age", "Gender"])
        with pytest.raises(ValueError):
            pairwise_corr(data, columns=["Neuroticism", "Age"], covar="Age")
        # Partial pairwise with missing values
        data.loc[[4, 5, 8, 20, 22], "Age"] = np.nan
        data.loc[[10, 12], "Neuroticism"] = np.nan
        pairwise_corr(data)
        pairwise_corr(data, covar="Age")
        # Listwise deletion
        assert pairwise_corr(data, covar="Age", nan_policy="listwise")["n"].nunique() == 1
        assert pairwise_corr(data, nan_policy="listwise")["n"].nunique() == 1
        ######################################################################
        # MultiIndex columns
        from numpy.random import random as rdm

        # Create MultiIndex dataframe
        columns = pd.MultiIndex.from_tuples(
            [
                ("Behavior", "Rating"),
                ("Behavior", "RT"),
                ("Physio", "BOLD"),
                ("Physio", "HR"),
                ("Psycho", "Anxiety"),
            ]
        )
        data = pd.DataFrame(
            dict(
                Rating=rdm(size=10),
                RT=rdm(size=10),
                BOLD=rdm(size=10),
                HR=rdm(size=10),
                Anxiety=rdm(size=10),
            )
        )
        data.columns = columns
        pairwise_corr(data, method="spearman")
        stats = pairwise_corr(data, columns=[("Behavior", "Rating")])
        assert stats.shape[0] == data.shape[1] - 1
        pairwise_corr(data, columns=[("Behavior", "Rating"), ("Behavior", "RT")])
        st1 = pairwise_corr(data, columns=[[("Behavior", "Rating"), ("Behavior", "RT")], None])
        st2 = pairwise_corr(data, columns=[[("Behavior", "Rating"), ("Behavior", "RT")]])
        assert st1["X"].equals(st2["X"])
        st3 = pairwise_corr(
            data, columns=[[("Behavior", "Rating")], [("Behavior", "RT"), ("Physio", "BOLD")]]
        )
        assert st3.shape[0] == 2
        # With covar
        pairwise_corr(data, covar=[("Psycho", "Anxiety")])
        pairwise_corr(data, columns=[("Behavior", "Rating")], covar=[("Psycho", "Anxiety")])
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
