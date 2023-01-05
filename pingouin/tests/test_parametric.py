import pytest
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal as array_equal

from pingouin import read_dataset
from pingouin.parametric import ttest, anova, rm_anova, mixed_anova, ancova, welch_anova


# Generate random data for ANOVA
df = read_dataset("mixed_anova.csv")

# With missing values
df_nan = df.copy()
df_nan.iloc[[4, 15], 0] = np.nan

# With categorical!
df_cat = df.copy()
df_cat[["Time", "Group", "Subject"]] = df_cat[["Time", "Group", "Subject"]].astype("category")
# Let's complicate things even more and add "ghost" Categories
df_cat["Time"] = df_cat["Time"].cat.add_categories("Casper")
df_cat["Group"] = df_cat["Group"].cat.add_categories("The Friendly Ghost")

# Create random normal variables
np.random.seed(1234)
x = np.random.normal(scale=1.0, size=100)
y = np.random.normal(scale=0.8, size=100)


class TestParametric(TestCase):
    """Test parametric.py."""

    def test_ttest(self):
        """Test function ttest.
        Compare with Matlab, R and JASP.
        """
        # Test different combination of argments
        h = np.random.normal(scale=0.9, size=95)
        ttest(x, 0.5)
        ttest(x, y, paired=False, correction="auto")
        ttest(x, y, paired=False, correction=True)
        ttest(x, y, paired=False, r=0.5)
        ttest(x, h, paired=True)

        a = [4, 7, 8, 6, 3, 2]
        b = [6, 8, 7, 10, 11, 9]

        # 1) One sample with y=0
        # R: t.test(a, mu=0)
        # Two-sided
        tt = ttest(a, y=0, alternative="two-sided")
        assert round(tt.loc["T-test", "T"], 5) == 5.17549
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.00354
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [2.52, 7.48])
        # Using a different confidence level
        tt = ttest(a, y=0, alternative="two-sided", confidence=0.90)
        array_equal(np.round(tt.loc["T-test", "CI90%"], 3), [3.053, 6.947])

        # One-sided (greater)
        tt = ttest(a, y=0, alternative="greater")
        assert round(tt.loc["T-test", "T"], 5) == 5.17549
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.00177
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [3.05, np.inf])
        # One-sided (less)
        tt = ttest(a, y=0, alternative="less")
        assert round(tt.loc["T-test", "T"], 5) == 5.17549
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.99823
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-np.inf, 6.95])

        # 2) One sample with y=4
        # R: t.test(a, mu=4)
        # Two-sided
        tt = ttest(a, y=4, alternative="two-sided")
        assert round(tt.loc["T-test", "T"], 5) == 1.0351
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.34807
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [2.52, 7.48])
        # One-sided (greater)
        tt = ttest(a, y=4, alternative="greater")
        assert round(tt.loc["T-test", "T"], 5) == 1.0351
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.17403
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [3.05, np.inf])
        # One-sided (less)
        tt = ttest(a, y=4, alternative="less")
        assert round(tt.loc["T-test", "T"], 5) == 1.0351
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.82597
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-np.inf, 6.95])

        # 3) Paired two-sample
        # R: t.test(a, b, paired=TRUE)
        # Two-sided
        tt = ttest(a, b, paired=True, alternative="two-sided")
        assert round(tt.loc["T-test", "T"], 5) == -2.44451
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.05833
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-7.18, 0.18])
        # One-sided (greater)
        tt = ttest(a, b, paired=True, alternative="greater")
        assert round(tt.loc["T-test", "T"], 5) == -2.44451
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.97084
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-6.39, np.inf])
        # With a different confidence level
        tt = ttest(a, b, paired=True, alternative="greater", confidence=0.99)
        array_equal(np.round(tt.loc["T-test", "CI99%"], 3), [-8.318, np.inf])

        # One-sided (less)
        tt = ttest(a, b, paired=True, alternative="less")
        assert round(tt.loc["T-test", "T"], 5) == -2.44451
        assert tt.loc["T-test", "dof"] == 5
        assert round(tt.loc["T-test", "p-val"], 5) == 0.02916
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-np.inf, -0.61])

        # When the two arrays are identical
        tt = ttest(a, a, paired=True)
        assert str(tt.loc["T-test", "T"]) == str(np.nan)
        assert str(tt.loc["T-test", "p-val"]) == str(np.nan)
        assert tt.loc["T-test", "cohen-d"] == 0.0
        assert tt.loc["T-test", "BF10"] == str(np.nan)

        # 4) Independent two-samples, equal variance (no correction)
        # R: t.test(a, b, paired=FALSE, var.equal=TRUE)
        # Two-sided
        tt = ttest(a, b, correction=False, alternative="two-sided")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert tt.loc["T-test", "dof"] == 10
        assert round(tt.loc["T-test", "p-val"], 5) == 0.01749
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-6.24, -0.76])
        # One-sided (greater)
        tt = ttest(a, b, correction=False, alternative="greater")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert tt.loc["T-test", "dof"] == 10
        assert round(tt.loc["T-test", "p-val"], 5) == 0.99126
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-5.73, np.inf])
        # One-sided (less)
        tt = ttest(a, b, correction=False, alternative="less")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert tt.loc["T-test", "dof"] == 10
        assert round(tt.loc["T-test", "p-val"], 5) == 0.00874
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-np.inf, -1.27])

        # 5) Independent two-samples, Welch correction
        # R: t.test(a, b, paired=FALSE, var.equal=FALSE)
        # Two-sided
        tt = ttest(a, b, correction=True, alternative="two-sided")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert round(tt.loc["T-test", "dof"], 5) == 9.49438
        assert round(tt.loc["T-test", "p-val"], 5) == 0.01837
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-6.26, -0.74])
        # One-sided (greater)
        tt = ttest(a, b, correction=True, alternative="greater")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert round(tt.loc["T-test", "dof"], 5) == 9.49438
        assert round(tt.loc["T-test", "p-val"], 5) == 0.99082
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-5.74, np.inf])
        # One-sided (less)
        tt = ttest(a, b, correction=True, alternative="less")
        assert round(tt.loc["T-test", "T"], 5) == -2.84199
        assert round(tt.loc["T-test", "dof"], 5) == 9.49438
        assert round(tt.loc["T-test", "p-val"], 5) == 0.00918
        array_equal(np.round(tt.loc["T-test", "CI95%"], 2), [-np.inf, -1.26])

    def test_anova(self):
        """Test function anova.
        Compare results to JASP.
        """
        # Pain dataset
        df_pain = read_dataset("anova")
        anova(dv="Pain threshold", between=["Hair color"], data=df_pain)
        # Compare with JASP
        aov = anova(dv="Pain threshold", between="Hair color", data=df_pain, detailed=True).round(3)
        assert aov.at[0, "F"] == 6.791
        assert aov.at[0, "p-unc"] == 0.004
        assert aov.at[0, "np2"] == 0.576
        aov = anova(
            dv="Pain threshold", between="Hair color", data=df_pain, effsize="n2", detailed=True
        ).round(3)
        assert aov.at[0, "n2"] == 0.576

        # Unbalanced and with missing values
        df_pain.loc[[17, 18], "Pain threshold"] = np.nan
        aov = df_pain.anova(dv="Pain threshold", between="Hair color").round(3)
        assert aov.at[0, "ddof1"] == 3
        assert aov.at[0, "ddof2"] == 13
        assert aov.at[0, "F"] == 4.359
        assert aov.at[0, "p-unc"] == 0.025
        assert aov.at[0, "np2"] == 0.501
        # Error: between is an empty list
        with pytest.raises(ValueError):
            anova(dv="Pain threshold", between=[], data=df_pain)

        # Unbalanced and with missing values AND between as a categorical
        df_paincat = df_pain.copy()
        df_paincat["Hair color"] = df_paincat["Hair color"].astype("category")
        df_paincat["Hair color"] = df_paincat["Hair color"].cat.add_categories("Bald")
        aov = df_paincat.anova(dv="Pain threshold", between="Hair color").round(3)
        assert aov.at[0, "ddof1"] == 3
        assert aov.at[0, "ddof2"] == 13
        assert aov.at[0, "F"] == 4.359
        assert aov.at[0, "p-unc"] == 0.025
        assert aov.at[0, "np2"] == 0.501

        # Two-way ANOVA with balanced design
        df_aov2 = read_dataset("anova2")
        aov2 = anova(dv="Yield", between=["Blend", "Crop"], data=df_aov2).round(4)
        array_equal(aov2.loc[:, "MS"], [2.0417, 1368.2917, 1180.0417, 541.8472])
        array_equal(aov2.loc[[0, 1, 2], "F"], [0.0038, 2.5252, 2.1778])
        array_equal(aov2.loc[[0, 1, 2], "p-unc"], [0.9517, 0.1080, 0.1422])
        array_equal(aov2.loc[[0, 1, 2], "np2"], [0.0002, 0.2191, 0.1948])
        # Same but with standard eta-square
        aov2 = anova(dv="Yield", between=["Blend", "Crop"], data=df_aov2, effsize="n2").round(4)
        array_equal(aov2.loc[[0, 1, 2], "n2"], [0.0001, 0.1843, 0.1589])

        # Two-way ANOVA with unbalanced design
        df_aov2 = read_dataset("anova2_unbalanced")
        aov2 = df_aov2.anova(dv="Scores", between=["Diet", "Exercise"]).round(3)
        array_equal(aov2.loc[:, "MS"], [390.625, 180.625, 15.625, 52.625])
        array_equal(aov2.loc[[0, 1, 2], "F"], [7.423, 3.432, 0.297])
        array_equal(aov2.loc[[0, 1, 2], "p-unc"], [0.034, 0.113, 0.605])
        array_equal(aov2.loc[[0, 1, 2], "np2"], [0.553, 0.364, 0.047])

        # Two-way ANOVA with unbalanced design and missing values
        df_aov2.loc[9, "Scores"] = np.nan
        # Type 2
        aov2 = anova(dv="Scores", between=["Diet", "Exercise"], data=df_aov2).round(3)
        array_equal(aov2.loc[[0, 1, 2], "F"], [10.403, 5.167, 0.761])
        array_equal(aov2.loc[[0, 1, 2], "p-unc"], [0.023, 0.072, 0.423])
        array_equal(aov2.loc[[0, 1, 2], "np2"], [0.675, 0.508, 0.132])
        # Type 1
        aov2_ss1 = anova(dv="Scores", between=["Diet", "Exercise"], ss_type=1, data=df_aov2).round(
            3
        )
        assert not aov2.equals(aov2_ss1)

        # Three-way ANOVA using statsmodels
        # Balanced
        df_aov3 = read_dataset("anova3")
        aov3_ss1 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=1, data=df_aov3
        ).round(3)
        aov3_ss2 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=2, data=df_aov3
        ).round(3)
        aov3_ss3 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=3, data=df_aov3
        ).round(3)
        # Check that type 1 == type 2 == type 3
        assert aov3_ss1.equals(aov3_ss2)
        assert aov3_ss2.equals(aov3_ss3)
        # Compare with JASP
        array_equal(
            aov3_ss1.loc[:, "F"], [2.462, 13.449, 0.484, 0.139, 1.522, 1.446, 1.094, np.nan]
        )
        array_equal(
            aov3_ss1.loc[:, "np2"], [0.049, 0.219, 0.020, 0.003, 0.060, 0.057, 0.044, np.nan]
        )
        array_equal(
            aov3_ss1.loc[:, "p-unc"], [0.123, 0.001, 0.619, 0.711, 0.229, 0.245, 0.343, np.nan]
        )
        # Unbalanced
        df_aov3 = read_dataset("anova3_unbalanced")
        aov3_ss1 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=1, data=df_aov3
        ).round(3)
        aov3_ss2 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=2, data=df_aov3
        ).round(3)
        aov3_ss3 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=3, data=df_aov3
        ).round(3)
        # Compare with JASP
        # Type 1
        array_equal(
            aov3_ss1.loc[:, "F"], [4.155, 15.166, 0.422, 0.085, 0.859, 1.170, 0.505, np.nan]
        )
        array_equal(
            aov3_ss1.loc[:, "np2"], [0.068, 0.210, 0.015, 0.001, 0.029, 0.039, 0.017, np.nan]
        )
        array_equal(
            aov3_ss1.loc[:, "p-unc"], [0.046, 0.0, 0.658, 0.772, 0.429, 0.318, 0.606, np.nan]
        )
        array_equal(
            aov3_ss1.loc[:, "Source"],
            [
                "Sex",
                "Risk",
                "Drug",
                "Sex * Risk",
                "Sex * Drug",
                "Risk * Drug",
                "Sex * Risk * Drug",
                "Residual",
            ],
        )
        # Type 2
        array_equal(
            aov3_ss2.loc[:, "F"], [3.759, 15.169, 0.429, 0.099, 0.739, 1.170, 0.505, np.nan]
        )
        array_equal(
            aov3_ss2.loc[:, "np2"], [0.062, 0.210, 0.015, 0.002, 0.025, 0.039, 0.017, np.nan]
        )
        array_equal(
            aov3_ss2.loc[:, "p-unc"], [0.057, 0.0, 0.653, 0.754, 0.482, 0.318, 0.606, np.nan]
        )

        # Type 3
        array_equal(
            aov3_ss3.loc[:, "F"], [3.910, 15.555, 0.484, 0.079, 0.750, 1.060, 0.505, np.nan]
        )
        array_equal(
            aov3_ss3.loc[:, "np2"], [0.064, 0.214, 0.017, 0.001, 0.026, 0.036, 0.017, np.nan]
        )
        array_equal(
            aov3_ss3.loc[:, "p-unc"], [0.053, 0.0, 0.619, 0.779, 0.477, 0.353, 0.606, np.nan]
        )

        aov3_ss3 = anova(
            dv="Cholesterol", between=["Sex", "Risk", "Drug"], ss_type=3, data=df_aov3, effsize="n2"
        ).round(3)
        array_equal(
            aov3_ss3.loc[:, "n2"], [0.048, 0.189, 0.012, 0.001, 0.018, 0.026, 0.012, np.nan]
        )

        # Error: invalid char in column names
        df_aov3["Sex:"] = np.random.normal(size=df_aov3.shape[0])
        with pytest.raises(ValueError):
            anova(dv="Cholesterol", between=["Sex:", "Risk", "Drug"], data=df_aov3)

    def test_welch_anova(self):
        """Test function welch_anova."""
        # Pain dataset
        df_pain = read_dataset("anova")
        aov = welch_anova(dv="Pain threshold", between="Hair color", data=df_pain).round(4)
        # Compare with JASP
        assert aov.at[0, "ddof1"] == 3
        assert aov.at[0, "ddof2"] == 8.3298
        assert aov.at[0, "F"] == 5.8901
        assert aov.at[0, "p-unc"] == 0.0188
        assert aov.at[0, "np2"] == 0.5760

    def test_rm_anova(self):
        """Test function rm_anova.

        Compare with JAMOVI. As of March 2022, the calculation of the eta-squared is wrong in JASP.

        https://github.com/raphaelvallat/pingouin/issues/251
        """
        rm_anova(
            dv="Scores", within="Time", subject="Subject", data=df, correction=False, detailed=False
        )
        rm_anova(
            dv="Scores", within="Time", subject="Subject", data=df, correction=True, detailed=False
        )
        # Compare with JAMOVI
        aov = rm_anova(
            dv="Scores", within="Time", subject="Subject", data=df, correction="auto", detailed=True
        ).round(5)
        assert aov.at[0, "F"] == 3.91280
        assert aov.at[0, "p-unc"] == 0.02263
        assert aov.at[0, "ng2"] == 0.03998

        # Same but with categorical columns
        aov = rm_anova(
            dv="Scores",
            within="Time",
            subject="Subject",
            data=df_cat,
            correction="auto",
            detailed=True,
        ).round(5)
        assert aov.at[0, "F"] == 3.91280
        assert aov.at[0, "p-unc"] == 0.02263
        assert aov.at[0, "ng2"] == 0.03998

        # With different effect sizes
        aov = rm_anova(
            dv="Scores", within="Time", subject="Subject", data=df, correction="auto", effsize="n2"
        ).round(5)
        assert aov.at[0, "n2"] == 0.03998  # n2 == ng2
        aov = rm_anova(
            dv="Scores",
            within="Time",
            subject="Subject",
            data=df,
            correction="auto",
            detailed=True,
            effsize="np2",
        ).round(5)
        assert aov.at[0, "np2"] == 0.06219

        rm_anova(
            dv="Scores", within="Time", subject="Subject", data=df, correction=True, detailed=True
        )
        rm_anova(dv="Scores", within=["Time"], subject="Subject", data=df_nan)

        # Using a wide dataframe with NaN and compare with JAMOVI
        data = read_dataset("rm_anova_wide")
        aov = data.rm_anova(detailed=True, correction=True).round(5)
        assert aov.at[0, "F"] == 5.20065
        assert aov.at[0, "p-unc"] == 0.00656
        assert aov.at[0, "ng2"] == 0.34639
        assert aov.at[0, "eps"] == 0.69433
        assert aov.at[0, "W-spher"] == 0.30678
        assert aov.at[0, "p-GG-corr"] == 0.01670
        # With different effect sizes
        aov = data.rm_anova(detailed=True, correction=True, effsize="n2").round(5)
        assert aov.at[0, "n2"] == 0.34639  # n2 == ng2
        aov = data.rm_anova(detailed=True, correction=True, effsize="np2").round(5)
        assert aov.at[0, "np2"] == 0.39397  # np2 is bigger than n2

    def test_rm_anova2(self):
        """Test function rm_anova2.

        Compare with JAMOVI.
        """
        data = read_dataset("rm_anova2")
        aov = rm_anova(
            data=data, subject="Subject", within=["Time", "Metric"], dv="Performance"
        ).round(5)
        array_equal(aov.loc[:, "MS"], [828.81667, 682.61667, 112.21667])
        array_equal(aov.loc[:, "F"], [33.85228, 26.95919, 12.63227])
        array_equal(aov.loc[:, "ng2"], [0.25401, 0.35933, 0.08442])
        array_equal(aov.loc[:, "eps"], [1.0, 0.96910, 0.72717])

        # With categorical
        data_cat = data.copy()
        data_cat[["Subject", "Time", "Metric"]] = data_cat[["Subject", "Time", "Metric"]].astype(
            "category"
        )
        data_cat["Time"] = data_cat["Time"].cat.add_categories("Casper")
        aov = rm_anova(
            data=data_cat, subject="Subject", within=["Time", "Metric"], dv="Performance"
        ).round(5)
        array_equal(aov.loc[:, "MS"], [828.81667, 682.61667, 112.21667])
        array_equal(aov.loc[:, "F"], [33.85228, 26.95919, 12.63227])
        array_equal(aov.loc[:, "ng2"], [0.25401, 0.35933, 0.08442])
        array_equal(aov.loc[:, "eps"], [1.0, 0.96910, 0.72717])

        # With different effect sizes
        aov = rm_anova(
            data=data, subject="Subject", within=["Time", "Metric"], dv="Performance", effsize="n2"
        ).round(5)
        array_equal(aov.loc[:, "n2"], [0.17080, 0.28134, 0.04625])

        aov = rm_anova(
            data=data, subject="Subject", within=["Time", "Metric"], dv="Performance", effsize="np2"
        ).round(5)
        array_equal(aov.loc[:, "np2"], [0.78998, 0.74972, 0.58395])

        # 2 factors with missing values. Cannot compare with JASP directly
        # because Pingouin applies an automatic removal of missing values
        # (on the last factor). JASP uses a regression-based approach which
        # can handle missing values.
        df2 = read_dataset("rm_missing")
        df2.rm_anova(dv="BOLD", within=["Session", "Time"], subject="Subj")

        # Error: more than two factors
        with pytest.raises(ValueError):
            df2.rm_anova(dv="BOLD", within=["Session", "Time", "Wrong"], subject="Subj")

    def test_mixed_anova(self):
        """Test function anova.
        Compare with JASP and ezANOVA.
        """
        # trigger error when specifying more than one within or between factor
        with pytest.raises(ValueError, match="/pingouin/issues/136"):
            mixed_anova(
                dv="Scores",
                within=["Time"],
                subject="Subject",
                between="Group",
                data=df,
            )

        # Balanced design, two groups, three within factors
        aov = mixed_anova(
            dv="Scores", within="Time", subject="Subject", between="Group", data=df, correction=True
        ).round(5)
        array_equal(aov.loc[:, "SS"], [5.45996, 7.62843, 5.16719])
        array_equal(aov.loc[:, "DF1"], [1, 2, 2])
        array_equal(aov.loc[:, "DF2"], [58, 116, 116])
        array_equal(aov.loc[:, "F"], [5.05171, 4.02739, 2.72800])
        array_equal(aov.loc[:, "np2"], [0.08012, 0.06493, 0.04492])
        assert round(aov.at[1, "eps"], 3) == 0.999  # Pingouin = 0.99875, JAMOVI = 0.99812
        assert round(aov.at[1, "W-spher"], 3) == 0.999  # Pingouin = 0.99875, JAMOVI = 0.99812
        assert round(aov.at[1, "p-GG-corr"], 2) == 0.02
        # With categorical: should be the same
        aov = mixed_anova(
            dv="Scores",
            within="Time",
            subject="Subject",
            between="Group",
            data=df_cat,
            correction=True,
        ).round(5)
        array_equal(aov.loc[:, "SS"], [5.45996, 7.62843, 5.16719])
        array_equal(aov.loc[:, "DF1"], [1, 2, 2])
        array_equal(aov.loc[:, "DF2"], [58, 116, 116])
        array_equal(aov.loc[:, "F"], [5.05171, 4.02739, 2.72800])
        array_equal(aov.loc[:, "np2"], [0.08012, 0.06493, 0.04492])
        assert round(aov.at[1, "eps"], 3) == 0.999  # Pingouin = 0.99875, JAMOVI = 0.99812
        assert round(aov.at[1, "W-spher"], 3) == 0.999  # Pingouin = 0.99875, JAMOVI = 0.99812
        assert round(aov.at[1, "p-GG-corr"], 2) == 0.02

        # Same with different effect sizes (compare with JAMOVI)
        aov = mixed_anova(
            dv="Scores", within="Time", subject="Subject", between="Group", data=df, effsize="n2"
        ).round(5)
        array_equal(aov.loc[:, "n2"], [0.02862, 0.03998, 0.02708])
        aov = mixed_anova(
            dv="Scores", within="Time", subject="Subject", between="Group", data=df, effsize="ng2"
        ).round(5)
        array_equal(aov.loc[:, "ng2"], [0.03067, 0.04234, 0.02908])

        # With missing values
        df_nan2 = df_nan.copy()
        df_nan2.iloc[158, 0] = np.nan
        aov = mixed_anova(
            dv="Scores",
            within="Time",
            subject="Subject",
            between="Group",
            data=df_nan2,
            correction=True,
        ).round(3)
        array_equal(aov.loc[:, "F"], [5.692, 3.054, 3.502])
        array_equal(aov.loc[:, "np2"], [0.094, 0.053, 0.060])
        assert aov.at[1, "eps"] == 0.997
        assert aov.at[1, "W-spher"] == 0.996

        # Unbalanced group
        df_unbalanced = df[df["Subject"] <= 54]
        aov = mixed_anova(
            data=df_unbalanced,
            dv="Scores",
            subject="Subject",
            within="Time",
            between="Group",
            correction=True,
        ).round(3)
        array_equal(aov.loc[:, "F"], [3.561, 2.421, 1.828])
        array_equal(aov.loc[:, "np2"], [0.063, 0.044, 0.033])
        assert aov.at[1, "eps"] == 1.0  # JASP = 0.998
        assert aov.at[1, "W-spher"] == 1.0  # JASP = 0.998

        # With three groups and four time points, unbalanced (JASP -- type II)
        df_unbalanced = read_dataset("mixed_anova_unbalanced.csv")
        aov = mixed_anova(
            data=df_unbalanced,
            dv="Scores",
            subject="Subject",
            correction=True,
            within="Time",
            between="Group",
        ).round(4)
        array_equal(aov.loc[:, "DF1"], [2, 3, 6])
        array_equal(aov.loc[:, "DF2"], [23, 69, 69])
        array_equal(aov.loc[:, "F"], [2.3026, 1.7071, 0.8877])
        array_equal(aov.loc[:, "p-unc"], [0.1226, 0.1736, 0.5088])
        array_equal(aov.loc[:, "np2"], [0.1668, 0.0691, 0.0717])
        # Check correction: values are very slightly different than ezANOVA
        assert np.isclose(aov.at[1, "eps"], 0.9254, atol=0.01)
        assert np.isclose(aov.at[1, "p-GG-corr"], 0.1779, atol=0.01)
        assert np.isclose(aov.at[1, "W-spher"], 0.8850, atol=0.01)
        assert np.isclose(aov.at[1, "p-spher"], 0.7535, atol=0.1)

        # Same but with different effect sizes
        aov = mixed_anova(
            data=df_unbalanced,
            dv="Scores",
            subject="Subject",
            within="Time",
            between="Group",
            effsize="n2",
        ).round(4)
        array_equal(aov.loc[:, "n2"], [0.0332, 0.0516, 0.0537])
        aov = mixed_anova(
            data=df_unbalanced,
            dv="Scores",
            subject="Subject",
            within="Time",
            between="Group",
            effsize="ng2",
        ).round(4)
        array_equal(aov.loc[:, "ng2"], [0.0371, 0.0566, 0.0587])

        # With overlapping subject IDs in the between-subject groups
        df_overlap = df.copy()
        df_overlap["Subject"] = df_overlap.groupby(["Group"], group_keys=False)["Subject"].apply(
            lambda x: x - x.min()
        )
        with pytest.raises(ValueError):
            mixed_anova(
                dv="Scores", within="Time", subject="Subject", between="Group", data=df_overlap
            )

        df_overlap = df.copy()
        df_overlap["Subject"] = df_overlap["Subject"].replace(57, 3)
        with pytest.raises(ValueError):
            mixed_anova(
                dv="Scores", within="Time", subject="Subject", between="Group", data=df_overlap
            )

    def test_ancova(self):
        """Test function ancovan.
        Compare with JASP.
        """
        df = read_dataset("ancova")
        # With one covariate, balanced design, no missing values
        aov = ancova(data=df, dv="Scores", covar="Income", between="Method").round(4)
        array_equal(aov["DF"], [3, 1, 31])
        array_equal(aov["F"], [3.3365, 29.4194, np.nan])
        array_equal(aov["p-unc"], [0.0319, 0.000, np.nan])
        array_equal(aov["np2"], [0.2441, 0.4869, np.nan])
        aov = ancova(data=df, dv="Scores", covar="Income", between="Method", effsize="n2").round(4)
        array_equal(aov["n2"], [0.1421, 0.4177, np.nan])
        # With one covariate, missing values and unbalanced design
        df.loc[[1, 2], "Scores"] = np.nan
        aov = ancova(data=df, dv="Scores", covar=["Income"], between="Method").round(4)
        array_equal(aov["DF"], [3, 1, 29])
        array_equal(aov["F"], [3.1471, 19.7811, np.nan])
        array_equal(aov["p-unc"], [0.0400, 0.0001, np.nan])
        array_equal(aov["np2"], [0.2456, 0.4055, np.nan])
        # With two covariates, missing values and unbalanced design
        aov = ancova(data=df, dv="Scores", covar=["Income", "BMI"], between="Method").round(4)
        array_equal(aov["DF"], [3, 1, 1, 28])
        array_equal(aov["F"], [3.0186, 19.6045, 1.2279, np.nan])
        array_equal(aov["p-unc"], [0.0464, 0.0001, 0.2772, np.nan])
        array_equal(aov["np2"], [0.2444, 0.4118, 0.0420, np.nan])
        # Same but using standard eta-squared
        aov = ancova(
            data=df, dv="Scores", covar=["Income", "BMI"], between="Method", effsize="n2"
        ).round(4)
        array_equal(aov["n2"], [0.1564, 0.3387, 0.0212, np.nan])
        # Other parameters
        ancova(data=df, dv="Scores", covar=["Income", "BMI"], between="Method")
        ancova(data=df, dv="Scores", covar=["Income"], between="Method")
