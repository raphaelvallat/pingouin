.. _Changelog:

What's new
##########

v0.1.7 (dev)
------------

**ANOVA**

a. Added rm_anova2 function (two-way repeated measures ANOVA).
b. Added ancova function (Analysis of covariance)

**Correlations**

a. Added intraclass_corr function (intraclass correlation).
b. The rm_corr function uses the new ancova function instead of statsmodels.

**Effect size**

a. Fixed bug in Cohen d: now use unbiased standard deviation (np.std(ddof=1)) for paired and one-sample Cohen d.
   Please make sure to use pingouin >= 0.1.7 to avoid any mistakes on the paired effect sizes.


v0.1.6 (September 2018)
-----------------------

**ANOVA**

a. Added JNS method to compute sphericity.

**Bug**

a. Added .csv datasets files to python site-packages folder
b. Fixed error in test_sphericity when ddof == 0.


v0.1.5 (August 2018)
--------------------

**ANOVA**

a. rm_anova, friedman and mixed_anova now require a subject identifier. This avoids improper collapsing when multiple repeated measures factors are present in the dataset.
b. rm_anova, friedman and mixed_anova now support the presence of other repeated measures factors in the dataset.
c. Fixed error in test_sphericity
d. Better output of ANOVA summary
e. Added epsilon function

**Code**

a. Added AppVeyor CI (Windows)
b. Cleaned some old functions

**Correlation**

a. Added repeated measures correlation (Bakdash and Marusich 2017).
b. Added robust skipped correlation (Rousselet and Pernet 2012).
c. Pairwise_corr function now automatically delete non-numeric columns.

**Dataset**

a. Added pingouin.datasets module (read_dataset & list_dataset functions)
b. Added datasets: bland1995, berens2009, dolan2009, mcclave1991

**Doc**

a. Examples are now Jupyter Notebooks.
b. Binder integration

**Misc**

a. Added median absolute deviation (mad)
b. Added mad median rule (Wilcox 2012)
c. Added mahal function (equivalent of Matlab mahal function)

**Parametric**

a. Added two-way ANOVA.
b. Added pairwise_tukey function


v0.1.4 (July 2018)
------------------
**Installation**

a. Fix bug with pip install caused by pingouin.external

**Circular statistics**

a. Added circ_corrcc, circ_corrcl, circ_r, circ_rayleigh

v0.1.3 (June 2018)
------------------
**Documentation**

a. Added several tutorials
b. Improved doc of several functions

**Bayesian**

a. T-test now reports the Bayes factor of the alternative hypothesis (BF10)
b. Pearson correlation now reports the Bayes factor of the alternative hypothesis (BF10)

**Non-parametric**

a. Kruskal-Wallis test
b. Friedman test

**Correlations**

a. Added Shepherd's pi correlation (Schwarzkopf et al. 2012)
b. Fixed bug in confidence intervals of correlation coefficients
c. Parametric 95% CI are returned by default when calling corr

v0.1.2 (June 2018)
------------------

**Correlation**

a. Pearson
b. Spearman
c. Kendall
d. Percentage bend (robust)
e. Pairwise correlations between all columns of a pandas dataframe

**Non-parametric**

a. Mann-Whitney U
b. Wilcoxon signed-rank
c. Rank-biserial correlation effect size
d. Common language effect size


v0.1.1 (April 2018)
-------------------

**ANOVA**

a. One-way
b. One-way repeated measures
c. Two-way split-plot (one between factor and one within factor)

**Miscellaneous statistical functions**

a. T-tests
b. Power of T-tests and one-way ANOVA

v0.1.0 (April 2018)
-------------------

Initial release.

**Pairwise comparisons**

a. FDR correction (BH / BY)
b. Bonferroni
c. Holm

**Effect sizes**:

a. Cohen's d (independent and repeated measures)
b. Hedges g
c. Glass delta
d. Eta-square
e. Odds-ratio
f. Area Under the Curve

**Miscellaneous statistical functions**

a. Geometric Z-score
b. Normality, sphericity homoscedasticity and distributions tests

**Code**

a. PEP8 and Flake8
b. Tests and code coverage
