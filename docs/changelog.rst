.. _Changelog:

What's new
##########

.. contents:: Table of Contents
   :depth: 2

v0.2.9 (September 2019)
-----------------------

**Bugfixes**

a. Disabled default l2 regularization of coefficients in :py:func:`pingouin.logistic_regression`. As pointed out by Eshin Jolly in `PR54 <https://github.com/raphaelvallat/pingouin/pull/54>`_, scikit-learn automatically applies a penalization of coefficients, which in turn makes the estimation of standard errors and p-values not totally correct/interpretable. This regularization behavior is now disabled, resulting in the same behavior as R ``glm(..., family=binomial)``.

**Code and dependencies**

a. Pandas methods are now internally defined using the `pandas_flavor package <https://github.com/Zsailer/pandas_flavor>`_ package.
b. Internal code refactoring of the :py:func:`pingouin.pairwise_ttests` (to slightly speed up computation and improve memory usage).
c. The first argument of the :py:func:`pingouin.anova`, :py:func:`pingouin.ancova`, :py:func:`pingouin.welch_anova`, :py:func:`pingouin.pairwise_ttests`, :py:func:`pingouin.pairwise_tukey`, :py:func:`pingouin.pairwise_gameshowell`, :py:func:`pingouin.welch_anova`, :py:func:`pingouin.kruskal`, :py:func:`pingouin.friedman`, :py:func:`pingouin.cochran`, :py:func:`pingouin.remove_rm_na` functions is now ``data`` instead of ``dv`` (to be consistent with other Pingouin functions). This will cause error if the user runs previous Pingouin code with positional-only arguments. As a general rule, **you should always pass keywords arguments** (read more `here <https://treyhunner.com/2018/04/keyword-arguments-in-python/>`_).
d. For clarity, :py:func:`pingouin.fdr`, :py:func:`pingouin.bonf`, :py:func:`pingouin.holm` have been deprecated from the API and must be called via :py:func:`pingouin.multicomp`.
e. :py:func:`pingouin.pairwise_ttests` output does not include the ``CLES`` column by default anymore. Users must explicitly pass ``effsize='CLES'``.
f. The ``remove_na`` argument of :py:func:`pingouin.cronbach_alpha` has been replaced with ``nan_policy`` (`'pairwise'`, or `'listwise'`).
g. Disabled Travis / AppVeyor testing for Python 3.5 While most functions should work just fine, please note that only Python >3.6 is supported now.

**New functions**

a. Added :py:func:`pingouin.harrelldavis`, a robust quantile estimation method (to be used in a future version of the :py:func:`pingouin.plot_shift` function). See `PR63 <https://github.com/raphaelvallat/pingouin/pull/63>`_ by Nicolas Legrand.
b. The :py:func:`pingouin.ancova` can now directly be used a Pandas method, e.g. ``data.ancova(...)``.
c. The :py:func:`pingouin.pairwise_tukey` can now directly be used a Pandas method, e.g. ``data.pairwise_tukey(...)``.
d. Added Sidak one-step correction to :py:func:`pingouin.multicomp` (``method='sidak'``).

**Enhancements**

a. Added support for pairwise deletion in :py:func:`pingouin.pairwise_ttests` (default is listwise deletion), using the ``nan_policy`` argument.
b. Added support for listwise deletion in :py:func:`pingouin.pairwise_corr` (default is pairwise deletion), using the ``nan_policy`` argument.
c. Added the ``interaction`` boolean argument to :py:func:`pingouin.pairwise_ttests`, useful if one is only interested in the main effects.
d. Added ``correction_uniform`` boolean argument to :py:func:`pingouin.circ_corrcc`. See `PR64 <https://github.com/raphaelvallat/pingouin/pull/64>`_ by Dominik Straub.

**Contributors**

* `Raphael Vallat <https://raphaelvallat.com>`_
* `Eshin Jolly <http://eshinjolly.com/>`_
* Nicolas Legrand
* Dominik Straub

v0.2.8 (July 2019)
------------------

**Dependencies**

a. Pingouin now requires SciPy >= 1.3.0 (better handling of tails in :py:func:`pingouin.wilcoxon` function) and Pandas >= 0.24 (fixes a minor bug with 2-way within factor interaction in :py:func:`pingouin.epsilon` with previous version)

**New functions**

a. Added :py:func:`pingouin.rcorr` Pandas method to calculate a correlation matrix with r-values on the lower triangle and p-values (or sample size) on the upper triangle.
b. Added :py:func:`pingouin.tost` function to calculate the two one-sided test (TOST) for equivalence. See `PR51 <https://github.com/raphaelvallat/pingouin/pull/51>`_ by Antoine Weill--Duflos.

**Enhancements**

a. :py:func:`pingouin.anova` now works with three or more between factors (requiring statsmodels). One-way ANOVA and balanced two-way ANOVA are computed in pure Pingouin (Python + Pandas) style, while ANOVA with three or more factors, or unbalanced two-way ANOVA are computed using statsmodels.
b. :py:func:`pingouin.anova` now accepts different sums of squares calculation method for unbalanced N-way design (type 1, 2, or 3).
c. :py:func:`pingouin.linear_regression` now includes several safety checks to remove duplicate predictors, predictors with only zeros, and predictors with only one unique value (excluding the intercept). This comes at the cost, however, of longer computation time, which is evident when using the :py:func:`pingouin.mediation_analysis` function.
d. :py:func:`pingouin.mad` now automatically removes missing values and can calculate the mad over the entire array using ``axis=None`` if array is multidimensional.
e. Better handling of alternative hypotheses in :py:func:`pingouin.wilcoxon`.
f. Better handling of alternative hypotheses in :py:func:`pingouin.bayesfactor_ttest` (support for 'greater' and 'less').
g. Better handling of alternative hypotheses in :py:func:`pingouin.ttest` (support for 'greater' and 'less'). This is also taken into account when calculating the Bayes Factor and power of the test.
h. Better handling of alternative hypotheses in :py:func:`pingouin.power_ttest` and :py:func:`pingouin.power_ttest2n` (support for 'greater' and 'less', and removed 'one-sided').
i. Implemented a new method to calculate the matched pair rank biserial correlation effect size for :py:func:`pingouin.wilcoxon`, which gives results almost identical to JASP.

v0.2.7 (June 2019)
------------------

**Dependencies**

a. Pingouin now requires statsmodels>=0.10.0 (latest release June 2019) and is compatible with SciPy 1.3.0.

**Enhancements**

a. Added support for long-format dataframe in :py:func:`pingouin.sphericity` and :py:func:`pingouin.epsilon`.
b. Added support for two within-factors interaction in :py:func:`pingouin.sphericity` and :py:func:`pingouin.epsilon` (for the former, granted that at least one of them has no more than two levels.)

**New functions**

a. Added :py:func:`pingouin.power_rm_anova` function.

v0.2.6 (June 2019)
------------------

**Bugfixes**

a. Fixed **major error in two-sided p-value for Wilcoxon test** (:py:func:`pingouin.wilcoxon`), the p-values were accidentally squared, and therefore smaller. Make sure to always use the latest release of Pingouin.
b. :py:func:`pingouin.wilcoxon` now uses the continuity correction by default (the documentation was saying that the correction was applied but it was not applied in the code.)
c. The ``show_median`` argument of the :py:func:`pingouin.plot_shift` function was not working properly when the percentiles were different that the default parameters.

**Dependencies**

a. The current release of statsmodels (0.9.0) is not compatible with the newest release of Scipy (1.3.0). In order to avoid compatibility issues in the :py:func:`pingouin.ancova` and :py:func:`pingouin.anova` functions (which rely on statsmodels for certain cases), Pingouin will require SciPy < 1.3.0 until a new stable version of statsmodels is released.

**New functions**

a. Added :py:func:`pingouin.chi2_independence` tests.
b. Added :py:func:`pingouin.chi2_mcnemar` tests.
c. Added :py:func:`pingouin.power_chi2` function.
d. Added :py:func:`pingouin.bayesfactor_binom` function.

**Enhancements**

a. :py:func:`pingouin.linear_regression` now returns the residuals.
b. Completely rewrote :py:func:`pingouin.normality` function, which now support pandas DataFrame (wide & long format), multiple normality tests (:py:func:`scipy.stats.shapiro`, :py:func:`scipy.stats.normaltest`), and an automatic casewise removal of missing values.
c. Completely rewrote :py:func:`pingouin.homoscedasticity` function, which now support pandas DataFrame (wide & long format).
d. Faster and more accurate algorithm in :py:func:`pingouin.bayesfactor_pearson` (same algorithm as JASP).
e. Support for one-sided Bayes Factors in :py:func:`pingouin.bayesfactor_pearson`.
f. Better handling of required parameters in :py:func:`pingouin.qqplot`.
g. The epsilon value for the interaction term in :py:func:`pingouin.rm_anova` are now computed using the Greenhouse-Geisser method instead of the lower bound. A warning message has been added to the documentation to alert the user that the value might slightly differ than from R or JASP.

Note that d. and e. also affect the behavior of the :py:func:`pingouin.corr` and :py:func:`pingouin.pairwise_corr` functions.

**Contributors**

* `Raphael Vallat <https://raphaelvallat.com>`_
* `Arthur Paulino <https://github.com/arthurpaulino>`_

v0.2.5 (May 2019)
-----------------

**MAJOR BUG FIXES**

a. Fixed error in p-values for **one-sample one-sided T-test** (:py:func:`pingouin.ttest`), the two-sided p-value was divided by 4 and not by 2, resulting in inaccurate (smaller) one-sided p-values.
b. Fixed global error for **unbalanced two-way ANOVA** (:py:func:`pingouin.anova`), the sums of squares were wrong, and as a consequence so were the F and p-values. In case of unbalanced design, Pingouin now computes a type II sums of squares via a call to the statsmodels package.
c. The epsilon factor for the interaction term in two-way repeated measures ANOVA (:py:func:`pingouin.rm_anova`) is now computed using the lower bound approach. This is more conservative than the Greenhouse-Geisser approach and therefore give (slightly) higher p-values. The reason for choosing this is that the Greenhouse-Geisser values for the interaction term differ than the ones returned by R and JASP. This will be hopefully fixed in future releases.

**New functions**

a. Added :py:func:`pingouin.multivariate_ttest` (Hotelling T-squared) test.
b. Added :py:func:`pingouin.cronbach_alpha` function.
c. Added :py:func:`pingouin.plot_shift` function.
d. Several functions of pandas can now be directly used as :py:class:`pandas.DataFrame` methods.
e. Added :py:func:`pingouin.pcorr` method to compute the partial Pearson correlation matrix of a :py:class:`pandas.DataFrame` (similar to the pcor function in the ppcor package).
f. The :py:func:`pingouin.partial_corr` now supports semi-partial correlation.

**Enhancements**

a. The :py:func:`pingouin.rm_corr` function now returns a :py:class:`pandas.DataFrame` with the r-value, degrees of freedom, p-value, confidence intervals and power.
b. :py:func:`pingouin.compute_esci` now works for paired and one-sample Cohen d.
c. :py:func:`pingouin.bayesfactor_ttest` and :py:func:`pingouin.bayesfactor_pearson` now return a formatted str and not a float.
d. :py:func:`pingouin.pairwise_ttests` now returns the degrees of freedom (dof).
e. Better rounding of float in :py:func:`pingouin.pairwise_ttests`.
f. Support for wide-format data in :py:func:`pingouin.rm_anova`
g. :py:func:`pingouin.ttest` now returns the confidence intervals around the difference in means.

**Missing values**

a. :py:func:`pingouin.remove_na` and :py:func:`pingouin.remove_rm_na` are now external function documented in the API.
b. :py:func:`pingouin.remove_rm_na` now works with multiple within-factors.
c. :py:func:`pingouin.remove_na` now works with 2D arrays.
d. Removed the `remove_na` argument in :py:func:`pingouin.rm_anova` and :py:func:`pingouin.mixed_anova`, an automatic listwise deletion of missing values is applied (same behavior as JASP). Note that this was also the default behavior of Pingouin, but the user could also specify not to remove the missing values, which most likely returned inaccurate results.
e. The :py:func:`pingouin.ancova` function now applies an automatic listwise deletion of missing values.
f. Added `remove_na` argument (default = False) in :py:func:`pingouin.linear_regression` and :py:func:`pingouin.logistic_regression` functions
g. Missing values are automatically removed in the :py:func:`pingouin.anova` function.

**Contributors**

* Raphael Vallat
* Nicolas Legrand

v0.2.4 (April 2019)
-------------------

**Correlation**

a. Added :py:func:`pingouin.distance_corr` (distance correlation) function.
b. :py:func:`pingouin.rm_corr` now requires at least 3 unique subjects (same behavior as the original R package).
c. The :py:func:`pingouin.pairwise_corr` is faster and returns the number of outlier if a robust correlation is used.
d. Added support for 2D level in the :py:func:`pingouin.pairwise_corr`. See Jupyter notebooks for examples.
e. Added support for partial correlation in the :py:func:`pingouin.pairwise_corr` function.
f. Greatly improved execution speed of :py:func:`pingouin.correlation.skipped` function.
g. Added default random state to compute the Min Covariance Determinant in the :py:func:`pingouin.correlation.skipped` function.
h. The default number of bootstrap samples for the :py:func:`pingouin.correlation.shepherd` function is now set to 200 (previously 2000) to increase computation speed.
i. :py:func:`pingouin.partial_corr` now automatically drops rows with missing values.

**Datasets**

a. Renamed :py:func:`pingouin.read_dataset` and :py:func:`pingouin.list_dataset` (before one needed to call these functions by calling pingouin.datasets)

**Pairwise T-tests and multi-comparisons**

a. Added support for non-parametric pairwise tests in :py:func:`pingouin.pairwise_ttests` function.
b. Common language effect size (CLES) is now reported by default in :py:func:`pingouin.pairwise_ttests` function.
c. CLES is now implemented in the :py:func:`pingouin.compute_effsize` function.
d. Better code, doc and testing for the functions in multicomp.py.
e. P-values adjustment methods now do not take into account NaN values (same behavior as the R function p.adjust)

**Plotting**

a. Added :py:func:`pingouin.plot_paired` function.

**Regression**

a. NaN are now automatically removed in :py:func:`pingouin.mediation_analysis`.
b. The :py:func:`pingouin.linear_regression` and :py:func:`pingouin.logistic_regression` now fail if NaN / Inf are present in the target or predictors variables. The user must remove then before running these functions.
c. Added support for multiple parallel mediator in :py:func:`pingouin.mediation_analysis`.
d. Added support for covariates in :py:func:`pingouin.mediation_analysis`.
e. Added seed argument to :py:func:`pingouin.mediation_analysis` for reproducible results.
f. :py:func:`pingouin.mediation_analysis` now returns two-sided p-values computed with a permutation test.
g. Added :py:func:`pingouin.utils._perm_pval` to compute p-value from a permutation test.

**Bugs and tests**

a. Travis and AppVeyor test for Python 3.5, 3.6 and 3.7.
b. Better doctest & improved examples for many functions.
c. Fixed bug with :py:func:`pingouin.mad` when axis was not 0.

v0.2.3 (February 2019)
----------------------

**Correlation**

a. `shepherd` now also returns the outlier vector (same behavior as skipped).
b. The `corr` function returns the number of outliers for shepherd and skipped.
c. Removed `mahal` function.

**Licensing**

a. Pingouin is now released under the GNU General Public Licence 3.
b. Added licenses files of external modules (qsturng and tabulate).

**Plotting**

a. NaN are automatically removed in qqplot function

v0.2.2 (December 2018)
----------------------

**Plotting**

a. Started working on Pingouin's plotting module
b. Added Seaborn and Matplotlib to dependencies
c. Added plot_skipped_corr function (PR from Nicolas Legrand)
d. Added qqplot function (Quantile-Quantile plot)
e. Added plot_blandaltman function (Bland-Altman plot)

**Power**

a. Added power_corr, based on the R `pwr` package.
b. Renamed anova_power and ttest_power to power_anova and power_ttest.
c. Added power column to corr() and pairwise_corr()
d. power_ttest function can now solve for sample size, alpha and d
e. power_ttest2n for two-sample T-test with unequal n.
f. power_anova can now solve for sample size, number of groups, alpha and eta

v0.2.1 (November 2018)
----------------------

**Effect size**

a. Separated compute_esci and compute_bootci
b. Added corrected percentile method and normal approximation to bootstrap
c. Fixed bootstrapping method

v0.2.0 (November 2018)
----------------------

**ANOVA**

a. Added Welch ANOVA
b. Added Games-Howell post-hoc test for one-way ANOVA with unequal variances
c. Pairwise T-tests now accepts two within or two between factors
d. Fixed error in padjust correction in the pairwise_ttests function: correction was applied on all p-values at the same time.

**Correlation/Regression**

a. Added linear_regression function.
b. Added logistic_regression function.
c. Added mediation_analysis function.
d. Support for advanced indexing (product / combination) in pairwise_corr function.

**Documentation**

a. Added Guidelines section with flow charts
b. Renamed API section to Functions
c. Major improvements to the documentation of several functions
d. Added Gitter channel

v0.1.10 (October 2018)
----------------------

**Bug**

a. Fixed dataset names in MANIFEST.in (.csv files were not copy-pasted with pip)

**Circular**

a. Added circ_vtest function

**Distribution**

a. Added multivariate_normality function (Henze-Zirkler's Multivariate Normality Test)
b. Renamed functions test_normality, test_sphericity and test_homoscedasticity to normality, sphericity and homoscedasticity to avoid bugs with pytest.
c. Moved distribution tests from parametric.py to distribution.py


v0.1.9 (October 2018)
---------------------

**Correlation**

a. Added partial_corr function (partial correlation)

**Doc**

a. Minor improvements in docs and binder notebooks


v0.1.8 (October 2018)
---------------------

**ANOVA**

a. Added support for multiple covariates in ANCOVA function (requires statsmodels).

**Documentation**

a. Major re-organization in API category
b. Added equations and references for effect sizes and Bayesian functions.

**Non-parametric**

a. Added cochran function (Cochran Q test)


v0.1.7 (September 2018)
-----------------------

**ANOVA**

a. Added rm_anova2 function (two-way repeated measures ANOVA).
b. Added ancova function (Analysis of covariance)

**Correlations**

a. Added intraclass_corr function (intraclass correlation).
b. The rm_corr function uses the new ancova function instead of statsmodels.

**Datasets**

a. Added ancova and icc datasets

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
