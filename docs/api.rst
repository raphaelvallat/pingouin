.. _api_ref:

.. currentmodule:: pingouin

Functions
=========

.. contents:: Table of Contents
   :depth: 2

ANOVA and T-test
----------------

.. _anova:

.. autosummary::
   :toctree: generated/

    anova
    ancova
    rm_anova
    epsilon
    mixed_anova
    welch_anova
    tost
    ttest

Bayesian
--------

.. _bayesian:

.. autosummary::
   :toctree: generated/

    bayesfactor_binom
    bayesfactor_ttest
    bayesfactor_pearson

Circular
--------

.. _circular:

.. autosummary::
   :toctree: generated/

    circ_axial
    circ_corrcc
    circ_corrcl
    circ_mean
    circ_r
    circ_rayleigh
    circ_vtest

Contingency
-----------

.. _contingency:

.. autosummary::
   :toctree: generated/

    chi2_independence
    chi2_mcnemar
    dichotomous_crosstab

Correlation and regression
--------------------------

.. _correlations:

.. autosummary::
   :toctree: generated/

    corr
    pairwise_corr
    partial_corr
    pcorr
    rcorr
    distance_corr
    rm_corr
    linear_regression
    logistic_regression
    mediation_analysis

Distribution
------------

.. _parametric:

.. autosummary::
   :toctree: generated/

    anderson
    gzscore
    homoscedasticity
    normality
    sphericity

Effect sizes
------------

.. _effsize:

.. autosummary::
   :toctree: generated/

    compute_effsize
    compute_effsize_from_t
    convert_effsize
    compute_esci
    compute_bootci

Multiple comparisons and post-hoc tests
---------------------------------------

.. _multicomp:

.. autosummary::
   :toctree: generated/

    pairwise_corr
    pairwise_ttests
    pairwise_tukey
    pairwise_gameshowell
    multicomp

Multivariate tests
------------------

.. _multivar:

.. autosummary::
   :toctree: generated/

    multivariate_normality
    multivariate_ttest

Non-parametric
--------------

.. _nonparametric:

.. autosummary::
   :toctree: generated/

    cochran
    friedman
    kruskal
    mad
    madmedianrule
    mwu
    wilcoxon
    harrelldavis

Others
------

.. _utils:

.. autosummary::
     :toctree: generated/

      print_table
      remove_na
      remove_rm_na
      read_dataset
      list_dataset

Plotting
--------

.. _plotting:

.. autosummary::
     :toctree: generated/

      plot_blandaltman
      plot_paired
      plot_shift
      plot_skipped_corr
      qqplot

Power analysis
--------------

.. _power:

.. autosummary::
     :toctree: generated/

      power_anova
      power_rm_anova
      power_chi2
      power_corr
      power_ttest
      power_ttest2n

Reliability and consistency
---------------------------

.. _reliability:

.. autosummary::
     :toctree: generated/

      cronbach_alpha
      intraclass_corr
