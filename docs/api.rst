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
    welch_anova
    rm_anova
    rm_anova2
    epsilon
    mixed_anova
    ancova
    ttest

Bayesian
--------

.. _bayesian:

.. autosummary::
   :toctree: generated/

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

Correlation and regression
--------------------------

.. _correlations:

.. autosummary::
   :toctree: generated/

    corr
    pairwise_corr
    partial_corr
    distance_corr
    rm_corr
    intraclass_corr
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
    multivariate_normality
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

    pairwise_ttests
    pairwise_tukey
    pairwise_gameshowell
    multicomp
    bonf
    holm
    fdr

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

Others
------

.. _utils:

.. autosummary::
     :toctree: generated/

      print_table
      read_dataset
      list_dataset

Plotting
--------

.. _plotting:

.. autosummary::
     :toctree: generated/

      plot_blandaltman
      plot_paired
      plot_skipped_corr
      qqplot

Power analysis
--------------

.. _power:

.. autosummary::
     :toctree: generated/

      power_anova
      power_corr
      power_ttest
      power_ttest2n
