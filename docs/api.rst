.. _api_ref:

.. currentmodule:: pingouin

API reference
=============

ANOVA and T-test
----------------

.. _anova:

.. autosummary::
   :toctree: generated/

    anova
    rm_anova
    rm_anova2
    epsilon
    mixed_anova
    ancova
    ttest
    ttest_power
    anova_power

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

Correlation
-----------

.. _correlations:

.. autosummary::
   :toctree: generated/

    corr
    pairwise_corr
    partial_corr
    rm_corr
    intraclass_corr

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

Multiple comparisons and post-hoc tests
---------------------------------------

.. _multicomp:

.. autosummary::
   :toctree: generated/

    pairwise_ttests
    pairwise_tukey
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
      pingouin.datasets.read_dataset
      pingouin.datasets.list_dataset
