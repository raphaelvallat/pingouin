.. _api_ref:

.. currentmodule:: pingouin

API reference
=============

ANOVA
-----

.. _anova:

.. autosummary::
   :toctree: generated/

    anova
    rm_anova
    mixed_anova
    kruskal
    friedman

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

    circ_corrcc
    circ_corrcl
    circ_r
    circ_rayleigh

Correlation
-----------

.. _correlations:

.. autosummary::
   :toctree: generated/

    corr
    pairwise_corr

Distribution
------------

.. _parametric:

.. autosummary::
   :toctree: generated/

    gzscore
    test_normality
    test_homoscedasticity
    test_sphericity
    test_dist


Effect sizes
------------

.. _effsize:

.. autosummary::
   :toctree: generated/

    compute_effsize
    compute_effsize_from_t
    convert_effsize
    compute_esci


Miscellaneous
-------------

.. _utils:

.. autosummary::
     :toctree: generated/

      print_table
      reshape_data


Multiple comparisons
--------------------

.. _multicomp:

.. autosummary::
   :toctree: generated/

    pairwise_ttests
    pairwise_tukey
    multicomp
    bonf
    holm
    fdr


Power
-----

.. _power:

.. autosummary::
   :toctree: generated/

    ttest_power
    anova_power


T-tests
-------

.. _ttests:

.. autosummary::
   :toctree: generated/

    ttest
    mwu
    wilcoxon
