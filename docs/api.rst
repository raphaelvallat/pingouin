.. _api_ref:

.. currentmodule:: pingouin

Functions
#########


.. _anova:

ANOVA and T-test
----------------

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

.. _bayesian:

Bayesian
--------

.. autosummary::
   :toctree: generated/

   bayesfactor_binom
   bayesfactor_ttest
   bayesfactor_pearson

.. _circular:

Circular
--------

.. autosummary::
   :toctree: generated/

   convert_angles
   circ_axial
   circ_corrcc
   circ_corrcl
   circ_mean
   circ_r
   circ_rayleigh
   circ_vtest

.. _contingency:

Contingency
-----------

.. autosummary::
   :toctree: generated/

   chi2_independence
   chi2_mcnemar
   dichotomous_crosstab

.. _correlations:

Correlation and regression
--------------------------

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

.. _distribution:

Distribution
------------

.. autosummary::
   :toctree: generated/

   anderson
   homoscedasticity
   normality
   sphericity

.. _effsize:

Effect sizes
------------

.. autosummary::
   :toctree: generated/

   compute_effsize
   compute_effsize_from_t
   convert_effsize
   compute_esci
   compute_bootci

.. _multicomp:

Multiple comparisons and post-hoc tests
---------------------------------------

.. autosummary::
   :toctree: generated/

   pairwise_tests
   pairwise_tukey
   pairwise_gameshowell
   ptests
   multicomp

.. _multivar:

Multivariate tests
------------------

.. autosummary::
   :toctree: generated/

   box_m
   multivariate_normality
   multivariate_ttest

.. _nonparametric:

Non-parametric
--------------

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

.. _utils:

Others
------

.. autosummary::
   :toctree: generated/

   print_table
   remove_na
   read_dataset
   list_dataset
   set_default_options

.. data:: options
   :type: dict

   Pingouin's global options. Changes apply to every subsequent call, and
   :py:func:`pingouin.set_default_options` restores the defaults.

   * ``options["round"]``: number of decimals of the output dataframes. The
     default is None, i.e. no rounding.
   * ``options["round.column.<name>"]``, ``options["round.row.<name>"]`` and
     ``options["round.cell.[<row>]x[<column>]"]``: rounding of a column, a row
     or a single cell. The first option found is used, in this order: cell,
     column, row, then ``options["round"]``. The value can also be a
     function that formats each value.

   By default, the ``CI95`` column is rounded to 2 decimals and the ``BF10``
   column is formatted as a string. See the
   `rounding notebook <https://github.com/raphaelvallat/pingouin/blob/main/notebooks/06_Rounding.ipynb>`_
   for examples.

   .. code-block:: python

      import pingouin as pg
      pg.options["round"] = 4
      pg.options["round.column.CI95"] = 3

.. _plotting:

Plotting
--------

.. autosummary::
   :toctree: generated/

   plot_blandaltman
   plot_circmean
   plot_paired
   plot_rm_corr
   qqplot

.. _power:

Power analysis
--------------

.. autosummary::
   :toctree: generated/

   power_anova
   power_rm_anova
   power_chi2
   power_corr
   power_ttest
   power_ttest2n

.. _reliability:

Reliability and consistency
---------------------------

.. autosummary::
   :toctree: generated/

   cronbach_alpha
   intraclass_corr
