|

.. image:: https://badge.fury.io/py/pingouin.svg
  :target: https://badge.fury.io/py/pingouin

.. image:: https://img.shields.io/github/license/raphaelvallat/pingouin.svg
  :target: https://github.com/raphaelvallat/pingouin/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/pingouin.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/pingouin

.. image:: https://codecov.io/gh/raphaelvallat/pingouin/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/pingouin


.. figure::  /pictures/logo_pingouin.png
  :align:   center

**Pingouin** is an open-source statistical package written in Python 3 and based on Pandas and NumPy.

It provides easy-to-grasp functions for computing several statistical tests:

1. ANOVAs: one-way, repeated measures, and mixed (split-plot)

2. Post-hocs tests

3. Parametric and non-parametric T-tests

4. Bayes Factor

5. (Robust) Correlations

6. Tests for sphericity, normality and homoscedasticity

7. Effect sizes

Pingouin is designed for users who want **simple yet exhaustive statistical functions**.

For example, the :code:`ttest_ind` function of SciPy returns only the T-value and the p-value. By contrast,
the :code:`ttest` function of Pingouin returns the t-value, p-value, degrees of freedom, effect size (Cohen's d), statistical power and Bayes Factor (BF10) of the test.


Installation
============

Dependencies
------------

Pingouin requires :

- NumPy
- SciPy
- Pandas

User installation
-----------------

.. code-block:: shell

  pip install pingouin

Develop mode

.. code-block:: shell

  git clone https://github.com/raphaelvallat/pingouin.git pingouin/
  cd pingouin/
  pip install -r requirements.txt
  python setup.py develop


Quick start
============

.. code-block:: python

    import pandas as pd
    from pingouin import mixed_anova, pairwise_ttests, print_table

    # Load dataset
    df = pd.read_csv('sleep_dataset.csv')

    # Compute two-way split-plot ANOVA
    aov = mixed_anova(dv='DV', within='Time', between='Group', data=df,
                     correction='auto', remove_na=False)
    print_table(aov)

    # Compute FDR-corrected post-hocs with effect sizes
    posthocs = pairwise_ttests(dv='DV', within='Time', between='Group', data=df,
                               tail='two-sided', padjust='fdr_bh', effsize='cohen',
                               return_desc=False)
    print_table(posthocs)


.. figure::  /pictures/readme_anova.png
  :align:   center


Contents:
=========

.. toctree::
   :maxdepth: 1

   api
   changelog
   examples


Development:
============

To see the code or report a bug, please visit the `github repository <https://github.com/raphaelvallat/pingouin>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with another statistical software.

Author
======

* `Raphael Vallat <https://raphaelvallat.github.io>`_
