Pingouin
########

.. image:: https://badge.fury.io/py/pingouin.svg
  :target: https://badge.fury.io/py/pingouin

.. image:: https://img.shields.io/github/license/raphaelvallat/pingouin.svg
  :target: https://github.com/raphaelvallat/pingouin/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/pingouin.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/pingouin

.. image:: https://codecov.io/gh/raphaelvallat/pingouin/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/pingouin


.. figure::  /pictures/pingouin.png
  :align:   center

**Pingouin** (French word for *penguin*) is an open-source statistical Python 3 package based on Pandas.

Its main features are:

1. ANOVA: one-way, repeated measures, and mixed (split-plot)

2. Post-hocs pairwise T-tests

3. Tests for sphericity, normality and homoscedasticity

4. Computation and conversion of effect sizes


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
     from pingouin import pairwise_ttests, print_table

     # Load a fake dataset: the INSOMNIA study
     # Goal: evaluate the influence of a treatment on sleep duration in a control
     # and insomnia group
     # Mixed repeated measures design
     #   - Dependent variable ('DV') = hours of sleep per night
     #   - Between-factor ('Group') = two-levels (Insomnia / Control)
     #   - Within-factor ('Time') = three levels (Pre, Post-6months, Post-12months)
     df = pd.read_csv('examples/sleep_dataset.csv')
     print(df.head())

     # Compute two-way split-plot ANOVA
     aov = mixed_anova(dv='DV', within='Time', between='Group', data=df,
                      correction='auto', remove_na=False)
     print_table(aov)

     # Compute FDR-corrected post-hocs with effect sizes
     posthocs = pairwise_ttests(dv='DV', within='Time', between='Group', data=df,
                                tail='two-sided', padjust='fdr_bh', effsize='cohen',
                                return_desc=False)
     print_table(posthocs)


Output:

.. figure::  /pictures/readme_anova.png
  :align:   center


Contents:
=========

.. toctree::
   :maxdepth: 1

   examples
   changelog
   api


Development:
============

To see the code or report a bug, please visit the `github repository <https://github.com/raphaelvallat/pingouin>`_.
