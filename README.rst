.. -*- mode: rst -*-

Pingouin
########

.. figure::  https://github.com/raphaelvallat/pingouin/blob/master/docs/pictures/pingouin.png
   :align:   center


**Pingouin** (French word for *penguin*) is an open-source statistical Python 3 package based on Pandas.

Its main features are:

1. One-way repeated measures ANOVA

2. Post-hocs pairwise T-tests from a mixed model ANOVA

3. Tests for sphericity, normality and homoscedasticity

4. Computation and conversion of effect sizes

Documentation
=============

- `Link to documentation <https://raphaelvallat.github.io/pingouin/build/html/index.html>`_

Installation
============

Dependencies
------------

Pingouin requires :

* NumPy
* SciPy
* Pandas
* Tabulate (optional: allows pretty-printing of output table)

User installation
-----------------

Installing Pingouin:

.. code-block:: shell

  git clone https://github.com/raphaelvallat/pingouin.git pingouin/
  cd pingouin/
  pip install -r requirements.txt
  python setup.py develop


Quick start
============

.. code-block:: python

  import pandas as pd
  from pingouin import pairwise_ttests

  # Load a fake dataset: the INSOMNIA study
  # Goal: evaluate the influence of a treatment on sleep duration in a control
  # and insomnia group
  # Mixed repeated measures design
  #   - Dependent variable ('DV') = hours of sleep per night
  #   - Between-factor ('Group') = two-levels (Insomnia / Control)
  #   - Within-factor ('Time') = three levels (Pre, Post1, Post2)
  df = pd.read_csv('examples/sleep_dataset.csv')

  stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                          effects='all', data=df, alpha=.05,
                          tail='two-sided', padjust='fdr_by', effsize='hedges')
  print(stats)

Output:

.. figure::  https://github.com/raphaelvallat/pingouin/blob/master/docs/pictures/pairwise_stats_all.png
   :align:   center


Author
======

* `Raphael Vallat <https://raphaelvallat.github.io>`_


Further reading
===============

* Effect size: see `Lakens et al. 2013 <https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full>`_
