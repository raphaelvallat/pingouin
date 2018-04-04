Pingouin
########

.. figure::  /pictures/pingouin.png
  :align:   center

**Pingouin** (French word for *penguin*) is an open-source statistical Python 3 package based on Pandas.

Its main features are:

1. One-way repeated measures ANOVA

2. Post-hocs pairwise T-tests from a mixed model ANOVA

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

.. figure::  /pictures/pairwise_stats_all.png
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
