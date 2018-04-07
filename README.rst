.. -*- mode: rst -*-

Pingouin
########

.. figure::  https://github.com/raphaelvallat/pingouin/blob/master/docs/pictures/pingouin.png
   :align:   center


**Pingouin** (French word for *penguin*) is an open-source statistical Python 3 package based on Pandas.

Its main features are:

1. ANOVA: one-way, repeated measures, and mixed (split-plot)

2. Post-hocs pairwise T-tests

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
    from pingouin import mixed_anova, pairwise_ttests, print_table

    # Load dataset
    df = pd.read_csv('sleep_dataset.csv')
    print(df.head())

    # Compute two-way split-plot ANOVA
    aov = mixed_anova(dv='DV', within='Time', between='Group', data=df)
    print_table(aov)

    # Compute FDR-corrected post-hocs with effect sizes
    posthocs = pairwise_ttests(dv='DV', within='Time', between='Group', data=df,
                               tail='two-sided', padjust='fdr_bh', effsize='cohen')
    print_table(posthocs)

Output:

.. figure::  https://github.com/raphaelvallat/pingouin/blob/master/docs/pictures/readme_anova.png
   :align:   center


Author
======

* `Raphael Vallat <https://raphaelvallat.github.io>`_
