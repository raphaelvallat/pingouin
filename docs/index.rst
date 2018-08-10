|

.. image:: https://badge.fury.io/py/pingouin.svg
  :target: https://badge.fury.io/py/pingouin

.. image:: https://img.shields.io/github/license/raphaelvallat/pingouin.svg
  :target: https://github.com/raphaelvallat/pingouin/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/pingouin.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/pingouin

.. image:: https://ci.appveyor.com/api/projects/status/v7fhavoqj8ig1bs2?svg=true
    :target: https://ci.appveyor.com/project/raphaelvallat/pingouin

.. image:: https://codecov.io/gh/raphaelvallat/pingouin/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/pingouin


.. figure::  /pictures/logo_pingouin.png
  :align:   center

**Pingouin** is an open-source statistical package written in Python 3 and based on Pandas and NumPy.

It provides easy-to-grasp functions for computing several statistical tests:

1. ANOVAs: one- and two-ways, repeated measures, and mixed (split-plot)

2. Post-hocs tests

3. Parametric and non-parametric T-tests

4. Bayes Factor

5. (Robust) Correlations

6. Tests for sphericity, normality and homoscedasticity

7. Effect sizes

8. Circular statistics

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

10 minutes to Pingouin
----------------------

1. T-test
#########

.. code-block:: ipython3

  # Generate two correlated random variables
  import numpy as np
  np.random.seed(123)
  mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
  x, y = np.random.multivariate_normal(mean, cov, n).T

  # T-test
  from pingouin import ttest
  ttest(x, y)

.. table:: Output
   :widths: auto

   =======  =======  =====  =========  =========  =======  ======
     T-val    p-val    dof  tail         cohen-d    power    BF10
   =======  =======  =====  =========  =========  =======  ======
    -3.401    0.001     58  two-sided      0.878    0.917  26.155
   =======  =======  =====  =========  =========  =======  ======

------------

2. Pearson's correlation
########################

.. code-block:: ipython3

  from pingouin import corr
  corr(x, y)

.. table:: Output
   :widths: auto

   =====  ===========  =====  ========  =======  ======
       r  CI95%           r2    adj_r2    p-val    BF10
   =====  ===========  =====  ========  =======  ======
   0.595  [0.3  0.79]  0.354     0.306    0.001  54.222
   =====  ===========  =====  ========  =======  ======

------------

3. Robust correlation
#####################

.. code-block:: ipython3

  # Introduce an outlier
  x[5] = 18
  # Use the robust Shepherd's pi correlation
  corr(x, y, method="shepherd")

.. table:: Output
   :widths: auto

   =====  ===========  =====  ========  =======
       r  CI95%           r2    adj_r2    p-val
   =====  ===========  =====  ========  =======
   0.561  [0.25 0.77]  0.315     0.264    0.003
   =====  ===========  =====  ========  =======

------------

4. Test the normality of the data
#################################

.. code-block:: ipython3

   from pingouin import test_normality
   # Return a boolean (true if normal) and the associated p-value
   test_normality(x, y)

.. parsed-literal::

   [False,  True], [2.71e-04, 0.552]

------------

5. One-way ANOVA using a pandas DataFrame
#########################################

.. code-block:: ipython3

  # Generate a pandas DataFrame
  import pandas as pd
  np.random.seed(123)
  mean, cov, n = [4, 6], [(1, .6), (.6, 1)], 10
  x, y = np.random.multivariate_normal(mean, cov, n).T
  z = np.random.normal(4, size=n)

  # DV = dependant variable / Group = between-subject factor
  df = pd.DataFrame({'Group': np.repeat(['A', 'B', 'C'], 10),
                     'DV': np.hstack([x, y, z])})

  # One-way ANOVA
  from pingouin import anova
  stats = anova(data=df, dv='DV', between='Group', detailed=True)
  print(stats)

.. table:: Output
  :widths: auto

  ========  ======  ====  ======  =======  =======  =======
  Source        SS    DF      MS        F    p-unc      np2
  ========  ======  ====  ======  =======  =======  =======
  Group     28.995     2  14.498    8.929    0.001    0.398
  Within    43.837    27   1.624  nan      nan      nan
  ========  ======  ====  ======  =======  =======  =======

------------

6. One-way non-parametric ANOVA (Kruskal-Wallis)
################################################

.. code-block:: ipython3

  from pingouin import kruskal
  stats = kruskal(data=df, dv='DV', between='Group')
  print(stats)

.. table:: Output
  :widths: auto

  ========  =======  ======  =======
  Source      ddof1       H    p-unc
  ========  =======  ======  =======
  Group           2  10.622    0.005
  ========  =======  ======  =======

------------

7. Post-hoc tests corrected for multiple-comparisons
####################################################

.. code-block:: ipython3

  from pingouin import pairwise_ttests, print_table

  # FDR-corrected post hocs with Hedges'g effect size
  posthoc = pairwise_ttests(data=df, dv='DV', between='Group', padjust='fdr_bh',
                            effsize='hedges')

  # Pretty printing of table
  print_table(posthoc)

.. table:: Output
  :widths: auto

  =======  ===  ===  ========  =======  =========  =======  ========  ==========  ======  ========  ========
  Type     A    B    Paired      T-val  tail         p-unc    p-corr  p-adjust      BF10    efsize  eftype
  =======  ===  ===  ========  =======  =========  =======  ========  ==========  ======  ========  ========
  between  A    B    False      -3.472  two-sided    0.003     0.004  fdr_bh      13.734    -1.487  hedges
  between  A    C    False      -0.096  two-sided    0.925     0.925  fdr_bh       0.399    -0.041  hedges
  between  B    C    False       3.851  two-sided    0.001     0.004  fdr_bh      26.509     1.650  hedges
  =======  ===  ===  ========  =======  =========  =======  ========  ==========  ======  ========  ========

------------

8. Two-way mixed ANOVA
######################

.. code-block:: ipython3

  # Add a "Time" column in the DataFrame
  df['Time'] = np.tile(np.repeat(['Pre', 'Post'], 5), 3)

  # Compute the two-way mixed ANOVA and export to a .csv file
  from pingouin import mixed_anova
  stats = mixed_anova(data=df, dv='DV', between='Group', within='Time',
                      correction=False, export_filename='mixed_anova.csv')
  print_table(stats)

.. table:: Output
  :widths: auto

  ===========  ======  =====  =====  ======  =====  =======  =====
  Source           SS    DF1    DF2      MS      F    p-unc    np2
  ===========  ======  =====  =====  ======  =====  =======  =====
  Group        28.995      2     12  14.498  8.622    0.005  0.590
  Time          6.839      1     12   6.839  4.995    0.045  0.294
  Interaction   0.391      2     12   0.195  0.143    0.868  0.023
  ===========  ======  =====  =====  ======  =====  =======  =====

------------

9. Pairwise correlations between columns of a dataframe
#######################################################

.. code-block:: ipython3

    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    from pingouin import pairwise_corr
    pairwise_corr(df, columns=['X', 'Y', 'Z'])

.. table:: Output
  :widths: auto

  ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======
  X    Y    method    tail           r  CI95%             r2    adj_r2      z    p-unc    BF10
  ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======
  X    Y    pearson   two-sided  0.707  [0.14 0.92]    0.500     0.357  0.881    0.022   3.227
  X    Z    pearson   two-sided  0.283  [-0.42  0.77]  0.080    -0.183  0.291    0.428   0.321
  Y    Z    pearson   two-sided  0.105  [-0.56  0.69]  0.011    -0.271  0.105    0.772   0.243
  ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======


10. Convert between effect sizes
################################

.. code-block:: ipython3

    from pingouin import convert_effsize
    # Convert from Cohen's d to Hedges' g
    convert_effsize(0.4, 'cohen', 'hedges', nx=10, ny=12)

.. parsed-literal::

    0.384

Contents
========

.. toctree::
   :maxdepth: 1

   api
   changelog
   examples


Development
===========

Pingouin was created and is maintained by `Raphael Vallat <https://raphaelvallat.github.io>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/pingouin>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with another statistical software.

Acknowledgement
===============

Several functions of Pingouin were translated to Python from the original R or Matlab toolboxes, including:

- `effsize package (R) <https://cran.r-project.org/web/packages/effsize/effsize.pdf>`_
- `ezANOVA package (R) <https://cran.r-project.org/web/packages/ez/ez.pdf>`_
- `circular statistics (Matlab) <https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics>`_ (Berens 2009)
- `robust correlations (Matlab) <https://sourceforge.net/projects/robustcorrtool/>`_ (Pernet, Wilcox & Rousselet, 2012)
- `repeated-measure correlation (R) <https://cran.r-project.org/web/packages/rmcorr/index.html>`_ (Bakdash & Marusich, 2017)
