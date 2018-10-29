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

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/raphaelvallat/pingouin/master


.. figure::  /pictures/logo_pingouin.png
  :align:   center

**Pingouin** is an open-source statistical package written in Python 3 and based on Pandas and NumPy.

1. ANOVAs: one- and two-ways, repeated measures, mixed, ancova

2. Post-hocs tests and pairwise comparisons

3. Robust correlations

4. Partial correlation, repeated measures correlation and intraclass correlation

5. Bayes Factor

6. Tests for sphericity, normality and homoscedasticity

7. Effect sizes

8. Circular statistics

9. Linear/logistic regression and mediation analysis

Pingouin is designed for users who want **simple yet exhaustive statistical functions**.

For example, the :code:`ttest_ind` function of SciPy returns only the T-value and the p-value. By contrast,
the :code:`ttest` function of Pingouin returns the t-value, p-value, degrees of freedom, effect size (Cohen's d), statistical power and Bayes Factor (BF10) of the test.


Installation
============

Dependencies
------------

The main dependencies of Pingouin are :

- NumPy
- SciPy
- Pandas

In addition, some functions require :

* Statsmodels
* Scikit-learn


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

New releases are frequent so always make sure that you have the latest version:

.. code-block:: shell

  pip install --upgrade pingouin

Quick start
============

Try before you buy! Click on the link below and navigate to the notebooks folder to load a collection of interactive Jupyter notebooks demonstrating the main functionalities of Pingouin. No need to install Pingouin beforehand as the notebooks run in a Binder environment.

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/raphaelvallat/pingouin/develop

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
   0.561  [0.25 0.77]  0.315     0.264    0.002
   =====  ===========  =====  ========  =======

------------

4. Test the normality of the data
#################################

.. code-block:: python

   from pingouin import normality, multivariate_normality
   # Return a boolean (true if normal) and the associated p-value
   print(normality(x, y))                                 # Univariate normality
   print(multivariate_normality(np.column_stack((x, y)))) # Multivariate normality

.. parsed-literal::

   (array([False,  True]), array([0., 0.552]))
   (False, 0.00018)

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
  Within    43.837    27   1.624
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
  # Create a subject identifier column
  df['Subject'] = np.r_[np.tile(np.arange(5), 2), np.tile(np.arange(5, 10), 2),
                        np.tile(np.arange(10, 15), 2)]

  # Compute the two-way mixed ANOVA and export to a .csv file
  from pingouin import mixed_anova
  stats = mixed_anova(data=df, dv='DV', between='Group', within='Time',
                      subject='Subject', correction=False,
                      export_filename='mixed_anova.csv')
  print_table(stats)

.. table:: Output
  :widths: auto

  ===========  ======  =====  =====  ======  =====  =======  =====  ===
  Source           SS    DF1    DF2      MS      F    p-unc    np2  eps
  ===========  ======  =====  =====  ======  =====  =======  =====  ===
  Group        28.995      2     12  14.498  8.622    0.005  0.590
  Time          6.839      1     12   6.839  4.995    0.045  0.294  1.0
  Interaction   0.391      2     12   0.195  0.143    0.868  0.023
  ===========  ======  =====  =====  ======  =====  =======  =====  ===

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

11. Multiple linear regression
##############################

.. code-block:: ipython3

    from pingouin import linear_regression
    lm_dict = linear_regression(df[['X', 'Z']], df['Y'])
    pd.DataFrame.from_dict(lm_dict)

.. table:: Linear regression summary
  :widths: auto

  =========  ======  =====  =======  =======  =====  ========  ======  =====
  names        coef     se    tvals    pvals     r2    adj_r2      ll     ul
  =========  ======  =====  =======  =======  =====  ========  ======  =====
  Intercept   3.855  1.417    2.720    0.030  0.510     0.370   0.504  7.205
  X           0.673  0.252    2.669    0.032  0.510     0.370   0.077  1.269
  Z          -0.124  0.331   -0.375    0.719  0.510     0.370  -0.906  0.658
  =========  ======  =====  =======  =======  =====  ========  ======  =====

12. Mediation analysis
######################

.. code-block:: ipython3

    from pingouin import mediation_analysis
    mediation_analysis(data=df, x='X', m='Z', y='Y', n_boot=500)

.. table:: Mediation summary
  :widths: auto

  ========  ======  ==========  ===========  =====
  Path        Beta    CI[2.5%]    CI[97.5%]  Sig
  ========  ======  ==========  ===========  =====
  X -> M     0.216      -0.380        0.812  No
  M -> Y     0.126      -0.846        1.099  No
  X -> Y     0.646       0.119        1.173  Yes
  Direct     0.673       0.077        1.270  Yes
  Indirect  -0.027      -0.485        0.153  No
  ========  ======  ==========  ===========  =====

Contents
========

.. toctree::
   :maxdepth: 1

   api
   guidelines
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

I am also grateful to Charles Zaiontz and his website `www.real-statistics.com <https://www.real-statistics.com/>`_ which has been useful to
understand the practical implementation of several functions.
