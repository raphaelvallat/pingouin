.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/pingouin.svg
  :target: https://badge.fury.io/py/pingouin

.. image:: https://img.shields.io/conda/vn/conda-forge/pingouin.svg
  :target: https://anaconda.org/conda-forge/pingouin

.. image:: https://img.shields.io/github/license/raphaelvallat/pingouin.svg
  :target: https://github.com/raphaelvallat/pingouin/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/pingouin.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/pingouin

.. image:: https://ci.appveyor.com/api/projects/status/v7fhavoqj8ig1bs2?svg=true
    :target: https://ci.appveyor.com/project/raphaelvallat/pingouin

.. image:: https://codecov.io/gh/raphaelvallat/pingouin/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/pingouin

.. image:: https://pepy.tech/badge/pingouin/month
    :target: https://pepy.tech/badge/pingouin/month

.. image:: http://joss.theoj.org/papers/d2254e6d8e8478da192148e4cfbe4244/status.svg
    :target: http://joss.theoj.org/papers/d2254e6d8e8478da192148e4cfbe4244

.. image:: https://badges.gitter.im/owner/repo.png
    :target: https://gitter.im/pingouin-stats/Lobby

----------------

.. figure::  https://github.com/raphaelvallat/pingouin/blob/master/docs/pictures/logo_pingouin.png
   :align:   center

**Pingouin** is an open-source statistical package written in Python 3 and based mostly on Pandas and NumPy. Some of its main features are listed below. For a full list of available functions, please refer to the `API documentation <https://pingouin-stats.org/api.html>`_.

1. ANOVAs: N-ways, repeated measures, mixed, ancova

2. Pairwise post-hocs tests (parametric and non-parametric) and pairwise correlations

3. Robust, partial, distance and repeated measures correlations

4. Linear/logistic regression and mediation analysis

5. Bayes Factors

6. Multivariate tests

7. Reliability and consistency

8. Effect sizes and power analysis

9. Parametric/bootstrapped confidence intervals around an effect size or a correlation coefficient

10. Circular statistics

11. Chi-squared tests

12. Plotting: Bland-Altman plot, Q-Q plot, paired plot, robust correlation...

Pingouin is designed for users who want **simple yet exhaustive statistical functions**.

For example, the :code:`ttest_ind` function of SciPy returns only the T-value and the p-value. By contrast,
the :code:`ttest` function of Pingouin returns the T-value, the p-value, the degrees of freedom, the effect size (Cohen's d), the 95% confidence intervals of the difference in means, the statistical power and the Bayes Factor (BF10) of the test.

Documentation
=============

- `Link to documentation <https://pingouin-stats.org/index.html>`_

Chat
====

If you have questions, please ask them in the public `Gitter chat <https://gitter.im/pingouin-stats/Lobby>`_

.. image:: https://badges.gitter.im/owner/repo.png
    :target: https://gitter.im/pingouin-stats/Lobby

Installation
============

Dependencies
------------

The main dependencies of Pingouin are :

* NumPy (>= 1.15)
* SciPy (>= 1.3.0)
* Pandas (>= 0.24)
* Pandas-flavor (>= 0.1.2)
* Matplotlib (>= 3.0.2)
* Seaborn (>= 0.9.0)

In addition, some functions require :

* Statsmodels
* Scikit-learn
* Mpmath

Pingouin is a Python 3 package and is currently tested for Python 3.6 and 3.7. Pingouin does not work with Python 2.7.

User installation
-----------------

Pingouin can be easily installed using pip

.. code-block:: shell

  pip install pingouin

or conda

.. code-block:: shell

  conda install -c conda-forge pingouin

New releases are frequent so always make sure that you have the latest version:

.. code-block:: shell

  pip install --upgrade pingouin

Quick start
============

Click on the link below and navigate to the notebooks/ folder to run a collection of interactive Jupyter notebooks showing the main functionalities of Pingouin. No need to install Pingouin beforehand, the notebooks run in a Binder environment.

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/raphaelvallat/pingouin/develop

10 minutes to Pingouin
----------------------

1. T-test
#########

.. code-block:: python

  import numpy as np
  import pingouin as pg

  np.random.seed(123)
  mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
  x, y = np.random.multivariate_normal(mean, cov, n).T

  # T-test
  pg.ttest(x, y)

.. table:: Output
   :widths: auto

   ======  =====  =========  =======  =============  =========  ======  =======
        T    dof  tail         p-val  CI95%            cohen-d    BF10    power
   ======  =====  =========  =======  =============  =========  ======  =======
   -3.401     58  two-sided    0.001  [-1.68 -0.43]      0.878  26.155    0.917
   ======  =====  =========  =======  =============  =========  ======  =======

------------

2. Pearson's correlation
########################

.. code-block:: python

  pg.corr(x, y)

.. table:: Output
   :widths: auto

   ===  =====  ===========  =====  ========  =======  ======  ======
     n      r  CI95%           r2    adj_r2    p-val    BF10   power
   ===  =====  ===========  =====  ========  =======  ======  ======
    30  0.595  [0.3  0.79]  0.354     0.306    0.001  69.723    0.95
   ===  =====  ===========  =====  ========  =======  ======  ======

------------

3. Robust correlation
#####################

.. code-block:: python

  # Introduce an outlier
  x[5] = 18
  # Use the robust Shepherd's pi correlation
  pg.corr(x, y, method="shepherd")

.. table:: Output
   :widths: auto

   ===  =====  ===========  =====  ========  =======  =======
     n      r  CI95%           r2    adj_r2    p-val    power
   ===  =====  ===========  =====  ========  =======  =======
    30  0.561  [0.25 0.77]  0.315     0.264    0.002    0.917
   ===  =====  ===========  =====  ========  =======  =======

------------

4. Test the normality of the data
#################################

The `pingouin.normality` function works with lists, arrays, or pandas DataFrame in wide or long-format.

.. code-block:: python

   print(pg.normality(x))                                    # Univariate normality
   print(pg.multivariate_normality(np.column_stack((x, y)))) # Multivariate normality

.. table:: Output
  :widths: auto

  =====  ======  ========
      W    pval    normal
  =====  ======  ========
  0.615   0.000  False
  =====  ======  ========

.. parsed-literal::

   (False, 0.00018)

------------

5. One-way ANOVA using a pandas DataFrame
#########################################

.. code-block:: python

  # Read an example dataset
  df = pg.read_dataset('mixed_anova')

  # Run the ANOVA
  aov = pg.anova(data=df, dv='Scores', between='Group', detailed=True)
  print(aov)

.. table:: Output
  :widths: auto

  ========  =======  ====  =====  =====  =======  =====
  Source         SS    DF     MS  F      p-unc    np2
  ========  =======  ====  =====  =====  =======  =====
  Group       5.460     1  5.460  5.244  0.02320  0.029
  Within    185.343   178  1.041  -      -        -
  ========  =======  ====  =====  =====  =======  =====

------------

6. Repeated measures ANOVA
##########################

.. code-block:: python

  pg.rm_anova(data=df, dv='Scores', within='Time', subject='Subject', detailed=True)

.. table:: Output
  :widths: auto

  ========  =======  ====  =====  =====  ========  =====  =====
  Source         SS    DF     MS  F      p-unc     np2    eps
  ========  =======  ====  =====  =====  ========  =====  =====
  Time        7.628     2  3.814  3.913  0.022629  0.062  0.999
  Error     115.027   118  0.975  -      -         -      -
  ========  =======  ====  =====  =====  ========  =====  =====

------------

7. Post-hoc tests corrected for multiple-comparisons
####################################################

.. code-block:: python

  # FDR-corrected post hocs with Hedges'g effect size
  posthoc = pg.pairwise_ttests(data=df, dv='Scores', within='Time', subject='Subject',
                               parametric=True, padjust='fdr_bh', effsize='hedges')

  # Pretty printing of table
  pg.print_table(posthoc, floatfmt='.3f')

.. table:: Output
  :widths: auto

  ==========  =======  =======  ========  ============  ======  ======  =========  =======  ========  ==========  ======  ======  ========
  Contrast    A        B        Paired    Parametric         T     dof  tail         p-unc    p-corr  p-adjust      BF10    CLES    hedges
  ==========  =======  =======  ========  ============  ======  ======  =========  =======  ========  ==========  ======  ======  ========
  Time        August   January  True      True          -1.740  59.000  two-sided    0.087     0.131  fdr_bh       0.582   0.585    -0.328
  Time        August   June     True      True          -2.743  59.000  two-sided    0.008     0.024  fdr_bh       4.232   0.644    -0.485
  Time        January  June     True      True          -1.024  59.000  two-sided    0.310     0.310  fdr_bh       0.232   0.571    -0.170
  ==========  =======  =======  ========  ============  ======  ======  =========  =======  ========  ==========  ======  ======  ========

------------

8. Two-way mixed ANOVA
######################

.. code-block:: python

  # Compute the two-way mixed ANOVA and export to a .csv file
  aov = pg.mixed_anova(data=df, dv='Scores', between='Group', within='Time',
                       subject='Subject', correction=False,
                       export_filename='mixed_anova.csv')
  pg.print_table(aov)

.. table:: Output
  :widths: auto

  ===========  =====  =====  =====  =====  =====  =======  =====  =====
  Source          SS    DF1    DF2     MS      F    p-unc    np2  eps
  ===========  =====  =====  =====  =====  =====  =======  =====  =====
  Group        5.460      1     58  5.460  5.052    0.028  0.080  -
  Time         7.628      2    116  3.814  4.027    0.020  0.065  0.999
  Interaction  5.168      2    116  2.584  2.728    0.070  0.045  -
  ===========  =====  =====  =====  =====  =====  =======  =====  =====

------------

9. Pairwise correlations between columns of a dataframe
#######################################################

.. code-block:: python

  import pandas as pd
  np.random.seed(123)
  z = np.random.normal(5, 1, 30)
  data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
  pg.pairwise_corr(data, columns=['X', 'Y', 'Z'])

.. table:: Output
  :widths: auto

  ===  ===  ========  =========  ===  =====  =============  =====  ========  =====  =======  ======  =======
  X    Y    method    tail         n      r  CI95%             r2    adj_r2      z    p-unc    BF10    power
  ===  ===  ========  =========  ===  =====  =============  =====  ========  =====  =======  ======  =======
  X    Y    pearson   two-sided   30  0.366  [0.01 0.64]    0.134     0.070  0.384    0.047   1.500    0.525
  X    Z    pearson   two-sided   30  0.251  [-0.12  0.56]  0.063    -0.006  0.256    0.181   0.534    0.272
  Y    Z    pearson   two-sided   30  0.020  [-0.34  0.38]  0.000    -0.074  0.020    0.916   0.228    0.051
  ===  ===  ========  =========  ===  =====  =============  =====  ========  =====  =======  ======  =======

10. Convert between effect sizes
################################

.. code-block:: python

    # Convert from Cohen's d to Hedges' g
    pg.convert_effsize(0.4, 'cohen', 'hedges', nx=10, ny=12)

.. parsed-literal::

    0.384

11. Multiple linear regression
##############################

.. code-block:: python

    pg.linear_regression(data[['X', 'Z']], data['Y'])

.. table:: Linear regression summary
  :widths: auto

  =========  ======  =====  ======  ======  =====  ========  ==========  ===========
  names        coef     se       T    pval     r2    adj_r2    CI[2.5%]    CI[97.5%]
  =========  ======  =====  ======  ======  =====  ========  ==========  ===========
  Intercept   4.650  0.841   5.530   0.000  0.139     0.076       2.925        6.376
  X           0.143  0.068   2.089   0.046  0.139     0.076       0.003        0.283
  Z          -0.069  0.167  -0.416   0.681  0.139     0.076      -0.412        0.273
  =========  ======  =====  ======  ======  =====  ========  ==========  ===========

12. Mediation analysis
######################

.. code-block:: python

    pg.mediation_analysis(data=data, x='X', m='Z', y='Y', seed=42, n_boot=1000)

.. table:: Mediation summary
  :widths: auto

  ========  ======  =====  ======  ==========  ===========  =====
  path        coef     se    pval    CI[2.5%]    CI[97.5%]  sig
  ========  ======  =====  ======  ==========  ===========  =====
  Z ~ X      0.103  0.075   0.181      -0.051        0.256  No
  Y ~ Z      0.018  0.171   0.916      -0.332        0.369  No
  Total      0.136  0.065   0.047       0.002        0.269  Yes
  Direct     0.143  0.068   0.046       0.003        0.283  Yes
  Indirect  -0.007  0.025   0.898      -0.070        0.029  No
  ========  ======  =====  ======  ==========  ===========  =====

13. Contingency analysis
########################

.. code-block:: python

    data = pg.read_dataset('chi2_independence')
    expected, observed, stats = pg.chi2_independence(data, x='sex', y='target')
    stats

.. table:: Chi-squared tests summary
  :widths: auto

  ==================  ========  ======  =====  =====  ========  =======
  test                  lambda    chi2    dof      p    cramer    power
  ==================  ========  ======  =====  =====  ========  =======
  pearson                1.000  22.717  1.000  0.000     0.274    0.997
  cressie-read           0.667  22.931  1.000  0.000     0.275    0.998
  log-likelihood         0.000  23.557  1.000  0.000     0.279    0.998
  freeman-tukey         -0.500  24.220  1.000  0.000     0.283    0.998
  mod-log-likelihood    -1.000  25.071  1.000  0.000     0.288    0.999
  neyman                -2.000  27.458  1.000  0.000     0.301    0.999
  ==================  ========  ======  =====  =====  ========  =======

Integration with Pandas
-----------------------

Several functions of Pingouin can be used directly as pandas DataFrame methods. Try for yourself with the code below:

.. code-block:: python

  import pingouin as pg

  # Example 1 | ANOVA
  df = pg.read_dataset('mixed_anova')
  df.anova(dv='Scores', between='Group', detailed=True)

  # Example 2 | Pairwise correlations
  data = pg.read_dataset('mediation')
  data.pairwise_corr(columns=['X', 'M', 'Y'], covar=['Mbin'])

  # Example 3 | Partial correlation matrix
  data.pcorr()

The functions that are currently supported as pandas method are:

* `pingouin.anova <https://pingouin-stats.org/generated/pingouin.anova.html#pingouin.anova>`_
* `pingouin.ancova <https://pingouin-stats.org/generated/pingouin.ancova.html#pingouin.ancova>`_
* `pingouin.rm_anova <https://pingouin-stats.org/generated/pingouin.rm_anova.html#pingouin.rm_anova>`_
* `pingouin.mixed_anova <https://pingouin-stats.org/generated/pingouin.mixed_anova.html#pingouin.mixed_anova>`_
* `pingouin.welch_anova <https://pingouin-stats.org/generated/pingouin.welch_anova.html#pingouin.welch_anova>`_
* `pingouin.pairwise_ttests <https://pingouin-stats.org/generated/pingouin.pairwise_ttests.html#pingouin.pairwise_ttests>`_
* `pingouin.pairwise_ttests <https://pingouin-stats.org/generated/pingouin.pairwise_tukey.html#pingouin.pairwise_tukey>`_
* `pingouin.pairwise_corr <https://pingouin-stats.org/generated/pingouin.pairwise_corr.html#pingouin.pairwise_corr>`_
* `pingouin.partial_corr <https://pingouin-stats.org/generated/pingouin.partial_corr.html#pingouin.partial_corr>`_
* `pingouin.pcorr <https://pingouin-stats.org/generated/pingouin.pcorr.html#pingouin.pcorr>`_
* `pingouin.rcorr <https://pingouin-stats.org/generated/pingouin.rcorr.html#pingouin.rcorr>`_
* `pingouin.mediation_analysis <https://pingouin-stats.org/generated/pingouin.mediation_analysis.html#pingouin.mediation_analysis>`_

Development
===========

Pingouin was created and is maintained by `Raphael Vallat <https://raphaelvallat.github.io>`_, mostly during his spare time. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/pingouin>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with another statistical software.

Contributors
------------

- Nicolas Legrand
- `Richard Höchenberger <http://hoechenberger.net/>`_
- `Arthur Paulino <https://github.com/arthurpaulino>`_

How to cite Pingouin?
=====================

If you want to cite Pingouin, please use the publication in JOSS:

Vallat, R. (2018). Pingouin: statistics in Python. *Journal of Open Source Software*, 3(31), 1026, `https://doi.org/10.21105/joss.01026 <https://doi.org/10.21105/joss.01026>`_

.. code-block:: latex

  @ARTICLE{Vallat2018,
    title    = "Pingouin: statistics in Python",
    author   = "Vallat, Raphael",
    journal  = "The Journal of Open Source Software",
    volume   =  3,
    number   =  31,
    pages    = "1026",
    month    =  nov,
    year     =  2018
  }

Acknowledgement
===============

Several functions of Pingouin were inspired from R or Matlab toolboxes, including:

- `effsize package (R) <https://cran.r-project.org/web/packages/effsize/effsize.pdf>`_
- `ezANOVA package (R) <https://cran.r-project.org/web/packages/ez/ez.pdf>`_
- `pwr package (R) <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_
- `circular statistics (Matlab) <https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics>`_ (Berens 2009)
- `robust correlations (Matlab) <https://sourceforge.net/projects/robustcorrtool/>`_ (Pernet, Wilcox & Rousselet, 2012)
- `repeated-measure correlation (R) <https://cran.r-project.org/web/packages/rmcorr/index.html>`_ (Bakdash & Marusich, 2017)

I am also grateful to Charles Zaiontz and his website `www.real-statistics.com <https://www.real-statistics.com/>`_ which has been useful to
understand the practical implementation of several functions.
