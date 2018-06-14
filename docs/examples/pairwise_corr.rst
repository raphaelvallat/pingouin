.. examples_pairwise_corr:

===========================================================
Calculating the pairwise correlations in a pandas dataframe
===========================================================

In this tutorial we will see how to compute pairwise correlations coefficients across columns of a pandas DataFrame using the `pairwise_corr <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.pairwise_corr.html#pingouin.pairwise_corr>`_ function.

We'll start by generating a pandas dataframe with three continuous variables, each in a separate column.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    np.random.seed(123)
    n = 20
    mean, cov = [4, 6], [(1, .6), (.6, 1)]
    x, y = np.random.multivariate_normal(mean, cov, n).T
    z = np.random.normal(size=n)
    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

Let's compute the pairwise correlations between all the columns of the DataFrame:

.. code:: ipython3

    from pingouin import pairwise_corr
    pairwise_corr(df)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ======
   X    Y    method    tail            r  CI95%             r2    adj_r2       z    p-unc    BF10
   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ======
   X    Y    pearson   two-sided   0.583  [0.19 0.82]    0.340     0.262   0.667    0.007   6.270
   X    Z    pearson   two-sided  -0.083  [-0.51  0.37]  0.007    -0.110  -0.083    0.729   0.181
   Y    Z    pearson   two-sided  -0.197  [-0.59  0.27]  0.039    -0.074  -0.200    0.404   0.241
   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ======

By default, the function returns the two-sided Pearson's correlation coefficients. This can be adjusted using the :code:`tail` and :code:`method` arguments. In addition, the table comprises:

1. the parametric 95% confidence intervals of the r value (:code:`CI95%`)
2. the R\ :sup:`2` (= coefficient of determination, :code:`r2`)
3. the adjusted R\ :sup:`2` (:code:`adj_r2`)
4. the standardized (Z-transformed) correlation coefficients (:code:`z`)
5. the uncorrected p-values (:code:`p-unc`)
6. the Bayes Factor for the alternative hypothesis (:code:`BF10`)

In the example above, we can see that there is a strong correlation between variables :code:`X` and :code:`Y`, as indicated by the correlation coefficient (0.583), the p-value (.007) and the Bayes Factor (6.27, meaning that the alternative hypothesis is ~6 times more likely than the null hypothesis given the data).

------------

Non-parametric correlation
--------------------------

If your data do not follow a normal distribution, the software will display a warning message suggesting you to use a non-parametric method such as the Spearman rank-correlation.
In the example below, we compute the one-sided Spearman pairwise correlations between a subset of columns:

.. code:: ipython3

    from pingouin import print_table
    stats = pairwise_corr(data=df, columns=['X', 'Y'], tail='one-sided', method='spearman')
    print_table(stats)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    method    tail           r  CI95%           r2    adj_r2      z    p-unc
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    spearman  one-sided  0.537  [0.12 0.79]  0.288     0.204  0.600    0.007
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======

*Note that the Bayes Factor is only computed when using the Pearson method and is therefore not present in the table above.*

------------

Robust correlation
------------------

If you believe that your dataset contains outliers, you can use a robust correlation method. There are currently two robust correlation methods implemented in Pingouin, namely the percentage bend correlation (`Wilcox 1994 <https://link.springer.com/article/10.1007/BF02294395>`_) and Shepherd's pi correlation (`Schwarzkopf et al. (2012). <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3397314/`_).

.. code:: ipython3

    # Introduce two outliers in variable X
    df.loc[[5, 12], 'X'] = 18
    stats = pairwise_corr(data=df, columns=['X', 'Y'], method='percbend')
    print_table(stats)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    method    tail           r  CI95%           r2    adj_r2      z    p-unc
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    percbend  two-sided  0.560  [0.16 0.8 ]  0.313     0.232  0.633    0.010
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======

.. code:: ipython3

   stats = pairwise_corr(data=df, columns=['X', 'Y'], method='shepherd')
   print_table(stats)

.. table:: Output
  :widths: auto

  ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
  X    Y    method    tail           r  CI95%           r2    adj_r2      z    p-unc
  ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
  X    Y    shepherd  two-sided  0.507  [0.08 0.78]  0.257     0.169  0.559    0.064
  ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======

------------

Correction for multiple comparisons
-----------------------------------

Finally, if you are computing a large number of correlation coefficients, you might want to correct the p-values for multiple comparisons. This can be done with :code:`padjust` argument:

.. code:: ipython3

    stats = pairwise_corr(df, padjust="fdr_bh")
    print_table(stats)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ========  ==========
   X    Y    method    tail            r  CI95%             r2    adj_r2       z    p-unc    p-corr  p-adjust
   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ========  ==========
   X    Y    shepherd  two-sided   0.507  [0.08 0.78]    0.257     0.169   0.559    0.064     0.191  fdr_bh
   X    Z    shepherd  two-sided  -0.148  [-0.55  0.32]  0.022    -0.093  -0.149    1.000     1.000  fdr_bh
   Y    Z    shepherd  two-sided  -0.330  [-0.67  0.13]  0.109     0.004  -0.343    0.336     0.504  fdr_bh
   ===  ===  ========  =========  ======  =============  =====  ========  ======  =======  ========  ==========
