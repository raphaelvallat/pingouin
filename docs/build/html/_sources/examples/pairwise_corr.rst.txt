.. examples_pairwise_corr:

===========================================
Pairwise correlations in a pandas dataframe
===========================================

In this tutorial we will see how to compute pairwise correlations coefficients across columns of a pandas DataFrame using the `pairwise_corr <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.pairwise_corr.html#pingouin.pairwise_corr>`_ function.

We'll start by generating a pandas dataframe with three continuous variables, each in a separate column.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    np.random.seed(123)
    n = 20
    raw = np.random.normal(size=n)
    df = pd.DataFrame({'X': raw,
                       'Y': raw + np.random.normal(size=n),
                       'Z': np.random.random(size=n)})

Let's compute the pairwise correlations between all the columns of the DataFrame:

.. code:: ipython3

    from pingouin import pairwise_corr
    pairwise_corr(df)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======
   X    Y    method    tail           r  CI95%             r2    adj_r2      z    p-unc    BF10
   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======
   X    Y    pearson   two-sided  0.716  [0.08 1.36]    0.513     0.456  0.899    0.000  86.949
   X    Z    pearson   two-sided  0.233  [-0.39  0.85]  0.054    -0.057  0.237    0.323   0.278
   Y    Z    pearson   two-sided  0.002  [-0.62  0.62]  0.000    -0.118  0.002    0.995   0.171
   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ======

By default, the function returns the two-sided Pearson's correlation coefficients. This can be adjusted using the :code:`tail` and :code:`method` arguments. In addition, the table comprises:

1. the parametric 95% confidence intervals of the r value (:code:`CI95%`)
2. the R\ :sup:`2` (= coefficient of determination, :code:`r2`)
3. the adjusted R\ :sup:`2` (:code:`adj_r2`)
4. the standardized (Z-transformed) correlation coefficients (:code:`z`)
5. the uncorrected p-values (:code:`p-unc`)
6. the Bayes Factor for the alternative hypothesis (:code:`BF10`)

In the example above, we can see that there is a strong correlation between variables :code:`X` and :code:`Y`, as indicated by the correlation coefficient (0.716), the p-value (<.001) and the Bayes Factor (86.9, meaning that the alternative hypothesis is ~87 times more likely than the null hypothesis given the data).

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
   X    Y    spearman  one-sided  0.662  [0.03 1.3 ]  0.438     0.372  0.796    0.001
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======

*Note that the Bayes Factor is only computed when using the Pearson method and is therefore not present in the table above.*

------------

Robust correlation
------------------

If you believe that your dataset contains outliers, you can use a robust correlation method (percentage bend, see `Wilcox 1994 <https://link.springer.com/article/10.1007/BF02294395>`_:)

.. code:: ipython3

    from pingouin import print_table
    stats = pairwise_corr(data=df, columns=['X', 'Y'], method='percbend')
    print_table(stats)

.. table:: Output
   :widths: auto

   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    method    tail           r  CI95%           r2    adj_r2      z    p-unc
   ===  ===  ========  =========  =====  ===========  =====  ========  =====  =======
   X    Y    percbend  two-sided  0.730  [0.09 1.37]  0.533     0.478  0.929    0.000
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

   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ========  ==========  ======
   X    Y    method    tail           r  CI95%             r2    adj_r2      z    p-unc    p-corr  p-adjust      BF10
   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ========  ==========  ======
   X    Y    pearson   two-sided  0.716  [0.08 1.36]    0.513     0.456  0.899    0.000     0.001  fdr_bh      86.949
   X    Z    pearson   two-sided  0.233  [-0.39  0.85]  0.054    -0.057  0.237    0.323     0.484  fdr_bh       0.278
   Y    Z    pearson   two-sided  0.002  [-0.62  0.62]  0.000    -0.118  0.002    0.995     0.995  fdr_bh       0.171
   ===  ===  ========  =========  =====  =============  =====  ========  =====  =======  ========  ==========  ======
