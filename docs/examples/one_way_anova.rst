.. examples_one_way_anova:

==========================
Performing a one-way ANOVA
==========================

The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. In Pingouin, the one-way ANOVA is implemented in the `anova <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.anova.html>`_ function. The ANOVA test has several assumptions that must be satisfied to provide accurate results:

1. The samples must be independent (i.e. by opposition with repeated measurements in a single group, see `rm_anova <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.rm_anova.html>`_)
2. Each sample should be normally distributed
3. The variance of the samples are all equal (= homoscedasticity)

Assumptions #2 and #3 can be checked using the `test_normality <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.test_normality.html>`_ and `test_homoscedasticity <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.test_homoscedasticity.html>`_ functions.

We'll start by generating a pandas dataframe with one column containing the dependent variable (:code:`DV`) and one column containing the factor (:code:`Group`):

.. code:: ipython3

    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'dv': [0.99, -0.42, 0.66, -0.61, -0.12, -0.41, 1.43, -0.55,
                              0.08, 1.36, 0.95, -0.09, 3.79, 1.86, 2.19, 3.7, 1.46, 1.91],
                       'Group': np.repeat(['A', 'B', 'C'], 6),
                       })
    # Print group means and standard deviations
    print(df.groupby('Group').agg(['mean', 'std']))

.. table:: Group means and STD
   :widths: auto

   =====  =====  =====
   Group  mean   std
   =====  =====  =====
   A      0.015  0.655
   B      0.530  0.828
   C      2.485  1.004
   =====  =====  =====

Let's now compute the one-way ANOVA:

.. code:: ipython3

    from pingouin import anova
    aov = anova(data=df, dv='dv', between='Group')
    print(aov)

.. table:: ANOVA summary
   :widths: auto

   ========  =======  =======  ======  =======  =====
   Source      ddof1    ddof2       F    p-unc    np2
   ========  =======  =======  ======  =======  =====
   Group           2       15  14.401    0.000  0.658
   ========  =======  =======  ======  =======  =====

And to get a more detailed summary table:

.. code:: ipython3

   from pingouin import anova
   aov = anova(data=df, dv='dv', between='Group', detailed=True)
   print(aov)

.. table:: ANOVA summary
  :widths: auto

  ========  ======  ====  ======  =======  =======  =======
  Source        SS    DF      MS        F    p-unc      np2
  ========  ======  ====  ======  =======  =======  =======
  Group     20.376     2  10.188  14.401   0.000    0.658
  Within    10.612    15   0.707  nan      nan      nan
  ========  ======  ====  ======  =======  =======  =======

The detailed ANOVA summary table includes the following columns:

1. :code:`SS` : sums of squares
2. :code:`DF` : degrees of freedom
3. :code:`MS` : mean squares (= SS / DF)
4. :code:`F` : F-value (test statistic)
5. :code:`p-unc` : uncorrected p-values
6. :code:`np2` : partial eta-square effect size\*

\* *In one-way ANOVA, partial eta-square is the same as eta-square and generalized eta-square.*

In the example above, there is a main effect of group (F(2, 15) = 14, p < .001)), so we can reject the null hypothesis that the groups have equal means.

------------

Post-hoc tests
--------------

Often, you will want to compute post-hoc tests to look at the pairwise differences between the groups. This can be achieved using the `pairwise_ttests <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.pairwise_ttests.html>`_ function:

.. code:: ipython3

    from pingouin import pairwise_ttests
    posthocs = pairwise_ttests(data=df, dv='dv', between='Group')
    print(posthocs)

.. table:: Post-hoc T-tests
   :widths: auto

   =======  ===  ===  ========  =======  =========  =======  ======  ========  ========
   Type     A    B    Paired      T-val  tail         p-unc    BF10    efsize  eftype
   =======  ===  ===  ========  =======  =========  =======  ======  ========  ========
   between  A    B    False      -1.195  two-sided    0.260   0.725    -0.637  hedges
   between  A    C    False      -5.048  two-sided    0.001  48.277    -2.690  hedges
   between  B    C    False      -3.680  two-sided    0.004   9.570    -1.961  hedges
   =======  ===  ===  ========  =======  =========  =======  ======  ========  ========

Looking at the table above, it is clear that the mean of group C is significantly higher than the mean of group A and group B. Note that the :code:`pairwise_ttests` function accepts several optional arguments allowing you to specify the tail of the test, whether to correct the p-values for multiple comparisons, the type of effect size and so on. Take a look at the `API <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.pairwise_ttests.html>`_ for more details.

------------

Power
-----

In some cases, it might be useful to compute the power of the test, i.e. the probability that we correctly reject the null hypothesis when it is indeed false (with higher power indicating higher reliability). This can be calculated easily from the ANOVA summary using the `anova_power <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.anova_power.html>`_ function:

.. code:: ipython3

    from pingouin import anova_power
    # eta = 'np2' column of the ANOVA summary
    achieved_power = anova_power(eta=0.658, ntot=18, ngroups=3)
    print(achieved_power)

.. parsed-literal::

    0.998

------------

Assumptions check
-----------------

Finally, to check that (1) each sample is normally distributed and (2) the variance of the samples are all equal, we can use the `test_normality <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.test_normality.html>`_ and `test_homoscedasticity <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.test_homoscedasticity.html>`_ functions, respectively:

.. code:: ipython3

    from pingouin import test_normality
    for group in df['Group'].unique():
        print(test_normality(df[df['Group'] == group]['dv'].values))

.. parsed-literal::

    (True, 0.185) # Sample 1 is normally distributed
    (True, 0.375) # idem
    (True, 0.098) # idem

.. code:: ipython3

    from pingouin import test_homoscedasticity
    equal_var, p = test_homoscedasticity(df.iloc[0:6, 0].values, df.iloc[6:12, 0].values, df.iloc[12:18, 0].values)
    print(equal_var, p)

.. parsed-literal::

    True 0.665 # The variance of the samples are all equal
