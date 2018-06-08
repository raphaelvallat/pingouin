.. examples_bayesian_ttest:

Performing a Bayesian T-test
============================

In this tutorial we will see how to compute a classical and Bayesian T-test in Pingouin using the `ttest <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.ttest.html#pingouin.ttest>`_ function.

The method used in Pingouin is derived from `Rouder et al. 2009 <http://pcl.missouri.edu/sites/default/files/Rouder.bf_.pdf>`_ and uses the recommended Cauchy prior distribution on effect size, centered around 0 and with a width (r) factor of 0.707. This default values applies well for most cases (at least for psychological studies with typically small to moderate effect sizes). In simple words, it means that you expect (i.e. your prior) that the effect size will most likely fall between -.707 and +.707. Note that this parameter can be adjusted by using the :code:`r` optional argument function. Use smaller values if you expect smaller effect sizes, and larger values if you expect larger effect sizes.

For the sake of this example, we will generate a fake drug / placebo study with 30 subjects in each group.
We are interested in comparing the memory performances after administration of the drug or the placebo.

.. code:: ipython3

    import numpy as np
    np.random.seed(123)
    drug = np.random.normal(loc=1, size=30)
    placebo = np.random.normal(loc=0, size=30)

Then to compute the T-test, simply:

.. code:: ipython3

    from pingouin import ttest
    ttest(drug, placebo)

.. table:: Output
   :widths: auto

   ======  =====  ===  =========  ======= ===== ====
   T -val  p-val  dof  tail       cohen-d power BF10
   ======  =====  ===  =========  ======= ===== ====
   2.891   0.005  58   two-sided  0.746   0.811 7.71
   ======  =====  ===  =========  ======= ===== ====

The p-value is significantly inferior to 0.05, meaning that we can reject the null hypothesis that the two groups have similar performances. However, the p-value can simply be used to reject or accept the null hypothesis, given an arbitrary threshold. It does not allow to quantify *per se* the evidence in favor of the alternative hypothesis (which is that the groups have different performances).

To do so, we need to rely on the Bayes Factor (:code:`BF10`), first introduced by Jeffreys in 1961. The Bayes Factor is an odds ratio and has therefore an intuitive interpretation: in the example above, it means that the **alternative hypothesis is 7.71 times more likely than the null hypothesis**, given the data. To quantify the odds in favor of the null hypothesis (BF01), we can simply compute the reciprocal of the BF10:

.. code:: ipython3

    BF01 = 1 / BF10 = 1 / 7.71 = 0.13

As a rule of thumbs, Jeffreys (1961) recommended that a Bayes Factor greater than 3 be considered *“some evidence”*, greater than 10 *“strong evidence”*, and greater than 30 *“very strong evidence”*. In the example above, it means that there are a moderate to strong evidence that the drug actually works.

------------

Note that the `ttest <https://raphaelvallat.github.io/pingouin/build/html/generated/pingouin.ttest.html#pingouin.ttest>`_ function has several optional arguments. For instance, if the data come from a single group (paired measurements), and if we have an a priori that the drug will indeed work, we could adjust the :code:`tail` and :code:`paired` arguments:

.. code:: ipython3

    ttest(drug, placebo, paired=True, tail='one-sided')

.. table:: Output
   :widths: auto

   ======  =====  ===  =========  ======= ===== ====
   T -val  p-val  dof  tail       cohen-d power BF10
   ======  =====  ===  =========  ======= ===== ====
   2.717   0.005  29   one-sided  0.759   0.992 8.31
   ======  =====  ===  =========  ======= ===== ====
