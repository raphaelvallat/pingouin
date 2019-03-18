.. _Guidelines:

Guidelines
##########

In this page, you will find a collection of flowcharts designed to help you choose
which functions of `Pingouin` are adequate for your analysis. Click on
the desired flowchart to view a full scale image with hyperlinks.

.. contents:: Table of Contents
   :depth: 2

One-way ANOVA
-------------

.. figure::  /pictures/flowchart/flowchart_one_way_ANOVA.svg
  :align: center
  :scale: 50
  :alt: One-way ANOVA

Example code
~~~~~~~~~~~~

.. code:: python

  import pingouin as pg

  # Load an example dataset comparing pain threshold as a function of hair color
  df = pg.read_dataset('anova')

  # 1. This is a between subject design, so the first step is to test for equality of variances
  groups = df['Hair color'].unique()
  a, b, c, d = [df.groupby('Hair color')['Pain threshold'].get_group(g).values for g in groups]
  equal_var, pval = pg.homoscedasticity(a, b, c, d)

  # 2. If the groups have equal variances, we can use a regular one-way ANOVA
  pg.anova(data=df, dv='Pain threshold', between='Hair color')

  # 3. If there is a main effect, we can proceed to post-hoc Tukey test
  pg.pairwise_tukey(data=df, dv='Pain threshold', between='Hair color')


|

Correlation
-----------

.. figure::  /pictures/flowchart/flowchart_correlations.svg
  :align: center
  :scale: 50
  :alt: Correlations

Example code
~~~~~~~~~~~~

.. code:: python

  import pingouin as pg
  import seaborn as sns

  # Load an example dataset with the personality scores of 500 participants
  df = pg.read_dataset('pairwise_corr')

  # 1.Test for bivariate normality
  print(multivariate_normality(df[['Neuroticism', 'Openness']]))

  # 1bis. Visual inspection with a histogram + scatter plot
  sns.jointplot(data=df, x='Neuroticism', y='Openness', kind='reg')

  # 2. If the data have a bivariate normal distribution and no clear outlier(s), we can use a regular Pearson correlation
  pg.corr(df['Neuroticism'], df['Openness'], method='pearson')


|

Non-parametric
--------------

.. figure::  /pictures/flowchart/flowchart_nonparametric.svg
  :align: center
  :scale: 50
  :alt: Non-parametric tests

Example code
~~~~~~~~~~~~

.. code:: python

  import pingouin as pg

  # Load an example dataset comparing pain threshold as a function of hair color
  df = pg.read_dataset('anova')

  # There are 4 independent groups in our dataset, we'll therefore use the Kruskal-Wallis test:
  pg.kruskal(data=df, dv='Pain threshold', between='Hair color')
