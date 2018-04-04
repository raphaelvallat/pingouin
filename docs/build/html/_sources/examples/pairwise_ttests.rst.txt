
Post-hoc comparisons
====================


Load a fake dataset: the INSOMNIA study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal: evaluate the influence of a treatment on sleep duration in a
control and insomnia group

**Mixed repeated measures design:**

-  Dependant variable (DV) = hours of sleep per night
-  Between-factor = two-levels (Insomnia / Control)
-  Within-factor = three levels (Pre, Post1, Post2)

.. code:: ipython3

    import pandas as pd
    # Change default display format of pandas
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

.. code:: ipython3

    df = pd.read_csv('sleep_dataset.csv')
    print(df.head())


.. parsed-literal::

         DV     Group Time
    0 5.682  Insomnia  Pre
    1 5.286  Insomnia  Pre
    2 4.890  Insomnia  Pre
    3 6.257  Insomnia  Pre
    4 4.615  Insomnia  Pre


.. code:: ipython3

    %matplotlib inline
    import seaborn as sns
    sns.pointplot(x='Time', y='DV', hue='Group', data=df, dodge=True)


.. image:: pairwise_ttests_seaborn.png


Pairwise comparisons
--------------------

-  Effect size type = cohen's d
-  Correction for multiple comparisons = False Discovery Rate
   (Benjamini–Hochberg procedure)

Within-factor
~~~~~~~~~~~~~

.. code:: ipython3

    from pingouin import pairwise_ttests

    stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                            effects='within', data=df, alpha=.05,
                            tail='two-sided', padjust='fdr_bh',
                            effsize='cohen', return_desc=False)
    print(stats)


Between-factor
~~~~~~~~~~~~~~

.. code:: ipython3

    stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                            effects='between', data=df, alpha=.05,
                            tail='two-sided', padjust='fdr_bh',
                            effsize='cohen', return_desc=False)
    print(stats)


Interaction
~~~~~~~~~~~

.. code:: ipython3

    stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                            effects='interaction', data=df, alpha=.05,
                            tail='two-sided', padjust='fdr_bh',
                            effsize='cohen', return_desc=False)
    print(stats)


All of the above
~~~~~~~~~~~~~~~~

We also set **return\_desc=True** in order to get the mean and
standard deviations in each comparisons

.. code:: ipython3

    stats = pairwise_ttests(dv='DV', within='Time', between='Group',
                            effects='all', data=df, alpha=.05,
                            tail='two-sided', padjust='fdr_bh',
                            effsize='cohen', return_desc=True)
    print(stats)
