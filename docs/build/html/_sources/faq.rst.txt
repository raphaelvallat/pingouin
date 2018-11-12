.. faq:

FAQ
***

Python
------

.. ----------------------------- INTRO -----------------------------
.. raw:: html

    <div class="panel-group">
      <div class="panel panel-default">
        <div class="panel-heading">
          <h5 class="panel-title">
            <a data-toggle="collapse" href="#collapse_python">I am new to Python, how can I install Python and Pingouin on my computer?</a>
          </h5>
        </div>
        <div id="collapse_python" class="panel-collapse collapse">
          <div class="panel-body">

To install Python  on your computer, you should use `Anaconda <https://conda.io/docs/index.html>`_, a Python distribution which natively includes all the most important packages. Then, open the newly installed Anaconda prompt and type:

.. code-block:: bash

    conda install pip

This will install pip, the most-widely used package manager in Python. Once pip is installed, you should be able to install Pingouin. Still in Anaconda prompt, run the following command:

.. code-block:: bash

    pip install pingouin

You are almost ready to use Pingouin. First, you need to open an interactive Python console (either `IPython <https://ipython.org/>`_ or `Jupyter <https://jupyter.readthedocs.io/en/latest/index.html>`_). To do so, type the following command:

.. code-block:: bash

    ipython

Now, let's do a simple paired T-test using Pingouin:

.. code-block:: python

    import pingouin as pg
    # Create two variables
    x = [4, 6, 5, 7, 6]
    y = [2, 2, 3, 1, 2]
    # Run a T-test
    pg.ttest(x, y, paired=True)


.. ----------------------------- IMPORT -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_import">How to import and use Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_import" class="panel-collapse collapse">
        <div class="panel-body">

.. code-block:: python

    # 1) Import the full package
    # --> Best if you are planning to use several Pingouin functions.
    import pingouin as pg
    pg.ttest(x, y)

    # 2) Import specific functions
    # --> Best if you are planning to use only this specific function.
    from pingouin import ttest
    ttest(x, y)

.. ----------------------------- STATSMODELS -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_sm">What are the differences between statsmodels and Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_sm" class="panel-collapse collapse">
        <div class="panel-body">

`Statsmodels <https://www.statsmodels.org/stable/index.html>`_ is a great statistical Python package that provides several advanced functions (regression, GLM, time-series analysis) as well as an R-like syntax for fitting models. However, statsmodels can be quite hard to grasp and use for Python beginners and/or users who just want to perform simple statistical tests. The goal of Pingouin is not to replace statsmodels but rather to provide some easy-to-use functions to perform the most widely-used statistical tests. The two packages are therefore complementary, for instance one can use Pingouin first to do some exploratory data analysis in a few lines of code and then switch to statsmodels if a more exhaustive output is needed.

.. ----------------------------- SCIPY -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_scp">What are the differences between scipy.stats and Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_scp" class="panel-collapse collapse">
        <div class="panel-body">

The `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module provides several low-level statistical functions. However, most of these functions do not return a very detailed output (e.g. only the T- and p-values for a T-test). Most of Pingouin function are using the low-level SciPy funtions to provide a richer, more exhaustive, output. See for yourself!:

.. code-block:: python

    import pingouin as pg
    from scipy.stats import ttest_ind

    x = [4, 6, 5, 7, 6]
    y = [2, 2, 3, 1, 2]

    print(pg.ttest(x, y))   # Pingouin: returns a DataFrame with T-value, p-value, degrees of freedom, tail, Cohen d, power and Bayes Factor
    print(ttest_ind(x, y))  # SciPy: returns only the T- and p-values

.. raw:: html

          </div>
        </div>

.. ############################################################################
.. ############################################################################
..                                  DATA
.. ############################################################################
.. ############################################################################

Data
----

.. ----------------------------- DATA FORMAT -----------------------------
.. raw:: html

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_data">I want to analyze a spreadsheet in Pingouin, how should I organize my data?</a>
        </h5>
      </div>
      <div id="collapse_data" class="panel-collapse collapse">
        <div class="panel-body">

Most Pingouin functions assume that your data is in tidy or long format, that is, each variable should be in one column and each observation should be in a different row. This is true for all the ANOVA / post-hocs function as well as the linear/logistic regression, pairwise correlations, partial correlation, mediation analysis, etc...

An example of data in long-format is shown below. Note that *Scores* is the dependant variable, *Subject* is the subject identifier, *Time* is a within-subject factor (two time points per subject), and *Age* and *Gender* are meta-data:

======= ====== === ==== ======
Subject Gender Age Time Scores
======= ====== === ==== ======
1       M      24  Pre  2.5
1       M      24  Post 3.1
2       F      32  Pre  4.2
2       F      32  Post 4.8
3       F      38  Pre  2.5
3       F      38  Post 2.9
======= ====== === ==== ======

To convert your data from a wide format (typical in Excel) to a long format, you can use the `pandas.melt() <https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.melt.html>`_ function

.. ----------------------------- READING -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_read">How can I load a .csv or .xlsx file in Python?</a>
        </h5>
      </div>
      <div id="collapse_read" class="panel-collapse collapse">
        <div class="panel-body">

You need to use the `Pandas <https://pandas.pydata.org/>`_ package:

.. code-block:: python

    import pandas as pd
    pd.read_csv('myfile.csv')     # Load a .csv file
    pd.read_excel('myfile.xlsx')  # Load an Excel file

.. ----------------------------- DESCRIPTIVE -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_desc">Can I compute descriptive statistics with Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_desc" class="panel-collapse collapse">
        <div class="panel-body">

No, the central idea behind Pingouin is that all data manipulations and descriptive statistics should be first performed in Pandas (or NumPy). For example, to compute the mean, standard deviation, and quartiles of all the numeric columns of a pandas DataFrame, one can easily use the `pandas.DataFrame.describe() <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html>`_ method:

.. code-block:: python

    data.describe()

.. ----------------------------- END -----------------------------
.. raw:: html

          </div>
        </div>
      </div>
    </div>
