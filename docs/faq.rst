.. _faq:

FAQ
===

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

`Statsmodels <https://www.statsmodels.org/stable/index.html>`_ is a great statistical Python package that provides several advanced functions (regression, GLM, time-series analysis) as well as an R-like syntax for fitting models. However, statsmodels can be quite hard to grasp and use for Python beginners and/or users who just want to perform simple statistical tests. The goal of Pingouin is not to replace statsmodels but rather to provide some easy-to-use functions to perform the most widely-used statistical tests. In addition, Pingouin also provides some novel functions (to cite but a few: effect sizes, pairwise T-tests and correlations, ICC, repeated measures correlation, circular statistics...).

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

.. ----------------------------- READING -----------------------------
.. raw:: html

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_read">How can I load a .csv or .xlsx file in Python?</a>
        </h5>
      </div>
      <div id="collapse_read" class="panel-collapse collapse">
        <div class="panel-body">

You need to use the :py:func:`pandas.read_csv` or :py:func:`pandas.read_excel` functions:

.. code-block:: python

    import pandas as pd
    pd.read_csv('myfile.csv')     # Load a .csv file
    pd.read_excel('myfile.xlsx')  # Load an Excel file

.. ----------------------------- MISSING VALUES -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_missing">How does Pingouin deal with missing values?</a>
        </h5>
      </div>
      <div id="collapse_missing" class="panel-collapse collapse">
        <div class="panel-body">

Pingouin hates missing values almost as much as you do!

Most functions of Pingouin will automatically remove the missing values. In the case of paired measurements (e.g. paired T-test, correlation, or repeated measures ANOVA), a listwise deletion of missing values is performed, meaning that the entire row is removed. This is generally the best strategy if you have a large sample size and only a few missing values. However, this can be quite drastic if there are a lot of missing values in your data. In that case, it might be useful to look at `imputation methods (see Pandas documentation) <https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html>`_.

If you prefer to know what's going on under the hood, you can also remove the missing values a priori using the :py:func:`pingouin.remove_na` and :py:func:`pingouin.remove_rm_na` functions. The first one is a convenient and flexible function to remove rows or columns with missing values in 1D or 2D array(s), and the second one is specifically geared at long-format repeated measures dataframe, such as the ones required by the :py:func:`pingouin.rm_anova` function.

.. ----------------------------- LONG <--> WIDE FORMAT -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_wide">What's the difference between wide format and long format data and how can I convert my data from one to the other?</a>
        </h5>
      </div>
      <div id="collapse_wide" class="panel-collapse collapse">
        <div class="panel-body">

In wide format, each row represent a subject, and each column a measurement (e.g. "Pre", "Post"). This is the most convenient way for humans to look at repeated measurements. It typically results in spreadsheet with a larger number of columns than rows. An example of wide-format dataframe is shown below:

+---------+-----+------+--------+-----+
| Subject | Pre | Post | Gender | Age |
+=========+=====+======+========+=====+
| 1       | 2.5 | 3.1  | M      | 24  |
+---------+-----+------+--------+-----+
| 2       | 4.2 | 4.8  | F      | 32  |
+---------+-----+------+--------+-----+
| 3       | 2.5 | 2.9  | F      | 38  |
+---------+-----+------+--------+-----+

In long-format, each row is one time point per subject and each column is a variable (e.g. one column with the "Subject" identifier, another with the "Scores" and another with the "Time" grouping factors). In long-format, there are usually many more rows than columns. While this is harder to read for humans, this is much easier to read for computers. For this reason, all the repeated measures functions in Pingouin work only with long-format dataframe. In the example below, the wide-format dataframe from above was converted into a long-format dataframe:

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


The `Pandas <https://pandas.pydata.org/>`_ package provides some convenient functions to convert from one format to the other:

* From wide-format to long-format (easier to read for computer), use the :py:func:`pandas.melt` function.
* From long-format to wide-format, use the :py:func:`pandas.pivot_table` function.

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

No, the central idea behind Pingouin is that all data manipulations and descriptive statistics should be first performed in Pandas (or NumPy). For example, to compute the mean, standard deviation, and quartiles of all the numeric columns of a pandas DataFrame, one can easily use the :py:meth:`pandas.DataFrame.describe` method:

.. code-block:: python

    data.describe()

.. raw:: html

          </div>
        </div>

.. ############################################################################
.. ############################################################################
..                                  OTHERS
.. ############################################################################
.. ############################################################################

Others
------

.. ----------------------------- LICENSE -----------------------------
.. raw:: html

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_license">Why is Pingouin licensed under the GNU-GPL v3?</a>
        </h5>
      </div>
      <div id="collapse_license" class="panel-collapse collapse">
        <div class="panel-body">

Pingouin is licensed under the GNU General Public License v3.0 (GPL-3), which is less permissive than the BSD or MIT licenses. The reason for this is that Pingouin borrows extensively from R packages, which are all licensed under the GPL-3.
To read more about what you can do and cannot do with a GPL-3 license, please visit `tldrlegal.com <https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#summary>`_ or `choosealicense.com <https://choosealicense.com/licenses/>`_.

.. ----------------------------- NEW RELEASES -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_release">How can I be notified of new releases?</a>
        </h5>
      </div>
      <div id="collapse_release" class="panel-collapse collapse">
        <div class="panel-body">

To be notified whenever a new release of Pingouin is available, you can click on "Watch releases" on the `GitHub <https://github.com/raphaelvallat/pingouin>`_ of Pingouin (see below).

.. figure::  /pictures/github_watch_release.png
  :align:   center

Whenever a new release is out there, you can simply upgrade your version by typing the following line in a terminal window:

.. code-block:: shell

    pip install --upgrade pingouin

.. ----------------------------- DONATION -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_donate">I am not a programmer, how can I contribute to Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_donate" class="panel-collapse collapse">
        <div class="panel-body">

There are many ways to contribute to Pingouin, even if you are not a programmer, for example, reporting bugs or results that are inconsistent with other statistical softwares, improving the documentation and examples, or, even `buying the developpers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

.. ----------------------------- CITING PINGOUIN -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h5 class="panel-title">
          <a data-toggle="collapse" href="#collapse_cite">How can I cite Pingouin?</a>
        </h5>
      </div>
      <div id="collapse_cite" class="panel-collapse collapse">
        <div class="panel-body">

To cite Pingouin, please use the publication in JOSS:

Vallat, R. (2018). Pingouin: statistics in Python. *Journal of Open Source Software*, 3(31), 1026, `https://doi.org/10.21105/joss.01026 <https://doi.org/10.21105/joss.01026>`_

BibTeX:

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

.. ----------------------------- END -----------------------------
.. raw:: html

          </div>
        </div>
      </div>
