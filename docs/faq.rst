.. faq:

FAQ
***

.. ----------------------------- INTRO -----------------------------
.. raw:: html

    <div class="panel-group">
      <div class="panel panel-default">
        <div class="panel-heading">
          <h4 class="panel-title">
            <a data-toggle="collapse" href="#collapse_python">How can I start with Python and Pingouin?</a>
          </h4>
        </div>
        <div id="collapse_python" class="panel-collapse collapse">
          <div class="panel-body">

If you are completely new to Python, this section is for you.
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
        <h4 class="panel-title">
          <a data-toggle="collapse" href="#collapse_import">How to import and use Pingouin?</a>
        </h4>
      </div>
      <div id="collapse_import" class="panel-collapse collapse">
        <div class="panel-body">

.. code-block:: python

    # 1) Import the full package
    import pingouin as pg
    pg.ttest(x, y)

    # 2) Import specific functions
    from pingouin import ttest
    ttest(x, y)


.. ----------------------------- DATA FORMAT -----------------------------
.. raw:: html

          </div>
        </div>
      </div>

    <div class="panel panel-default">
      <div class="panel-heading">
        <h4 class="panel-title">
          <a data-toggle="collapse" href="#collapse_data">I want to analyze a spreadsheet (.csv or .xlsx file), how should I format my data?</a>
        </h4>
      </div>
      <div id="collapse_data" class="panel-collapse collapse">
        <div class="panel-body">

Most Pingouin functions assume that your data is in tidy or long format, that is, each variable should be in one column and each observation should be in a different row. This is true for all the ANOVA / post-hocs function as well as the linear/logistic regression, mediation analysis, etc...

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

.. ----------------------------- END -----------------------------
.. raw:: html

          </div>
        </div>
      </div>
    </div>
