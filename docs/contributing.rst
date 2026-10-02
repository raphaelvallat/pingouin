.. _Contribute:

Contribute to Pingouin
######################

There are many ways to contribute to Pingouin: reporting bugs or results that are inconsistent with other statistical softwares, adding new functions, improving the documentation, etc...

If you like Pingouin, you can also consider `buying the developers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

Code guidelines
---------------

*Before starting new code*, we highly recommend opening an issue on `GitHub <https://github.com/raphaelvallat/pingouin>`_ to discuss potential changes.

* Please follow `PEP 8 <https://peps.python.org/pep-0008/>`_ Python style guidelines. Pingouin uses `Ruff <https://github.com/astral-sh/ruff>`_ for linting and formatting. The easiest way to ensure your code is properly formatted before committing is to use the `pre-commit <https://pre-commit.com/>`_ hooks (see :ref:`pre-commit-hooks`). Alternatively, you can run the following commands manually from the root folder of Pingouin:

  .. code-block:: bash

    $ ruff check --fix

    $ ruff format

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* New functionality must be **validated** against at least one other statistical software including R, SPSS, Matlab or JASP.

* When adding new functions, make sure that they are **generalizable to various situations**, including missing data, unbalanced groups, etc.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

  .. code-block:: bash

     $ pytest --verbose

Deprecations
------------

When a function or argument is deprecated, it keeps working for at least one release and emits a ``FutureWarning``, which is shown to end users by default (unlike ``DeprecationWarning``). Use ``stacklevel=2`` so that the warning points to the user's code, and say in the message what to use instead and in which version the old behavior will be removed:

.. code-block:: python

  warnings.warn(
      "The `alpha` argument is deprecated and will be removed in version 0.9.0. "
      "Use `confidence` instead.",
      FutureWarning,
      stacklevel=2,
  )

Add a ``.. deprecated::`` directive to the docstring, an entry to the changelog, and a test that checks the warning with ``pytest.warns(FutureWarning)``.

.. _pre-commit-hooks:

Pre-commit hooks
-----------------

Pingouin uses `pre-commit <https://pre-commit.com/>`_ to automatically run Ruff linting and formatting on every commit. pre-commit is part of the ``dev`` dependency group (see :ref:`dev-environment`). To set up the hooks:

.. code-block:: bash

  $ pre-commit install

Once installed, Ruff will run automatically on all staged files before each commit.

.. _dev-environment:

Setting up a development environment
-------------------------------------

Pingouin uses `uv <https://docs.astral.sh/uv/>`_ for fast dependency management. To set up a local development environment, first clone the repository and then install the package in editable mode with all development dependencies (testing, linting, pre-commit):

.. code-block:: bash

  $ git clone https://github.com/raphaelvallat/pingouin.git
  $ cd pingouin
  $ uv pip install --group=dev --editable .

Continuous Integration
-----------------------

Pingouin uses `GitHub Actions <https://docs.github.com/en/actions>`_ for continuous integration. The following workflows run automatically on every pull request and on every push to the ``main`` branch:

* **PyTest** — runs the test suite on Ubuntu, macOS and Windows with Python 3.11 and 3.14, and the docstring examples. A second job tests a range of dependency versions, from the minimum supported to the latest, on Python 3.11 to 3.13. The coverage of the tests is uploaded to `Codecov <https://codecov.io/gh/raphaelvallat/pingouin>`_.
* **Ruff** — checks code style and formatting.
* **Documentation** — builds the Sphinx documentation, failing on any warning, and uploads the result as a downloadable artifact. On ``main``, the documentation is deployed to `pingouin-stats.org <https://pingouin-stats.org>`_ instead.

A separate **PyTest (pre-release)** workflow runs weekly against pre-release versions of all major dependencies to catch compatibility issues early.

Checking and building documentation
------------------------------------

Pingouin's documentation (including docstrings in code) uses ReStructuredText format,
see `Sphinx documentation <https://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings.

Build locally
^^^^^^^^^^^^^

If you want to test the documentation locally, install the package with the ``docs`` dependency group:

.. code-block:: bash

  $ uv pip install --group=docs --editable .

Then, within the ``pingouin/docs`` directory, run:

.. code-block:: bash

  $ make html

or call make from the root ``pingouin`` directory directly,
using the ``-C`` flag to tell the ``make`` command to first switch to the ``docs`` directory,
and then come back after executing the ``html`` recipe.

.. code-block:: bash

  $ make -C docs html

The CI build treats warnings as errors, and runs in nitpicky mode (``-n``): every cross-reference that cannot be resolved is a warning. To refer to a function that no longer exists, e.g. in the changelog, prefix it with ``!`` (:literal:`:py:func:\`!pingouin.old_function\``). To check this locally, use:

.. code-block:: bash

  $ make -C docs html SPHINXOPTS="-W -n --keep-going"

Inspect on GitHub
^^^^^^^^^^^^^^^^^

The documentation is also built automatically on GitHub after every commit you make as part of a Pull Request.
To inspect the rendered documentation, follow these steps:

* Click on the "Show all checks" dropdown menu at the end of the Pull Request user interface
* Click on the check named **Build documentation and upload as artifact to GitHub Actions / docs**
* In the top-right corner of the opening window, click the **Artifacts** dropdown menu
* Download the ``docs-artifact`` zip file

You can then unpack that zip file on your computer, enter the directory, and open the ``index.html`` file that you will find there.
That should open the Pingouin documentation based on the changes from your Pull Request.
