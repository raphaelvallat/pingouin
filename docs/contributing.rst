.. _Contribute:

Contribute to Pingouin
######################

There are many ways to contribute to Pingouin: reporting bugs or results that are inconsistent with other statistical softwares, adding new functions, improving the documentation, etc...

If you like Pingouin, you can also consider `buying the developers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

Code guidelines
---------------

*Before starting new code*, we highly recommend opening an issue on `GitHub <https://github.com/raphaelvallat/pingouin>`_ to discuss potential changes.

* Please follow `PEP 8 <https://peps.python.org/pep-0008/>`_ Python style guidelines. Pingouin uses `Ruff <https://github.com/astral-sh/ruff>`_ for linting and formatting. Before submitting a PR, please run the following commands from the root folder of Pingouin to sort imports and format code:

  .. code-block:: bash

    $ ruff check --select I --fix

    $ ruff format

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* New functionality must be **validated** against at least one other statistical software including R, SPSS, Matlab or JASP.

* When adding new functions, make sure that they are **generalizable to various situations**, including missing data, unbalanced groups, etc.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

  .. code-block:: bash

     $ pytest --verbose

Setting up a development environment
-------------------------------------

Pingouin uses `uv <https://docs.astral.sh/uv/>`_ for fast dependency management. To set up a local development environment, first clone the repository and then install the package in editable mode with the test dependencies:

.. code-block:: bash

  $ git clone https://github.com/raphaelvallat/pingouin.git
  $ cd pingouin
  $ uv pip install --group=test --editable .

To also install the development tools (Ruff), add the ``dev`` group:

.. code-block:: bash

  $ uv pip install --group=dev --group=test --editable .

Continuous Integration
-----------------------

Pingouin uses `GitHub Actions <https://docs.github.com/en/actions>`_ for continuous integration. The following workflows run automatically on every push and pull request to the ``main`` branch:

* **PyTest** — runs the test suite on Ubuntu, macOS and Windows across Python 3.10, 3.12 and 3.14, as well as against a range of historical dependency versions (from minimum supported to latest).
* **Coverage** — measures test coverage and uploads the report to `Codecov <https://codecov.io/gh/raphaelvallat/pingouin>`_.
* **Ruff** — checks code style and formatting.
* **Documentation** — builds the Sphinx documentation and uploads the result as a downloadable artifact.

A separate **PyTest (pre-release)** workflow runs weekly against pre-release versions of all major dependencies to catch compatibility issues early.

Checking and building documentation
------------------------------------

Pingouin's documentation (including docstrings in code) uses ReStructuredText format,
see `Sphinx documentation <http://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
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

Inspect on GitHub
^^^^^^^^^^^^^^^^^

The documentation is also built automatically on GitHub after every commit you make as part of a Pull Request.
To inspect the rendered documentation, follow these steps:

* Click on the "Show all checks" dropdown menu at the end of the Pull Request user interface
* Click on the check named **Documentation / docs**
* In the top-right corner of the opening window, click the **Artifacts** dropdown menu
* Download the ``docs-artifact`` zip file

You can then unpack that zip file on your computer, enter the directory, and open the ``index.html`` file that you will find there.
That should open the Pingouin documentation based on the changes from your Pull Request.
