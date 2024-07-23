.. _Contribute:

Contribute to Pingouin
######################

There are many ways to contribute to Pingouin: reporting bugs or results that are inconsistent with other statistical softwares, adding new functions, improving the documentation, etc...

If you like Pingouin, you can also consider `buying the developers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

Code guidelines
---------------

*Before starting new code*, we highly recommend opening an issue on `GitHub <https://github.com/raphaelvallat/pingouin>`_ to discuss potential changes.

* Please use standard `pep8 <https://pypi.python.org/pypi/pep8>`_ and `flake8 <http://flake8.pycqa.org/>`_ Python style guidelines. Pingouin uses `black <https://github.com/psf/black>`_ for code formatting. Before submitting a PR, please make sure to run the following command in the root folder of Pingouin:

  .. code-block:: bash

     $ black . --line-length=100

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* New functionality must be **validated** against at least one other statistical software including R, SPSS, Matlab or JASP.

* When adding new functions, make sure that they are **generalizable to various situations**, including missing data, unbalanced groups, etc.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

  .. code-block:: bash

     $ pytest --doctest-modules

Checking and building documentation
-----------------------------------

Pingouin's documentation (including docstring in code) uses ReStructuredText format,
see `Sphinx documentation <http://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.


All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings.

Build locally
^^^^^^^^^^^^^

If you want to test the documentation locally, you will need to install the following packages:

.. code-block:: bash

  $ pip install --upgrade sphinx sphinx_bootstrap_theme numpydoc sphinx-copybutton

and then within the ``pingouin/docs`` directory do:

.. code-block:: bash

  $ make html

or call make from the root ``pingouin`` directory directly,
using the ``-C`` flag to tell the ``make`` command to first switch to the ``docs`` directory,
and then come back after executing the ``html`` recipe.

.. code-block:: bash

  $ make -C docs html

Inspect on GitHub
^^^^^^^^^^^^^^^^^

Thanks to the `GitHub Actions <https://docs.github.com/en/free-pro-team@latest/actions>`_ continuous integration service,
the documentation is also built on GitHub servers after every commit you make as part of a Pull Request.
To inspect these build artifacts, follow these steps:

* Click on the "Show all checks" dropdown menu at the end of the Pull Request user interface

.. figure::  /pictures/github_checks.png
  :align:   center
  :alt: GitHub checks dropdown menu

  Screenshot of the GitHub checks dropdown menu

* Click on the check that starts with ``Python tests / build (ubuntu-latest, 3.8)``
* Now in the top right corner of the opening window, you will see a small dropdown menu called "Artifacts"

.. figure::  /pictures/github_build_artifacts.png
  :align:   center
  :alt: GitHub build artifacts dropdown menu

  Screenshot of the GitHub build artifacts dropdown menu

* Click on that drowndown menu and download the ``docs-artifact`` zip file

You can then unpack that zip file on your computer, enter the directory, and open the ``index.html`` file that you will find there.
That should open the Pingouin documentation based on the changes from your Pull Request.
