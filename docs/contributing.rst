.. _Contribute:

Contribute to Pingouin
**********************

There are many ways to contribute to Pingouin: reporting bugs or results that are inconsistent with other statistical softwares, adding new functions, improving the documentation, etc...

If you like Pingouin, you can also consider `buying the developpers a coffee <https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=K2FZVJGCKYPAG&currency_code=USD&source=url>`_!

Code guidelines
---------------

*Before starting new code*, we highly recommend opening an issue on `GitHub <https://github.com/raphaelvallat/pingouin>`_ to discuss potential changes.

* Please use standard `pep8 <https://pypi.python.org/pypi/pep8>`_ and `flake8 <http://flake8.pycqa.org/>`_ Python style guidelines. To test that your code complies with those, you can run:

  .. code-block:: bash

     $ flake8

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


All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings. If you want to test the documentation locally, you will need to install the following packages:

.. code-block:: bash

  $ pip install --upgrade sphinx sphinx_bootstrap_theme numpydoc

and then within the ``pingouin/docs`` directory do:

.. code-block:: bash

  $ make html
