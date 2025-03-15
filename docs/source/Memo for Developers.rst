
Memo for Developers
===================

Short memo for contributors to this repository.

Files to update version number when doing a release:

- `pyproject.toml`
- `readme.md`
- `docs/source/conf.py`
- `src/falwa/__init__.py`


Quick command to uninstall and reinstall from local source code:

.. code-block:: bash

   pip uninstall falwa -y && python -m pip install .

Additional packages to be installed if compiling documentation locally:

- sphinx
- sphinx_rtd_theme
- nbsphinx

To run coverage test

.. code-block:: bash

   pip install pytest-cov
   coverage run -m pytest
   coverage report -m

