.. falwa documentation master file, created by
   sphinx-quickstart on Mon Aug 21 13:58:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
falwa: Finite-amplitude local wave activity
==================================================

.. image:: https://github.com/csyhuang/csyhuang.github.io/raw/master/assets/img/falwa_diagram.png

`falwa <https://github.com/csyhuang/hn2016_falwa>`_ is a python library that provides tools to measure and study life cycle of large-scale
extreme weather events. It implements the finite-amplitude local wave activity and flux diagnostic presented in:

- `Huang and Nakamura (2016, JAS) <https://journals.ametsoc.org/doi/abs/10.1175/JAS-D-15-0194.1>`_
- `Huang and Nakamura (2017, GRL) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017GL073760>`_
- `Nakamura and Huang (2018, Science) <https://doi.org/10.1126/science.aat0721>`_
- `Neal et al (2022, GRL) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL097699>`_
- Lubis et al (accepted, Nature Communication). *Cloud-Radiative Effects Significantly Increase Wintertime Atmospheric Blocking in the Euro-Atlantic Sector*.

Citing this package
---------------------

We would be grateful if you mention `falwa` and cite our `software package paper <https://doi.org/10.22541/essoar.173179959.91611195/v2>`_ published in the Geoscience Data Journal upon usage:

.. code-block:: language

   @article{falwa_csyhuang_cpolster,
     author={Clare S. Y. Huang and Christopher Polster and Noboru Nakamura},
     title={falwa: Python package to implement Finite-amplitude Local Wave Activity diagnostics on climate data},
     journal={Geoscience Data Journal},
     note = {(in press)},
     year = {2025},
     doi = {10.1002/GDJ3.70006},
     publisher={Wiley Online Library}}

Package Installation
------------------------------

**Attention: substantial changes took place in release v2.0.0. Installation in develop mode is no longer available.**

Since release v2.0.0, the F2PY modules in `falwa` is compiled with `meson` (See Issue #95 for details) to cope with the deprecation of `numpy.disutils` in python 3.12.

First-time installation
^^^^^^^^^^^^^^^^^^^^^^^

1. To build the package from source, you need a fortran compiler (e.g., [gfortran](http://hpc.sourceforge.net/)) to implement the installation.
2. Clone the package repo by `git clone https://github.com/csyhuang/hn2016_falwa.git` .
3. Navigate into the repository and set up a python environment satisfying the installation requirement by `conda env create -f environment.yml`. The environment name in the file is set to be `falwa_env` (which users can change).
4. Install the package with the command `python -m pip install .`. The compile modules will be saved to python site-packages directory.
5. If the installation is successful, you should be able to run through all unit tests in the folder `tests/` by executing `pytest tests/`.

Get updated code from new releases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. To incorporate updates, first, pull the new version of the code from GitHub by `git pull`.
2. Uninstall existing version of `falwa`: `pip uninstall falwa`
3. If there is change in `environment.yml`, remove the existing environment by `conda remove --name falwa_env --all` and create the environment again from the updated YML file: `conda env create -f environment.yml`.
4. Reinstall the updated version by `python -m pip install .`.
5. Run through all unit tests in the folder `tests/` by executing `pytest tests/` to make sure the package has been properly installed.

Quick start
-----------------------------------------

There are some readily run python scripts (in `scripts/`) and jupyter notebooks (in `notebooks/`) which you can start with.
The netCDF files needed can be found in `Clare's Dropbox folder <https://www.dropbox.com/scl/fo/b84pwlr7zzsndq8mpthd8/AKMmwRiYhK4mRmdOgmFg5SM?rlkey=k15p1acgksnl2xwcxve3alm0u&st=2mg4svks&dl=0>`_.

Depending on what you want to do, the methods to be use may be different.

1. If you solely want to compute equivalent latitude and local wave activity from a 2D field, you can refer to `notebooks/simple/Example_barotropic.ipynb`. This is useful for users who want to use LWA to quantify field anomalies.

2. If you want to compute zonal wind reference states and wave activity fluxes in QG Formalism, look at `notebooks/nh2018_science/demo_script_for_nh2018.ipynb` for the usage of `QGField`. This notebook demonstrates how to compute wave activity and reference states presented in Nakamura and Huang (2018). To make sure the package is properly installed in your environment, run through the notebook after installation to see if there is error.

Issues Reporting
------------------------------

- If you are interested in getting email message related to update of this package, please leave your contact `here <https://goo.gl/forms/5L8fv0mUordugq6v2>`_ such that I can keep you updated of any changes made.
- If you encounter *coding issues/bugs* when using the package, please create an `Issue ticket <https://github.com/csyhuang/hn2016_falwa/issues>`_.
- If you have scientific questions, please create a thread in the `Discussion Board <https://github.com/csyhuang/hn2016_falwa/discussions>`_ with the category "General" or "Q&A" according to the circumstance.

Modules
========

.. toctree::
   :maxdepth: 2

   Example Notebooks
   Object Oriented Interface
   Data Storage
   Xarray Interface
   Barotropic Field
   Utility Functions
   Plot Utilities
   netCDF Utilities
   Statistics Utilities
   Basis Functions
   Wrapper Functions

Notes
========

.. toctree::
   :maxdepth: 2

   Preprocessing
   Memo for Developers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
