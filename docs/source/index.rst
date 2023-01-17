.. hn2016_falwa documentation master file, created by
   sphinx-quickstart on Mon Aug 21 13:58:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
hn2016_falwa: Finite-amplitude local wave activity
==================================================

.. image:: https://github.com/csyhuang/csyhuang.github.io/raw/master/assets/img/hn2016_falwa_diagram.png

`hn2016_falwa <https://github.com/csyhuang/hn2016_falwa>`_ is a python library that provides tools to measure and study life cycle of large-scale
extreme weather events. It implements the finite-amplitude local wave activity and flux diagnostic introduced in:

- `Huang and Nakamura (2016, JAS) <https://journals.ametsoc.org/doi/abs/10.1175/JAS-D-15-0194.1>`_
- `Huang and Nakamura (2017, GRL) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017GL073760>`_
- `Nakamura and Huang (2018, Science) <https://doi.org/10.1126/science.aat0721>`_
- `Neal et al (2022, GRL) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL097699>`_

Package Installation
------------------------------

This current version works in both Python 2.7 and 3.x. Note that from v0.3.0 onword, some functions are
having  backend in Fortran. You will need a fortran compiler (e.g. `gfortran <http://hpc.sourceforge.net/>`_) 
to implement the installation.

Dependencies include Numpy, Scipy and optionally Matplotlib. Since the package is still being actively developed, please use the *develop* mode for installation::

 git clone https://github.com/csyhuang/hn2016_falwa.git
 cd hn2016_falwa
 python setup.py develop

To incorporate updates, pull the new version of the code from GitHub by::

 git pull

If there are updates in the Fortran modules in the new commits, please re-compile them::

 python setup develop -u
 python setup develop
 pytest # to check if the package can be run properly


Quick start
-----------------------------------------

The jupyter notebook in `examples/nh2018_science` demonstrates how to compute wave activity and reference states presented in Nakamura and Huang (2018).
To make sure the package is properly installed in your environment, run through the notebook after installation to see if there is error.

THe conda environment for running the notebook can be found in `environment.yml`. To create the conda environment, execute::

 conda env create -f environment.yml

Developer v.s. Object-oriented Interfaces
-----------------------------------------

There are two interfaces for this library. One is the **developer interface**; the other is the **object-oriented 
interface**, which is a wrapper for the basis functions in the developer interface and also compiled fortran modules.
Users are strongly adviced to use only the object-oriented interface.

The **object-oriented interface** is an easy-to-use interface that takes in the climate field and coordinates as the attributes of an object, and implement the wrapper functions above as methods.

There are two classes with object-oriented interface: *QGField* and *BarotropicField*. Please refer to the **sample scripts** for their usage:

* nh2018_science/demo_script_for_nh2018.ipynb: Compute wave activity and flux terms in the QG framework presented in Nakamura and Huang (2018, Science). Sample data can be retrieved with `download_example.py` in the same directory.

* simple/oopinterface_example_BarotropicField.ipynb: It reads in a sample datasets "barotropic_vorticity.nc", which contains absolute vorticity field snapsnots from a barotropic decay model (Held and Phillips 1987). It computes both the **equivalent-latitude** relationship (e.g. Nakamura 1996) and **local wave activity** (Huang and Nakamura 2016) in a global domain.

The **developer interface**  contains separate functions that users can alter the inputs more flexibly. Functions 
are added upon users' request on new functionalities to test hypotheses (also see the *test* branch). The 
**developer interface** consists of 4 types of functions:  

* The **basis functions** are smallest unit of functions that make up the **wrapper functions** and **object-oriented interface**.  

* The **wrapper functions** implement particular analysis tasks for published work/manuscripts in preparation  

* The **utility functions** compute general quantities, such as static stability or quasi-geostrophic potential vorticity that are not specific to the finite-amplitude wave theory.   

The **object-oriented interface** is an easy-to-use interface that takes in the climate field and coordinates as the attributes of an object, and implement the wrapper functions above as methods.



.. toctree::
   :maxdepth: 2

   Object Oriented Interface
   Xarray Interface
   Barotropic Field
   Basis Functions
   Utility Functions
   Wrapper Functions
   Beta-version Functions



Issues Reporting
------------------------------

Please make inquiries about / report issues / with the package and suggest feature extensions on the `Issues page <https://github.com/csyhuang/hn2016_falwa/issues>`_. 

If you need help analyzing output from particular model/analysis with our techniques, feel free to email me *csyhuang@protonmail.com* with sample datasets and I can configure the code for you.
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
