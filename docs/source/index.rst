.. hn2016_falwa documentation master file, created by
   sphinx-quickstart on Mon Aug 21 13:58:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
hn2016_falwa: Finite-amplitude local wave activity
==================================================

`hn2016_falwa <https://github.com/csyhuang/hn2016_falwa>`_ is a python library that provides tools to measure and study life cycle of large-scale
extreme weather events. It implements the finite-amplitude local wave activity diagnostic and related 
proposed in `Huang and Nakamura (2016) <http://dx.doi.org/10.1175/JAS-D-15-0194.1/>`_ on gridded climate 
data.

.. image:: https://github.com/csyhuang/csyhuang.github.io/raw/master/assets/img/hn2016_falwa_diagram.png

Installation
------------------------------

This package works in both Python 2 and 3. Dependencies include Numpy, Scipy and optionally 
Matplotlib. It can be installed from the source distribution::

 git clone https://github.com/csyhuang/hn2016_falwa.git
 cd hn2016_falwa
 python setup.py install


Developer v.s. Object-oriented Interfaces
-----------------------------------------

There are two interfaces for this library. One is the **developer interface**; the other is the **object-oriented 
interface**, which is a wrapper for the basis functions in the developer interface.

The **developer interface**  contains separate functions that users can alter the inputs more flexibly. Functions 
are added upon users' request on new functionalities to test hypotheses (also see the *test* branch). The 
**developer interface** consists of 4 types of functions:  

* The **basis functions** are smallest unit of functions that make up the **wrapper functions** and **object-oriented interface**.  

* The **wrapper functions** implement particular analysis tasks for published work/manuscripts in preparation  

* The **utility functions** compute general quantities, such as static stability or quasi-geostrophic potential vorticity that are not specific to the finite-amplitude wave theory.   

* The **beta-version functions** include utilities that are not fully documented but has been used in research.  

The **object-oriented interface** is an easy-to-use interface that takes in the climate field and coordinates as the attributes of an object, and implement the wrapper functions above as methods.


.. toctree::
   :maxdepth: 2

   Object Oriented Interface
   Basis Functions
   Wrapper Functions   
   Utility Functions
   Beta-version Functions



Issues Reporting
------------------------------

Please make inquiries about / report issues / with the package and suggest feature extensions on the `Issues page <https://github.com/csyhuang/hn2016_falwa/issues>`_. 

If you need help analyzing output from particular model/analysis with our techniques, feel free to email me *csyhuang@uchicago.edu* with sample datasets and I can configure the code for you.
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
