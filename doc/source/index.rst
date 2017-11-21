.. hn2016_falwa documentation master file, created by
   sphinx-quickstart on Mon Aug 21 13:58:59 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
hn2016_falwa: Finite-amplitude local wave activity
==================================================

*hn2016_falwa* is a python library that provides tools to measure and study life cycle of large-scale
extreme weather events. It implements the finite-amplitude local wave activity diagnostic and related 
proposed in `Huang and Nakamura (2016) <http://dx.doi.org/10.1175/JAS-D-15-0194.1/>`_ on gridded climate 
data.

.. image:: https://github.com/csyhuang/csyhuang.github.io/raw/master/assets/img/hn2016_falwa_diagram.png

Installation
------------------------------

This package works in Python 2.7 (Python 3 version will be included soon). Dependencies include Numpy, 
Scipy and optionally Matplotlib. It can be installed with pip via::

 pip install hn2016_falwa

You can also install from the source distribution::

 git clone https://github.com/csyhuang/hn2016_falwa.git
 cd hn2016_falwa
 python setup.py install


Issues Reporting
------------------------------

Please make inquiries about / report issues / with the package and suggest feature extensions on the `Issues page <https://github.com/csyhuang/hn2016_falwa/issues>`_. 

If you need help analyzing output from particular model/analysis with our techniques, feel free to email me *clare1068@gmail.com* with sample datasets and I can configure the code for you.
  
Developer v.s. Object-oriented Interfaces
==========================================

There are two interfaces for this library. One is the developer interface, where modules are separated and independent of each other.

.. toctree::
   :maxdepth: 2
   :caption: Documentation for hn2016_falwa

   Object Oriented Interface
   Basis Functions
   Wrapper Functions   
   Utility Functions
   Beta-version Functions
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
