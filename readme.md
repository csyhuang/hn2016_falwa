## Python Library: hn2016_falwa (v0.3.0)

[![Build Status](https://travis-ci.org/csyhuang/hn2016_falwa.svg?branch=master)](https://travis-ci.org/csyhuang/hn2016_falwa)[![codecov.io](https://codecov.io/gh/csyhuang/hn2016_falwa/branch/master/graph/badge.svg)](https://codecov.io/gh/csyhuang/hn2016_falwa)[![Documentation Status](https://readthedocs.org/projects/hn2016-falwa/badge/?version=latest)](http://hn2016-falwa.readthedocs.io/en/latest/?badge=latest)

![hn2016_falwa_diagram](https://github.com/csyhuang/csyhuang.github.io/blob/master/assets/img/hn2016_falwa_diagram.png)

Compute from gridded climate data the Finite-amplitude Local Wave Activity (FALWA) and flux terms introduced in:

- [Huang and Nakamura (2016, JAS)](http://dx.doi.org/10.1175/JAS-D-15-0194.1)
- [Huang and Nakamura (2017, GRL)](http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full).
- Nakamura and Huang (2018, Science) *Atmospheric Blocking as a Traffic Jam in the Jet Stream*.

## Important update (May 24, 2018)

The most updated version v0.3.0 enhanced the methods in the class *QGField* (in `hn2016_falwa.oopinterface`) with functionality
to compute (the barotropic components of) LWA and flux terms present in Nakamura and Huang (2018, Science). 
Refer to the scripts in `examples/nh2018_science/` for details.

Please check the updated [documentation page](http://hn2016-falwa.readthedocs.io) as well.

## Installation

This current version works in both Python 2.7 and 3.6. Note that since v0.3.0, some functions are having 
backend in Fortran. You will need a fortran compiler (e.g. [gfortran](http://hpc.sourceforge.net/)) to 
implement the installation.

To install the package:
```
git clone https://github.com/csyhuang/hn2016_falwa.git
cd hn2016_falwa
python setup.py install
```
Bugs have been fixed since the release. To clone the version with bug fixed (on *test* branch):
```
git clone -b test https://github.com/csyhuang/hn2016_falwa.git
```
Changes will be merged in the next updated version.

There are two interfaces for this library. One is the **developer interface**; the other is the **object-oriented 
interface**, which is a wrapper for the basis functions in the developer interface.

### Developer Interface

The **developer interface**  contains separate functions that users can alter the inputs more flexibly. Functions 
are added upon users' request on new functionalities to test hypotheses (also see the *test* branch). The 
**developer interface** consists of 4 types of functions:  

- The **basis functions** are smallest unit of functions that make up the **wrapper functions** and **object-oriented interface**.  

- The **wrapper functions** implement particular analysis tasks for published work/manuscripts in preparation  

- The **utility functions** compute general quantities, such as static stability or quasi-geostrophic potential vorticity that are not specific to the finite-amplitude wave theory.   

- The **beta-version functions** include utilities that are not fully documented but has been used in research.  

Sample Script | Description
------------- | -------------
Example_barotropic.ipynb | It reads in a sample datasets "barotropic_vorticity.nc", which contains absolute vorticity field snapsnots from a barotropic decay model (Held and Phillips 1987). It computes both the equivalent-latitude relationship (e.g. Nakamura 1996) and local wave activity (Huang and Nakamura 2016) in a global domain.
Example_qgpv.ipynb | It reads in a sample datasets u_QGPV_240hPa_2012Oct28to31.nc", which contains zonal velocity and QGPV field at 240hPa derived form ERA-Interim reanalysis data. Similar to fig. 9 in Huang and Nakamura (2016), a hemispheric domain is used here.


### Object-oriented interface

The **object-oriented interface** is an easy-to-use interface that takes in the climate field and coordinates as the attributes of an object, and implement the wrapper functions above as methods.

There are two classes in the interface, *BarotropicField* and *QGField* - the latter is under development for more methods. Please refer to the example/ directory:

Sample Script | Description
------------ | -------------
oopinterface_example_BarotropicField.ipynb | Same as *Example_barotropic.ipynb*.
oopinterface_example_QGField.ipynb | Same as *Example_qgpv.ipynb* 


## Inquiries / Issues reporting

Please make inquiries about / report issues with the package on the [Issues page](https://github.com/csyhuang/hn2016_falwa/issues). If you need help analyzing output from particular model/analysis with our techniques, feel free to email me <csyhuang@uchicago.edu> with sample datasets and I can configure the code for you.

