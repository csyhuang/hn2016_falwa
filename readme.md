## Python Library: hn2016_falwa (v0.2.0)

[![Build Status](https://travis-ci.org/csyhuang/hn2016_falwa.svg?branch=master)](https://travis-ci.org/csyhuang/hn2016_falwa)[![codecov.io](https://codecov.io/gh/csyhuang/hn2016_falwa/branch/master/graph/badge.svg)](https://codecov.io/gh/csyhuang/hn2016_falwa)

Compute Finite-amplitude Local Wave Activity (FALWA) introduced in [Huang and Nakamura (2016)](http://dx.doi.org/10.1175/JAS-D-15-0194.1) and [Huang and Nakamura (2017)](http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full) from gridded climate data.

![hn2016_falwa_diagram](https://github.com/csyhuang/csyhuang.github.io/blob/master/assets/img/hn2016_falwa_diagram.png)

The functions in the library can also be used to compute the tracer equivalent-latitude relationship proposed in Nakamura (1996) / Allen and Nakamura (2003) and the (zonal mean) finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

Please check the [documentation page](https://cdn.rawgit.com/csyhuang/hn2016_falwa/b7efacaa/docs/build/html/index.html) for more details.

## Installation

This current version works in both Python 2.7 and 3.6. To install from the source:
```
git clone https://github.com/csyhuang/hn2016_falwa.git
cd hn2016_falwa
python setup.py install
```

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

