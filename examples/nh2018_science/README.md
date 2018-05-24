# Demo script for the analyses done in Nakamura and Huang (2018, Science)

The jupyter notebook `demo_script_for_nh2018.ipynb` is a complimentary demo script that can 
be used to reproduce data required in the analyses done in

Nakamura and Huang, Atmospheric Blocking as a Traffic Jam in the Jet Stream. Science. (2018)

This notebook demonstrate how to compute local wave activity and all the flux terms in equations 
(2) and (3) in NH2018 with the updated functionality in the python package `hn2016_falwa`. 
To run the script, please install the 
package `hn2016_falwa` using
```
python setup.py install
```
after cloning the [GitHub repo](http://github.com/csyhuang/hn2016_falwa).

The sample data can be downloaded with the script `download_example.py`, given that you have 
the python package `ecmwfapi` installed and have an account on ECMWF server.

The functionalities are enhanced and included in the class object `QGField` under 
`hn2016_falwa.oopinterface`. Please refer to the [documentation](http://hn2016-falwa.readthedocs.io/) (search `QGField`) 
for the input/methods this class provides.

Feel free to contact Clare S. Y. Huang (csyhuang@uchicago.edu) if you have any questions or suggestions regarding the package.