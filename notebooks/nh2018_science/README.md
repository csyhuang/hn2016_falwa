## Demo script for the analyses done in Nakamura and Huang (2018, Science)

### Update on 2024/10/13: ERA-Interim dataset is replaced by the ERA5 version

Given the download API for ERA-Interim has been deprecated, the version of notebooks after commit 0081017 consumes ERA5 data which can be downloaded from [Clare's dropbox folder](https://www.dropbox.com/scl/fo/b84pwlr7zzsndq8mpthd8/h?dl=0&rlkey=f8c1gm2xaxvx3c7cf06vop6or):

`notebooks/nh2018_science/2005-01-23_to_2005-01-30_[uvt].nc`

The legacy ERA-Interim data can be found in the same folder:

`notebooks/nh2018_science/legacy_erai_2005-01-23_to_2005-01-30_[uvt].nc`

### Update in Release 0.6.1: xarray interface

Thanks [Christopher Polster](https://github.com/chpolste) for creating an Xarray interface for the `QGField` class! 
The Xarray example can be found in the jupyter notebook `demo_script_for_nh2018_with_xarray.ipynb` in this directory. 
Documentation can be found on: 
https://hn2016-falwa.readthedocs.io/en/latest/Xarray%20Interface.html .

### Sample script Description

The jupyter notebook `demo_script_for_nh2018.ipynb` is a complimentary demo script that can 
be used to reproduce data required in the analyses done in 

>Nakamura and Huang, Atmospheric Blocking as a Traffic Jam in the Jet Stream. Science. (2018)

This notebook demonstrates how to compute local wave activity (LWA), LWA flux and flux con/divergence 
in equations (2) and (3) in NH2018 with the updated functionality in the python package `hn2016_falwa` 
(version 0.3.1 and after).

To set up the environment to run the scripts, create an environment with `examples/nh2018_science/environment.yml`:
```
conda env create -f environment.yml
```
Optional packages used in jupyter notebook includes `netcdf4`, `matplotlib` and `jupyterlab`:
```
conda install -c conda-forge netcdf4
conda install -c conda-forge matplotlib-base
conda install -c conda-forge jupyterlab
```

To run the script, clone the [GitHub repo](http://github.com/csyhuang/hn2016_falwa) and install 
the package with
```
python setup.py install
```

The sample data can be downloaded with the script `download_example.py`, given that you installed 
the python package `ecmwfapi` and have an account on ECMWF server. [Update for Release 0.6.1: this package seems to 
have deprecated. The download script will use `cdsapi` in the upcoming release.]

The functionalities are enhanced and included in the class object `QGField` under 
`hn2016_falwa.oopinterface`. Please refer to the [documentation](http://hn2016-falwa.readthedocs.io/) (search `QGField`) 
for the input/methods this class provides.

- If you encounter *coding issues/bugs* when using the package, please create an [Issue ticket](https://github.com/csyhuang/hn2016_falwa/issues).
- If you have scientific questions, you can also open a discussion on [Issue page](https://github.com/csyhuang/hn2016_falwa/issues) and label the ticket `'scientific question'`. You can also email Clare S. Y. Huang via email(csyhuang at uchicago.edu).