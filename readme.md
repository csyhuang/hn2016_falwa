## Python Library: hn2016_falwa (v0.7.2)

[![Build Status](https://github.com/csyhuang/hn2016_falwa/actions/workflows/workflow.yml/badge.svg)](https://github.com/csyhuang/hn2016_falwa/actions/workflows/workflow.yml)[![codecov.io](https://codecov.io/gh/csyhuang/hn2016_falwa/branch/master/graph/badge.svg)](https://codecov.io/gh/csyhuang/hn2016_falwa)[![Documentation Status](https://readthedocs.org/projects/hn2016-falwa/badge/?version=latest)](http://hn2016-falwa.readthedocs.io/en/latest/?badge=latest)[![DOI](https://zenodo.org/badge/63908662.svg)](https://zenodo.org/badge/latestdoi/63908662)


![hn2016_falwa_diagram](https://github.com/csyhuang/csyhuang.github.io/blob/master/assets/img/hn2016_falwa_diagram.png)

Compute from gridded climate data the Finite-amplitude Local Wave Activity (FALWA) and flux terms introduced in:

- [Huang and Nakamura (2016, JAS)](http://dx.doi.org/10.1175/JAS-D-15-0194.1)
- [Huang and Nakamura (2017, GRL)](http://onlinelibrary.wiley.com/doi/10.1002/2017GL073760/full).
- [Nakamura and Huang (2018, Science)](https://doi.org/10.1126/science.aat0721) *Atmospheric Blocking as a Traffic Jam in the Jet Stream*.
- [Neal et al (2022, GRL)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL097699) *The 2021 Pacific Northwest Heat Wave and Associated Blocking: Meteorology and the Role of an Upstream Cyclone as a Diabatic Source of Wave Activity*

## Package Installation

This current version works for Python 3.x. Note that since v0.3.0, some functions have backend in Fortran. To build the package from source, you need a fortran compiler (e.g. [gfortran](http://hpc.sourceforge.net/)) to implement the installation.

Since the package is still being actively developed, please use the *develop* mode for installation.

To install the package for the first time, clone the GitHub repo and install via `develop` mode:
```
git clone https://github.com/csyhuang/hn2016_falwa.git
cd hn2016_falwa
python setup.py develop
```

To incorporate updates, pull the new version of the code from GitHub. Remove any existing f2py modules and recompile.
```
# Assume you are already in the hn2016_falwa/ repo
git pull
rm hn2016_falwa/*.so
python setup.py develop
pytest tests/ # to check if the package can be run properly
```

## Quick start

There are some readily run python scripts (in `scripts/`) and jupyter notebooks (in `notebooks/`) which you can start with. 
The netCDF files needed can be found in [Clare's Dropbox folder](https://www.dropbox.com/scl/fo/b84pwlr7zzsndq8mpthd8/h?dl=0&rlkey=f8c1gm2xaxvx3c7cf06vop6or).

Depending on what you want to do, the methods to be use may be different.

1. If you solely want to compute equivalent latitude and local wave activity from a 2D field, you can refer to `notebooks/simple/Example_barotropic.ipynb`. This is useful for users who want to use LWA to quantify field anomalies.

2. If you want to compute zonal wind reference states and wave activity fluxes in QG Formalism, look at `notebooks/nh2018_science/demo_script_for_nh2018.ipynb` for the usage of `QGField`. This notebook demonstrates how to compute wave activity and reference states presented in Nakamura and Huang (2018). To make sure the package is properly installed in your environment, run through the notebook after installation to see if there is error.

THe conda environment for running the notebook can be found in `environment.yml`. To create the conda environment, execute:

```bash
conda env create -f environment.yml
```

## Sponsorship acknowledgement

<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png"  width="100">

This project is sponsored by JetBrains as one of the non-commercial open source projects. JetBrains provides core project contributors with a set of best-in-class developer tools free of charge. Please check out their [Open Source Support page](https://www.jetbrains.com/community/opensource/) for details.  

## Inquiries / Issues reporting

- If you are interested in using the package, please leave your contact [here](https://goo.gl/forms/5L8fv0mUordugq6v2) or email me(csyhuang@uchicago.edu) such that I can keep you updated of any changes made.
- If you encounter *coding issues/bugs* when using the package, please create an [Issue ticket](https://github.com/csyhuang/hn2016_falwa/issues).
- If you have scientific questions, please contact Clare S. Y. Huang via email(csyhuang@uchicago.edu).