# Demo script for the analyses done in Neal et al. (submitted to GRL)

This directory contains the sample script to reproduce the plots in 
Neal, et al. (2022). [The 2021 Pacific Northwest heat wave and associated blocking: meteorology and the role of an upstream cyclone as a diabatic source of wave activity](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL097699). *Geophysical Research Letters*, 49(8), e2021GL097699.

## Data required

To produce all the plots, download ERA5 data files which contain (6-hourly, 1° x 1° spatial resolution at all pressure levels 
(37 available levels), from June 1-30, 2021):

- geopotential height (z)
- zonal wind  (u)
- meridional wind (v)
- temperature (t)
- 2-m temperature (t2m)
- net OLR (mtnlwrf)
- clear-sky OLR (mtnlwrfcs)
- total column water (tcw)
- total column water vapor (tcwv)
- sea level pressure (sp)

## Scripts
 
The run script `sample_run_script.py` called functions in `QGField` object (see `hn2016_falwa/oopinterface.py`) 
to compute LWA and fluxes, while `scripts/graph_plot_module.py` contains the graph plotting functions to reproduce 
all the figures in the paper.

## Note on equatorward boundary condition used

Note that there is a difference in computing reference state in this most recent manuscript. In the past (and also 
the analysis in NH18 Science), we used equator as the latitudinal boundary. In this current version of the script 
(release 0.6.0), we use absolute vorticity at 5°N as boundary condition such that the solution is no longer sensitive 
to fields at the equator, which improves the quality of the analysis.

## How to create a virtual environment to run this script

Create a conda environment by running:
```bash
conda env create -f scripts/nhn_grl2022/env_nhn_grl2022.yml
```

Activate the environment
```bash
conda activate env_nhn_grl2022
```

Remove any existing compiled fortran modules by:
```bash
rm falwa/*.so
```

Install hn2016_falwa by running in command line:
```bash
python setup.py develop
```

You can then run the script to reproduce the graphs in the GRL paper:
```bash
python scripts/nhn_grl2022/sample_run_script.py
```

## Problem solving

If you have any questions, please [submit an issue](https://github.com/csyhuang/hn2016_falwa/issues) or email me. 
I'll get back ASAP.

Thanks!

Clare S. Y. Huang (csyhuang@uchicago.edu)