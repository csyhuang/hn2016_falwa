## Demo script for the analyses done in Neal et al. (submitted to GRL)

This repo contains the sample script to reproduce the plots in Neal et al. "The 2021 Pacific Northwest heat wave and 
associated blocking: Meteorology and the role of an upstream cyclone as a diabatic source of wave activity". 

To produce all the plots, required ERA5 data (6-hourly, 1° x 1° spatial resolution at all pressure levels 
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
 
The run script `sample_run_script.py` called functions in `QGField` object (see `hn2016_falwa/oopinterface.py`) 
to compute LWA and fluxes, while `scripts/graph_plot_module.py` contains the graph plotting functions to reproduce 
all the figures in the paper.

Note that there is a difference in computing reference state in this most recent manuscript. In the past, and also 
the analysis in NH18 Science, we used equator as the latitudinal boundary. In this current version of the script 
(release 0.6.1), we use absolute vorticity at 5°N as boundary condition such that the solution is no longer sensitive 
to fields at the equator, which improves the quality of the analysis.

We are in a hurry submitting the manuscript, so this version of code (v0.6.1) has not been properly refactored yet. 
If you have any questions, please [submit an issue](https://github.com/csyhuang/hn2016_falwa/issues) or email me. 
I'll get back ASAP.

Thanks!

Clare (csyhuang@protonmail.com)