# Direct Diabatic Heating Calculations

This directory contains the python script that calls `falwa.oopinterface.QGFieldNHN22` to calculate layerwise (per pressure level) direct diabatic heating rates from MEERA2 data presented in:

> Lubis, et al. 2025, "Cloud-Radiative Effects Significantly Increase Wintertime Atmospheric Blocking in the Euro-Atlantic Sector" (accepted, Nature Communications)

This repository only contains the python scripts calling `falwa`. The repository for publication which contains the full set of scripts used to generate the figures (NCL) and analysis is https://github.com/sandrolubis/NC_Blocking_CRE/ .

# Requirements

When the analyses were performed, the environment used was from an earlier version of `environment.yml` shown below. The minimum required version of `falwa` is v2.2.0.

```
name: falwa_env

channels:
- conda-forge
- defaults

dependencies:
- python=3.10
- numpy=1.22.3
- scipy=1.9
- pytest=7.4.0
- xarray=2023.2.0
- netCDF4=1.5.8
- gridfill
- jupyter
- matplotlib
- twine
- cartopy
```

