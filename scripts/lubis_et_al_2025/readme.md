# Direct Diabatic Heating Calculations

This directory contains the python script that calls `falwa.oopinterface.QGFieldNHN22` to calculate direct diabatic heating rates from MEERA2 data presented in:

> Lubis, et al. 2025, "Cloud-Radiative Effects Significantly Increase Wintertime Atmospheric Blocking in the Euro-Atlantic Sector" (in review, Nature Communications)

This repository only contains the python scripts calling `falwa`. The repository for publication which contains the full set of scripts used to generate the figures (NCL) and analysis is https://github.com/sandrolubis/NC_Blocking_CRE/ .

# Requirements

Use `environment.yml` in this repository to create a conda environment with the required dependencies. The minimum required version of `falwa` is v2.2.0.

