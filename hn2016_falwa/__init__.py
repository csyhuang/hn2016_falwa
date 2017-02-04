# To make it a package
# This repository contains python modules that can be used to compute the Local Finite-Amplitude Wave Activity proposed in Huang & Nakamura (2016).
# HN2016_FALWA is a python module developed for readers to implement local finite-amplitude wave activity calculations proposed in Huang & Nakamura (2016, JAS) (http://dx.doi.org/10.1175/JAS-D-15-0194.1) on climate datasets. 
# See/Run Example_plotting_script.py for example usage of this module to reproduce figures in HN2015.

# outer __init__.py
from barotropic_lwa import barotropic_Eqlat_LWA
from qgpv_lwa import qgpv_Eqlat_LWA
from lwa import LWA
from eqv_lat import EqvLat, EqvLat_hemispheric
from Solve_URef_noslip_hemisphere import Solve_Uref_noslip
from static_stability import static_stability

