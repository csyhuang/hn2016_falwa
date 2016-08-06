The package hn2016_falwa contains modules to compute the finite-amplitude local wave activity (FALWA) proposed in [Huang and Nakamura (2016, JAS)](http://dx.doi.org/10.1175/JAS-D-15-0194.1).

The current version of the library handles calculation of FALWA in a barotropic model and QGPV fields on isobaric surfaces with spherical geometry.

The functions in this library can compute the tracer equivalent-latitude relationship proposed in Nakamura (1996) (Also, see Allen and Nakamura (2003)) and the (zonal mean) finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

You can start using the package by running the two examples provided:

Sample Script | Description
------------ | -------------
Example_barotropic.py | Read in a sample datasets "barotropic_vorticity.nc", which contains absolute vorticity field snapsnots from a barotropic decay model (Held and Phillips 1987). It computes both the equivalent-latitude relationship (e.g. Nakamura 1996) and local wave activity (Huang and Nakamura 2016) in a global domain.
Example_qgpv.py | Read in a sample datasets u_QGPV_240hPa_2012Oct28to31.nc", which contains zonal velocity and QGPV field at 240hPa derived form ERA-Interim reanalysis data. Similar to fig. 9 in Huang and Nakamura (2016), a hemispheric domain is used here.

If you have any inquiries/suggestions, please email Clare S. Y. Huang: clare1068@gmail.com