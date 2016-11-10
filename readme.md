## Description

The package hn2016_falwa contains modules to compute the finite-amplitude local wave activity (FALWA) proposed in [Huang and Nakamura (2016, JAS)](http://dx.doi.org/10.1175/JAS-D-15-0194.1). 
The utilities in the library can be used to compute the tracer equivalent-latitude relationship proposed in Nakamura (1996) / Allen and Nakamura (2003) and the (zonal mean) finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

## Examples

You can start using the package by running the two examples provided with the command $ python sample_script.py.

Sample Script | Description
------------ | -------------
Example_barotropic.ipynb <br> [HTML version](http://htmlpreview.github.com/?https://github.com/csyhuang/hn2016_falwa/blob/master/Example_barotropic.html) | It reads in a sample datasets "barotropic_vorticity.nc", which contains absolute vorticity field snapsnots from a barotropic decay model (Held and Phillips 1987). It computes both the equivalent-latitude relationship (e.g. Nakamura 1996) and local wave activity (Huang and Nakamura 2016) in a global domain.
Example_qgpv.ipynb <br> [HTML version](http://htmlpreview.github.com/?Example_qgpv.html) | It reads in a sample datasets u_QGPV_240hPa_2012Oct28to31.nc", which contains zonal velocity and QGPV field at 240hPa derived form ERA-Interim reanalysis data. Similar to fig. 9 in Huang and Nakamura (2016), a hemispheric domain is used here.

## List of functions (to be updated)

Function | Description
---------| -------------
barotropic_Eqlat_LWA | Compute local wave activity and corresponding equivalent-latitude profile from absolute vorticity field in a barotropic model with spherical geometry.
qgpv_Eqlat_LWA | Compute local wave activity and corresponding equivalent-latitude profile from quasi-geostrophic potential vortcitiy (QGPV) field in **hemispheric** domains.
EqvLat| Compute Equivalent-latitude relationship from a tracer field on a sphere.
LWA| Compute wave activity of a given trafer field, given the equivalent-latitude.

## To-do list (to be updated)

Below are functions I plan to add into my library soon. Please feel free to make suggestions.

Function | Description
---------| -------------
vertical_integral | To compute density-weighted vertical average of local wave activity, zonal wind, or other 3-D fields.
Compute_QGPV | To compute QGPV from velocity and temperature fields.
Solve_Uref| To compute the eddy-free referece state of zonal wind and temperature in Nakamura and Solomon (2010) from wave activity and zonal wind fields.

## Contact

Feel free to email Clare S. Y. Huang: clare1068@gmail.com if you have any inquiries/suggestions.
