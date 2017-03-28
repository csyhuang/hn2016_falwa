## Python Library: hn2016_falwa

Compute Finite-amplitude Local Wave Activity (FALWA) introduced in [Huang and Nakamura (2016)](http://dx.doi.org/10.1175/JAS-D-15-0194.1) from gridded climate data.

![hn2016_falwa_diagram](https://github.com/csyhuang/csyhuang.github.io/blob/master/hn2016_falwa_diagram.png)

The utilities in the library can also be used to compute the tracer equivalent-latitude relationship proposed in Nakamura (1996) / Allen and Nakamura (2003) and the (zonal mean) finite-amplitude wave activity in spherical geometry as in Nakamura and Solomon (2010).

## Installation

This package works in Python 2.7 (Python 3 version will be included soon). It can be installed with pip:
```
pip install hn2016_falwa
```
You can also install from the source distribution:
```
git clone https://github.com/csyhuang/hn2016_falwa.git
cd hn2016_falwa
python setup.py install
```

## Examples

Please refer to the example/ directory for these ipython notebooks. 

Sample Script | Description
------------ | -------------
Example_barotropic.ipynb | It reads in a sample datasets "barotropic_vorticity.nc", which contains absolute vorticity field snapsnots from a barotropic decay model (Held and Phillips 1987). It computes both the equivalent-latitude relationship (e.g. Nakamura 1996) and local wave activity (Huang and Nakamura 2016) in a global domain.
Example_qgpv.ipynb | It reads in a sample datasets u_QGPV_240hPa_2012Oct28to31.nc", which contains zonal velocity and QGPV field at 240hPa derived form ERA-Interim reanalysis data. Similar to fig. 9 in Huang and Nakamura (2016), a hemispheric domain is used here.

## List of functions (to be updated)

Type help(function) for instructions on format of input variables. Tentative future updates are listed on the [Project page](https://github.com/csyhuang/hn2016_falwa/projects/1).

Function | Description
---------| -------------
barotropic_Eqlat_LWA | Compute local wave activity and corresponding equivalent-latitude profile from absolute vorticity field in a barotropic model with spherical geometry.
qgpv_Eqlat_LWA | Compute local wave activity and corresponding equivalent-latitude profile from quasi-geostrophic potential vortcitiy (QGPV) field in **hemispheric** domains.
EqvLat| Compute Equivalent-latitude relationship from a tracer field on a sphere.
static_stability| Compute static stability in hemispheric domain from 3-D potential temperature field.
Compute_QGPV_GivenVort| Compute quasi-geostrophic potential vorticity as outlined in Huang and Nakamura (2016,JAS), given absolute vorticity and temperature as the inputs.
Solve_URef_noslip_hemisphere| To compute the eddy-free referece state of zonal wind and temperature in Nakamura and Solomon (2010) but in a *hemispheric domain* from wave activity and zonal wind fields, given no-slip lower boundary conditions. Documentation to be updated soon. Please contact me directly if you want assistance using it in a timely manner.

## Inquiries / Issues reporting

Please make inquiries about / report issues with the package on the [Issues page](https://github.com/csyhuang/hn2016_falwa/issues). If you need help analyzing output from particular model/analysis with our techniques, feel free to email me <clare1068@gmail.com> with sample datasets and I can configure the code for you.

