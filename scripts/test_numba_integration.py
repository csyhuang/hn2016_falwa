#!/usr/bin/env python
"""Integration test to verify Numba modules work correctly in oopinterface."""

import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNH18, QGFieldNHN22

# Load test data
data_dir = 'tests/data'
u_data = xr.open_dataset(f'{data_dir}/2005-01-23-0000-u.nc')
v_data = xr.open_dataset(f'{data_dir}/2005-01-23-0000-v.nc')
t_data = xr.open_dataset(f'{data_dir}/2005-01-23-0000-t.nc')

xlon = u_data.longitude.values
ylat = u_data.latitude.values
if np.diff(ylat)[0] < 0:
    ylat = ylat[::-1]

plev = u_data.level.values
if np.diff(plev)[0] > 0:
    plev = plev[::-1]

uu = u_data.u.values[:, ::-1, :]
vv = v_data.v.values[:, ::-1, :]
tt = t_data.t.values[:, ::-1, :]

print('Testing QGFieldNH18...')
qgfield_nh18 = QGFieldNH18(xlon, ylat, plev, uu, vv, tt, northern_hemisphere_results_only=False)
qgfield_nh18.interpolate_fields(return_named_tuple=False)
print(f'  QGPV shape: {qgfield_nh18.qgpv.shape}')
print(f'  QGPV range: {qgfield_nh18.qgpv.min():.6e} to {qgfield_nh18.qgpv.max():.6e}')
qgfield_nh18.compute_reference_states(return_named_tuple=False)
print(f'  qref shape: {qgfield_nh18.qref.shape}')
print('  QGFieldNH18: OK')

print()
print('Testing QGFieldNHN22...')
qgfield_nhn22 = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, northern_hemisphere_results_only=False)
qgfield_nhn22.interpolate_fields(return_named_tuple=False)
print(f'  QGPV shape: {qgfield_nhn22.qgpv.shape}')
print(f'  QGPV range: {qgfield_nhn22.qgpv.min():.6e} to {qgfield_nhn22.qgpv.max():.6e}')
qgfield_nhn22.compute_reference_states(return_named_tuple=False)
print(f'  qref shape: {qgfield_nhn22.qref.shape}')
print('  QGFieldNHN22: OK')

print()
print('All tests passed! The Numba modules are working correctly in oopinterface.')

