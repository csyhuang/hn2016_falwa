#!/usr/bin/env python
"""Quick test to verify compute_reference_states Numba module works."""

import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNH18

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

uu = u_data.u.values[::-1, ::-1, :]
vv = v_data.v.values[::-1, ::-1, :]
tt = t_data.t.values[::-1, ::-1, :]

print('Testing QGFieldNH18 with Numba compute_reference_states...')
qgfield = QGFieldNH18(xlon, ylat, plev, uu, vv, tt, northern_hemisphere_results_only=False)

print('  Interpolating fields...')
qgfield.interpolate_fields(return_named_tuple=False)
print(f'    QGPV shape: {qgfield.qgpv.shape}')

print('  Computing reference states (using Numba module)...')
qgfield.compute_reference_states(return_named_tuple=False)
print(f'    qref shape: {qgfield.qref.shape}')
print(f'    uref shape: {qgfield.uref.shape}')
print(f'    ptref shape: {qgfield.ptref.shape}')

print()
print('SUCCESS: compute_reference_states Numba module is working!')

