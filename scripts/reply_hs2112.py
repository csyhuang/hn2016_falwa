import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from hn2016_falwa.oopinterface import QGField
data = xr.open_dataset("export_tstep_0.nc")
coarsened = data.isel({
    "latitude":  slice(None, None, 4),
    "longitude": slice(None, None, 4)})

uu = coarsened.variables['u'][:, ::-1, :]
vv = coarsened.variables['v'][:, ::-1, :]
tt = coarsened.variables['t'][:, ::-1, :]
xlon = np.arange(360)
ylat = np.arange(-90, 91, 1)
plev = coarsened.variables['level']
kmax = 17
dz = 1000.
height = np.arange(0, kmax)*dz

eq_boundary_index = 5  # use domain from 5N to North pole
qgfield_object = QGField(xlon, ylat, plev, uu, vv, tt, kmax=kmax, dz=dz, eq_boundary_index=eq_boundary_index)

qgpv_temp, interpolated_u_temp, interpolated_v_temp, interpolated_avort_temp, interpolated_theta_temp, \
static_stability_n, static_stability_s, tn0, ts0 = qgfield_object._interpolate_field_dirinv()

qref, uref, tref, fawa, ubar, tbar = qgfield_object._compute_qref_fawa_and_bc()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
cs1 = ax1.contourf(ylat[90:], height, np.swapaxes(qref, 0, 1), 30, cmap='rainbow')
ax1.set_title('Qref')
cbar1 = fig.colorbar(cs1)
cs2 = ax2.contourf(ylat[90+eq_boundary_index:], height, np.swapaxes(uref, 0, 1), np.arange(-20, 30, 5), cmap='rainbow')
ax2.set_title('Uref')
cbar2 = fig.colorbar(cs2)
plt.show()

# astarbaro, ubaro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4baro, astar1, astar2 = \
#     qgfield_object._compute_lwa_flux_dirinv(qref, uref, tref)

