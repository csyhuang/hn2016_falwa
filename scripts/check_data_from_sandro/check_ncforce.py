import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from falwa.oopinterface import QGFieldNH18, QGFieldNHN22
from cartopy.crs import PlateCarree
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.util import add_cyclic_point
from falwa.constant import *

data_path = "/Users/claresyhuang/Dropbox/GitHub/hn2016_falwa/github_data_storage/for_clare_ncforce_e3sm/"

uvt_qrl_data = xr.open_mfdataset([
    f"{data_path}u.1902.nc",
    f"{data_path}v.1902.nc",
    f"{data_path}t.1902.nc",
    f"{data_path}qrl.1902.nc"])
qdot_data = xr.open_dataset(
    "/Users/claresyhuang/Dropbox/GitHub/hn2016_falwa/github_data_storage/for_clare_ncforce_e3sm/qdot_qrl.1902.nc")

"""
Dimensions:  (time: 31, lon: 360, lat: 181, level: 37)
Coordinates:
  * time     (time) object 1902-01-01 09:00:00 ... 1902-01-31 09:00:00
  * lon      (lon) float64 0.0 1.0 2.0 3.0 4.0 ... 355.0 356.0 357.0 358.0 359.0
  * lat      (lat) float64 -90.0 -89.0 -88.0 -87.0 -86.0 ... 87.0 88.0 89.0 90.0
  * level    (level) float32 1e+03 975.0 950.0 925.0 900.0 ... 5.0 3.0 2.0 1.0
Data variables:
    T        (time, level, lat, lon) float32 dask.array<chunksize=(31, 37, 181, 360), meta=np.ndarray>
    U        (time, level, lat, lon) float32 dask.array<chunksize=(31, 37, 181, 360), meta=np.ndarray>
    V        (time, level, lat, lon) float32 dask.array<chunksize=(31, 37, 181, 360), meta=np.ndarray>
Attributes:
    CDI:          Climate Data Interface version 1.9.10 (https://mpimet.mpg.d...
    Conventions:  CF-1.6
    history:      Sat Jun 08 16:28:57 2024: cdo selmon,1 ../../../../t/intp/r...
    CDO:          Climate Data Operators version 1.9.10 (https://mpimet.mpg.d...
"""

plev = uvt_qrl_data.coords['level'].values
zlev = -SCALE_HEIGHT * np.log(plev / P_GROUND)
dz = 1000.
kmax = 49
height = np.arange(0, kmax) * dz
new_plev = P_GROUND * np.exp(-height / SCALE_HEIGHT)
tstep = 0
original_temp = uvt_qrl_data.variables["T"].isel(time=tstep).values
new = uvt_qrl_data.interp(coords={"level": new_plev})
interp_to_regular_zgrid = lambda field_to_interp: interp1d(
    zlev, field_to_interp, axis=0, kind='linear', fill_value="extrapolate")(height)
uu = new.variables["U"].isel(time=tstep)
vv = new.variables["V"].isel(time=tstep)
tt = new.variables["T"].isel(time=tstep)
qrl = new.variables['QRL'].isel(time=tstep)
qq = qdot_data.variables["Q"].isel(time=tstep)
xlon = new.coords['lon'].values
ylat = new.coords['lat'].values


def calculate_static_stability(temperature, clat):
    """
        ! reference theta
    do kk = 1,kmax
        t0(kk) = 0.
        csm = 0.
        do j = 1,nlat
            phi0 = -90.+float(j-1)*180./float(nlat-1)
            phi0 = phi0*pi/180.
            t0(kk) = t0(kk) + tzd(j,kk)*cos(phi0)
            csm = csm + cos(phi0)
        enddo
        t0(kk) = t0(kk)/csm

    Parameters
    ----------
    temperature
    area

    Returns
    -------

    """
    csm = clat.sum()
    t0 = np.mean(temperature * clat[np.newaxis, :, np.newaxis], axis=-1).sum(axis=-1) / csm
    # t0.shape = temperature.shape[0]
    return t0


def plot_two_graphs(cal_qdot_2d, int_qdot):
    # *** Plot on map distribution ***
    projection = PlateCarree(central_longitude=180.)
    transform = PlateCarree()
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), subplot_kw={"projection": projection})

    axs[0].set_title(f"Qdot_QRL at lowest level")
    QGPV_plot, lons = add_cyclic_point(cal_qdot_2d, xlon)
    cs = axs[0].contourf(lons, ylat, QGPV_plot, 21, transform=transform, cmap="turbo")
    cb = fig.colorbar(cs, ax=axs[0])

    axs[1].set_title(f"Vertically averaged Qdot")
    LWA_plot, lons = add_cyclic_point(int_qdot, xlon)
    cs = axs[1].contourf(lons, ylat, LWA_plot, 21, transform=transform, cmap="turbo")
    cb = fig.colorbar(cs, ax=axs[1])

    for ax in axs:
        ax.coastlines()
        ax.set_xticks([0, 60, 120, 180, 240, 300], crs=transform)
        ax.set_yticks([30, 40, 50, 60, 70, 80], crs=transform)
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
        ax.gridlines()
        ax.set_extent([0., 360., 20., 81.], crs=transform)
        ax.set_aspect(2.2)
    plt.show()


qgfield_object = QGFieldNHN22(
    xlon, ylat, new_plev, uu, vv, tt,
    northern_hemisphere_results_only=False,
    data_on_evenly_spaced_pseudoheight_grid=True)
equator_idx = qgfield_object.equator_idx
qgfield_object.interpolate_fields(return_named_tuple=False)
qgfield_object.compute_reference_states(return_named_tuple=False)
qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=qq)
cal_qdot = np.zeros_like(qrl)
# static_stability = qgfield_object.static_stability  # HN18
# static_stability = 0.5*(qgfield_object.static_stability[0] + qgfield_object.static_stability[1])  # NHN22 average of both hem
static_stability = qgfield_object.static_stability[1]  # NHN22 NHem static stability

qgfield_object2 = QGFieldNHN22(
    xlon, ylat, new_plev, uu, vv, tt,
    northern_hemisphere_results_only=False,
    data_on_evenly_spaced_pseudoheight_grid=True)
qgfield_object2.interpolate_fields(return_named_tuple=False)
qgfield_object2.compute_reference_states(return_named_tuple=False)


# *** Compute static stability from original data ***
original_theta = original_temp * np.exp(DRY_GAS_CONSTANT / CP * zlev[:, np.newaxis, np.newaxis] / SCALE_HEIGHT)
t0_from_data = calculate_static_stability(original_theta, np.cos(np.deg2rad(ylat)))
uni_spline = UnivariateSpline(x=zlev, y=t0_from_data)
uni_spline_derivative = uni_spline.derivative()
plt.plot(t0_from_data, zlev, "bx-"); plt.plot(qgfield_object2._domain_average_storage.tn0, height, "rx-"); plt.show()
plt.plot(uni_spline_derivative(zlev), zlev, "b"); plt.plot(qgfield_object2.static_stability[1], height, "r"); plt.title('static stability');plt.show()
recompute_static_stability = uni_spline_derivative(height)

cal_qdot[1:-1, :, :] = \
    2. * EARTH_OMEGA * np.sin(np.deg2rad(ylat[np.newaxis, :, np.newaxis])) \
    * np.exp(height[1:-1, np.newaxis, np.newaxis] / SCALE_HEIGHT) * \
    (np.exp(-height[2:, np.newaxis, np.newaxis] / SCALE_HEIGHT) * qrl[2:, :, :]/recompute_static_stability[2:, np.newaxis, np.newaxis]
     - np.exp(-height[:-2, np.newaxis, np.newaxis] / SCALE_HEIGHT) * qrl[:-2, :, :] / recompute_static_stability[:-2, np.newaxis, np.newaxis]) / (2.*dz)
cal_qdot[0, :, :] = 2 * cal_qdot[1, :, :] - cal_qdot[2, :, :]
cal_qdot[-1, :, :] = 2 * cal_qdot[-2, :, :] - cal_qdot[-3, :, :]
qgfield_object2.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=cal_qdot)

plot_two_graphs(cal_qdot[0, :, :], qgfield_object2.ncforce_baro)
print("Pause")

# # *** Compare different layers ***
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
# from_qq = qgfield_object.ncforce_baro
# from_cal_qdot = qgfield_object2.ncforce_baro
# max_value = max([np.abs(from_qq).max(), np.abs(from_cal_qdot).max()])
# caxis = np.linspace(-max_value, max_value, 21, endpoint=True)
# cx1 = ax1.contourf(xlon, ylat, from_cal_qdot, caxis, cmap='bwr')
# plt.colorbar(cx1, ax=ax1)
# ax1.set_title(f"from cal_qdot")
# cx2 = ax2.contourf(xlon, ylat, from_qq, caxis, cmap='bwr')
# plt.colorbar(cx2, ax=ax2)
# ax2.set_title(f"Sandro's cal from_qq")
# plt.tight_layout()
# plt.savefig(f"comparison_NHN22_integrated_ncforce.png")
# plt.show()


# for k_level in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40]:
#     # *** Compare different layers ***
#     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
#     max_value = max([np.abs(qq[k_level, :, :]).max(), np.abs(cal_qdot[k_level, :, :]).max()])
#     caxis = np.linspace(-max_value, max_value, 21, endpoint=True)
#     cx1 = ax1.contourf(xlon, ylat, cal_qdot[k_level, :, :], caxis, cmap='bwr')
#     plt.colorbar(cx1, ax=ax1)
#     ax1.set_title(f"my cal at k={k_level} (NHN22)")
#     cx2 = ax2.contourf(xlon, ylat, qq[k_level, :, :], caxis, cmap='bwr')
#     plt.colorbar(cx2, ax=ax2)
#     ax2.set_title(f"Sandro's cal at k={k_level}")
#     plt.tight_layout()
#     plt.savefig(f"comparison_NHN22_at_k_{k_level}.png")
#     plt.show()

print("Pause")






