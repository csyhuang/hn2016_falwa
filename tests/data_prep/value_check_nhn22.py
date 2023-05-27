import os
import numpy as np
from math import pi
import xarray as xr
from hn2016_falwa.oopinterface import QGField, Protocol


def extract_data_from_notebook_dir(dir: str, test_data_dir: str):
    for var in ['u', 'v', 't']:
        ds_disk = xr.open_dataset(f"{dir}2005-01-23_to_2005-01-30_{var}.nc")
        field = ds_disk.isel(time=0)
        field.to_netcdf(f"{test_data_dir}2005-01-23-0000-{var}.nc")


test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data"
output_netcdf_file = test_data_dir + "/expected_values_nhn22.nc"

if __name__ == "__main__":
    u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc")
    v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc")
    t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc")

    # *** Set coordinates ***
    xlon = u_file.longitude.values
    ylat = u_file.latitude.values[::-1]  # latitude has to be in ascending order
    plev = u_file.level.values[::-1]  # pressure level has to be in descending order (ascending height)

    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size

    # *** Set other parameters and constants ***
    clat = np.cos(np.deg2rad(ylat))  # cosine latitude
    p0 = 1000.  # surface pressure [hPa]
    kmax = 49  # number of grid points for vertical extrapolation (dimension of height)
    dz = 1000.  # differential height element
    height = np.arange(0, kmax) * dz  # pseudoheight [m]
    dphi = np.diff(ylat)[0] * pi / 180.  # differential latitudinal element
    dlambda = np.diff(xlon)[0] * pi / 180.  # differential latitudinal element
    hh = 7000.  # scale height
    cp = 1004.  # heat capacity of dry air
    rr = 287.  # gas constant
    omega = 7.29e-5  # rotation rate of the earth
    aa = 6.378e+6  # earth radius
    prefactor = np.array([np.exp(-z / hh) for z in height[1:]]).sum()  # integrated sum of density from the level
    # just above the ground (z=1km) to aloft
    npart = nlat  # number of partitions to construct the equivalent latitude grids
    maxits = 100000  # maximum number of iteration in the SOR solver to solve for reference state
    tol = 1.e-5  # tolerance that define convergence of solution
    rjac = 0.95  # spectral radius of the Jacobi iteration in the SOR solver.
    jd = nlat // 2 + 1  # (one plus) index of latitude grid point with value 0 deg
    eq_boundary_index = 5

    # This is to be input to fortran code. The index convention is different.

    uu = u_file.u.values
    vv = v_file.v.values
    tt = t_file.t.values

    qgfield_object = QGField(
        xlon, ylat, plev,
        uu[::-1, ::-1, :], vv[::-1, ::-1, :], tt[::-1, ::-1, :],
        eq_boundary_index=eq_boundary_index, protocol=Protocol.NHN22,
        northern_hemisphere_results_only=False)
    qgfield_object.interpolate_fields()
    qgfield_object.compute_reference_states()
    qgfield_object.compute_lwa_and_barotropic_fluxes()

    # *** Output test data ***
    ds = xr.Dataset({
        "qgpv": (("levelist", "latitude", "longitude"), qgfield_object.qgpv),
        "lwa": (("levelist", "latitude", "longitude"), qgfield_object.lwa),
        "static_stability_s": (("levelist"), qgfield_object.static_stability[0]),
        "static_stability_n": (("levelist"), qgfield_object.static_stability[1]),
        "qref": (("levelist", "latitude"), qgfield_object.qref),
        "uref": (("levelist", "latitude"), qgfield_object.uref),
        "ptref": (("levelist", "latitude"), qgfield_object.ptref),
        "u_baro": (("latitude", "longitude"), qgfield_object.u_baro),
        "lwa_baro": (("latitude", "longitude"), qgfield_object.lwa_baro)
    }, coords={
        "levelist": p0 * np.exp(-height / hh),
        "latitude": ylat,
        "longitude": xlon})

    ds.to_netcdf(output_netcdf_file)
    print(f"Finished preparing {output_netcdf_file}")
