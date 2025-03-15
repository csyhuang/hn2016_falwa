import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNHN22
from falwa.xarrayinterface import QGDataset


def test_qgfield_nhn22_ncforce_integration(test_data_dir):
    """
    Mirroring the implementation in notebooks/lubis_et_al_2024/ncforce_qgfield_2.1.0.ipynb
    """

    # Load MERRA2 dataset
    uvt_data = xr.open_mfdataset(f"{test_data_dir}/[UVT].daily.merra.sample.nc")
    dtdtlwr_data = xr.open_mfdataset(f"{test_data_dir}/DTDTLWR.daily.merra.sample.nc")

    # Prepare coordinates
    xlon = uvt_data['lon'].values
    ylat = uvt_data['lat'].values  # latitude has to be in ascending order
    plev = uvt_data['level'].values  # pressure level has to be in descending order (ascending height)

    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size
    print(f"nlon: {nlon}, nlat: {nlat}, nlev:{nlev}")

    # Get once slice of data to test
    uu = uvt_data['U'].values[0, :, :, :]
    vv = uvt_data['V'].values[0, :, :, :]
    tt = uvt_data['T'].values[0, :, :, :]
    dtdtlwr = dtdtlwr_data['DTDTLWR'].values[0, :, :, :]

    qgfield_nhn22 = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, northern_hemisphere_results_only=False, eq_boundary_index=5)
    qgfield_nhn22.interpolate_fields(return_named_tuple=False)
    qgfield_nhn22.compute_reference_states(return_named_tuple=False)

    # New procedure: compute q_dot with the discretization scheme in qgfield_nhn22.
    # This is handled by the method compute_ncforce_from_heating_rate
    ncforce = qgfield_nhn22.compute_ncforce_from_heating_rate(heating_rate=dtdtlwr)

    # Existing func: Pass in the resultant ncforce term into "compute_lwa_and_barotropic_fluxes" would compute the barotropic component
    qgfield_nhn22.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=ncforce)

    # Assert all flux fields are computed
    assert np.isnan(qgfield_nhn22.convergence_zonal_advective_flux).sum() < 30  # TODO: ensure no NANs in concerned domain
    assert np.abs(np.nan_to_num(qgfield_nhn22.convergence_zonal_advective_flux)).sum() > 0
    assert np.isnan(qgfield_nhn22.ncforce_baro).sum() < 30
    assert np.abs(np.nan_to_num(qgfield_nhn22.ncforce_baro)).sum() > 0


def test_qgdataset_nhn22_ncforce_integration(test_data_dir):
    """
    Mirroring the implementation in notebooks/lubis_et_al_2024/ncforce_qgdataset_2.1.0.ipynb
    """

    # Load MERRA2 dataset
    uvt_data = xr.open_mfdataset(f"{test_data_dir}/[UVT].daily.merra.sample.nc")
    dtdtlwr_data = xr.open_mfdataset(f"{test_data_dir}/DTDTLWR.daily.merra.sample.nc")

    qgds = QGDataset(uvt_data, qgfield=QGFieldNHN22)

    qgds.interpolate_fields()
    qgds.compute_reference_states()
    ncforce = qgds.compute_ncforce_from_heating_rate(heating_rate=dtdtlwr_data['DTDTLWR'])
    qgds.compute_lwa_and_barotropic_fluxes(ncforce=ncforce)

    convergence_zonal_advective_flux = qgds.convergence_zonal_advective_flux.isel(time=0)
    ncforce_baro = qgds.ncforce_baro.isel(time=0)

    # Assert all flux fields are computed
    assert np.isnan(convergence_zonal_advective_flux).sum() < 30  # TODO: ensure no NANs in concerned domain
    assert np.abs(np.nan_to_num(convergence_zonal_advective_flux)).sum() > 0
    assert np.isnan(ncforce_baro).sum() < 30
    assert np.abs(np.nan_to_num(ncforce_baro)).sum() > 0
