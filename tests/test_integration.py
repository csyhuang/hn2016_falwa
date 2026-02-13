"""
These are integration tests that mirrors the implementation in notebook examples.
This shall be merged with test_output_results.py in the next iterations.
"""
import pytest
from math import pi
import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNHN22
from falwa.xarrayinterface import QGDataset


@pytest.fixture(scope="module")
def merra_uvt_data(test_data_dir):
    """
    load sampled MERRA2 data
    """
    merra_uvt_data = xr.open_mfdataset(f"{test_data_dir}/[UVT].daily.merra.sample.nc")
    return merra_uvt_data


@pytest.fixture(scope="module")
def merra_dtdtlwr_data(test_data_dir):
    """
    load sampled MERRA2 diabatic heating data
    """
    merra_dtdtlwr_data = xr.open_mfdataset(f"{test_data_dir}/DTDTLWR.daily.merra.sample.nc")
    return merra_dtdtlwr_data


def test_qgfield_nhn22_ncforce_integration(merra_uvt_data, merra_dtdtlwr_data):
    """
    Mirroring the implementation in notebooks/lubis_et_al_2025/ncforce_qgfield_2.1.0.ipynb
    """
    import falwa
    print(f"falwa.__version__: {falwa.__version__}")

    # Prepare coordinates
    xlon = merra_uvt_data['lon'].values
    ylat = merra_uvt_data['lat'].values  # latitude has to be in ascending order
    plev = merra_uvt_data['level'].values  # pressure level has to be in descending order (ascending height)

    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size
    print(f"nlon: {nlon}, nlat: {nlat}, nlev:{nlev}")

    # Get once slice of data to test
    uu = merra_uvt_data['U'].values[0, :, :, :]
    vv = merra_uvt_data['V'].values[0, :, :, :]
    tt = merra_uvt_data['T'].values[0, :, :, :]
    dtdtlwr = merra_dtdtlwr_data['DTDTLWR'].values[0, :, :, :]

    qgfield_nhn22 = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, northern_hemisphere_results_only=False, eq_boundary_index=5)
    qgfield_nhn22.interpolate_fields(return_named_tuple=False)
    qgfield_nhn22.compute_reference_states(return_named_tuple=False)

    # New procedure: compute q_dot with the discretization scheme in qgfield_nhn22.
    # This is handled by the method compute_ncforce_from_heating_rate
    ncforce = qgfield_nhn22.compute_ncforce_from_heating_rate(heating_rate=dtdtlwr)

    # Existing func: Pass in the resultant ncforce term into "compute_lwa_and_barotropic_fluxes" would compute the barotropic component
    qgfield_nhn22.compute_layerwise_lwa_fluxes(ncforce=ncforce)
    qgfield_nhn22.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=ncforce)

    # Assert all flux fields are computed
    num_of_nan_allowed: int = 50  # TODO: ensure no NANs in concerned domain
    assert np.isnan(qgfield_nhn22.convergence_zonal_advective_flux).sum() < num_of_nan_allowed
    assert np.abs(np.nan_to_num(qgfield_nhn22.convergence_zonal_advective_flux)).sum() > 0
    assert np.isnan(qgfield_nhn22.ncforce_baro).sum() < num_of_nan_allowed
    assert np.abs(np.nan_to_num(qgfield_nhn22.ncforce_baro)).sum() > 0


def test_qgdataset_nhn22_ncforce_integration(merra_uvt_data, merra_dtdtlwr_data):
    """
    Mirroring the implementation in notebooks/lubis_et_al_2025/ncforce_qgdataset_2.1.0.ipynb
    """

    qgds = QGDataset(merra_uvt_data, qgfield=QGFieldNHN22)

    qgds.interpolate_fields()
    qgds.compute_reference_states()
    ncforce = qgds.compute_ncforce_from_heating_rate(heating_rate=merra_dtdtlwr_data['DTDTLWR'])
    qgds.compute_lwa_and_barotropic_fluxes(ncforce=ncforce)

    convergence_zonal_advective_flux = qgds.convergence_zonal_advective_flux.isel(time=0)
    ncforce_baro = qgds.ncforce_baro.isel(time=0)

    # Assert all flux fields are computed
    assert np.isnan(convergence_zonal_advective_flux).sum() < 30  # TODO: ensure no NANs in concerned domain
    assert np.abs(np.nan_to_num(convergence_zonal_advective_flux)).sum() > 0
    assert np.isnan(ncforce_baro).sum() < 30
    assert np.abs(np.nan_to_num(ncforce_baro)).sum() > 0


# **** BarotropicField example ****
@pytest.fixture(scope="module")
def barotropic_field_data(test_data_dir):
    """
    load sampled barotropic vorticity data
    """
    barotropic_field_data = xr.open_dataset(
        f"{test_data_dir}/barotropic_vorticity.nc")  # This is a soft link to notebook data
    return barotropic_field_data


def test_barotropic_field(barotropic_field_data):
    from falwa.barotropic_field import BarotropicField

    # === Load data and coordinates ===
    abs_vorticity = barotropic_field_data.absolute_vorticity.values

    xlon = np.linspace(0, 360., 512, endpoint=False)
    ylat = np.linspace(-90, 90., 256, endpoint=True)

    cc1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)  # area computed in the class assumed uniform grid
    cc1_eqvlat = cc1.equivalent_latitudes  # Compute Equivalent Latitudes
    cc1_lwa = cc1.lwa  # Compute Local Wave Activity

    cc2 = BarotropicField(xlon, ylat, pv_field=abs_vorticity, return_partitioned_lwa=True)  # area computed in the class assumed uniform grid
    cc2_eqvlat = cc2.equivalent_latitudes  # Compute Equivalent Latitudes
    cc2_lwa = cc2.lwa  # Compute Local Wave Activity

    assert np.isclose(cc1_eqvlat, cc2_eqvlat).all()
    assert np.isclose(cc1_lwa, cc2_lwa.sum(axis=0)).all()
