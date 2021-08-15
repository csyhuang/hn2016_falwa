import numpy as np

from hn2016_falwa.barotropic_field import BarotropicField
from hn2016_falwa.constant import EARTH_OMEGA

# === Parameters specific for testing the qgfield class ===
nlat = 31
nlon = 60
xlon = np.linspace(0, 360, nlon, endpoint=False)
ylat = np.linspace(-90., 90., nlat, endpoint=True)
plev = np.array([1000,  900,  800,  700,  600,  500,  400,  300,  200, 100,   10,    1])
nlev = plev.size
kmax = 49

# === Construct test vorticity field ===
zeta_0 = 8.e-5
sigma = 10    # in degree
ylat_0 = 36   # in degree
long_var = np.cos(3*np.deg2rad(xlon))
lat_var = zeta_0 * np.cos(np.deg2rad(ylat)) * np.exp(-(ylat-ylat_0)**2/sigma**2)
pv_field = np.multiply(lat_var.reshape(nlat, 1), long_var.reshape(1, nlon)) \
           + 2 * EARTH_OMEGA * np.sin(np.deg2rad(ylat[:, np.newaxis]))

# === Store values for tests ===
answer_key = dict()
answer_key['eqv_lat'] = [
    -1.45800000e-04, -1.42949614e-04, -1.40099229e-04, -1.37248843e-04,
    -1.33297467e-04, -1.28392619e-04, -1.20907978e-04, -1.11063638e-04,
    -1.01211730e-04, -8.14807659e-05, -7.16202750e-05, -6.17571838e-05,
    -4.20180950e-05, -3.21514559e-05, -1.24095227e-05, -2.54052114e-06,
    1.70060059e-05,  2.58382415e-05,  4.22608624e-05,  4.86083082e-05,
    6.36631843e-05,  8.87920495e-05,  1.11234061e-04,  1.17873494e-04,
    1.25186315e-04,  1.31933407e-04,  1.36129493e-04,  1.40325579e-04,
    1.45482242e-04,  1.50420449e-04,  1.50420449e-04]
answer_key['lwa_zonal_mean'] = [
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    2.78559787e-02, 7.10018930e-01, 1.15941636e+00, 1.21253264e+00,
    1.81290612e+00, 0.00000000e+00, 0.00000000e+00, 1.49792983e+00,
    0.00000000e+00, 1.20073402e+00, 0.00000000e+00, 1.69682067e+00,
    0.00000000e+00, 3.18338839e+00, 5.71579135e+00, 1.32708939e+01,
    2.01584444e+01, 1.95740843e+01, 1.18914496e+01, 6.79453254e+00,
    3.79448248e+00, 2.08640756e+00, 1.20716646e+00, 6.47041885e-01,
    2.62663095e-01, 1.88963620e-16, 0.00000000e+00]
answer_key['lwa_longitudinal_variation'] = [
    41.20590115, 39.32040089, 33.84846603, 25.32572766, 14.58645081,  5.34276464,
    0.,          8.64894682, 16.38582659, 24.10885087, 26.82811918, 24.10885087,
    16.38582659,  8.64894682,  0.,          5.34276464, 14.58645081, 25.32572766,
    33.84846603, 39.32040089, 41.20590115, 39.32040089, 33.84846603, 25.32572766,
    14.58645081,  5.34276464,  0.,          8.64894682, 16.38582659, 24.10885087,
    26.82811918, 24.10885087, 16.38582659,  8.64894682,  0.,          5.34276464,
    14.58645081, 25.32572766, 33.84846603, 39.32040089, 41.20590115, 39.32040089,
    33.84846603, 25.32572766, 14.58645081,  5.34276464,  0.,          8.64894682,
    16.38582659, 24.10885087, 26.82811918, 24.10885087, 16.38582659,  8.64894682,
    0.,          5.34276464, 14.58645081, 25.32572766, 33.84846603, 39.32040089]


def test_barotropic_field():

    barotropic_field = BarotropicField(xlon, ylat, pv_field)

    # *** Test related to equivalent latitdue
    eqv_lat = barotropic_field.equivalent_latitudes
    # Check output shape of equivalent latitude is correct
    assert eqv_lat.shape == (nlat, )
    # Check that equivalent latitude is monotonically increasing
    assert (np.diff(eqv_lat)[1:-1] >= 0.).all()
    # Check the expected value of equivalent latitude
    assert np.allclose(eqv_lat, answer_key['eqv_lat'], rtol=1e-05, atol=1e-08)

    # *** Test related to local wave activity ***
    lwa = barotropic_field.lwa
    # Check output shape
    assert lwa.shape == (nlat, nlon)
    # Check wave activity zonal mean structure
    assert np.allclose(lwa.mean(axis=-1), answer_key['lwa_zonal_mean'], rtol=1e-05, atol=1e-08)
    # Check longitudinal structure of LWA at peak latitude
    assert np.allclose(lwa[20, :], answer_key['lwa_longitudinal_variation'], rtol=1e-05, atol=1e-08)

