import numpy as np
from math import pi
import hn2016_falwa.utilities as utilities


def test_zonal_convergence():

    '''
    To assert that the zonal_convergence version in utilities.py is computing the zonal
    convergence of a known function right. Here, I will use the example
    f(lat, lon) = cos(lat) sin(lon)
    zonal_convergence(lat, lon) =
        -1/(planet_radius * cos(lat)) * partial d(f(lat, lon))/partial d(lon)
    zonal_convergence of f is given by
        - cos(lon)/planet_radius
    '''

    planet_radius = 1.
    nlat = 1001
    nlon = 1000
    tol = 1.e-5
    ylat = np.linspace(0, pi/2., nlat, endpoint=True)
    xlon = np.linspace(0, 2.*pi, nlon, endpoint=False)
    clat = np.cos(ylat)

    # Define the field to compute derivative
    field = clat[:, np.newaxis] * np.sin(xlon)

    expected_answer = \
        - np.ones((nlat, nlon)) * np.cos(xlon) / planet_radius

    computed = utilities.zonal_convergence(
        field, clat, xlon[1] - xlon[0],
        planet_radius=planet_radius,
        tol=tol
    )

    # Discard boundary in the latitudinal domain
    assert np.abs(computed[1:-1, :] - expected_answer[1:-1, :]).max() < tol
