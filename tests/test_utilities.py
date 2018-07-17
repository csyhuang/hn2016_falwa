import unittest
import numpy as np
import numpy.testing as npt
from math import pi
import hn2016_falwa.utilities as utilities

class basisTestCase(unittest.TestCase):

    def test_zonal_convergence(self):
    
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
        ylat = np.deg2rad(np.linspace(0, 90, nlat, endpoint=True))   # in radian
        xlon = np.deg2rad(np.linspace(0, 360, nlon, endpoint=False)) # in radian
        clat = np.cos(ylat)
        
        # Define the field to compute derivative
        field = clat[:, np.newaxis] * np.sin(xlon)
        
        expected_answer = - np.ones((nlat, nlon)) * np.cos(xlon) / planet_radius

        computed = utilities.zonal_convergence(field, clat, xlon[1] - xlon[0],
                                               planet_radius=planet_radius,
                                               tol=tol)

        # Discard boundary in the latitudinal domain
        self.assertTrue(np.abs(computed[1:-1, :] - expected_answer[1:-1, :]).max() < tol)


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(basisTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
