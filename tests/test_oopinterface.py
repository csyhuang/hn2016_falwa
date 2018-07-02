import unittest
import numpy as np
from math import pi, exp, sin, cos
from hn2016_falwa.oopinterface import QGField

class basisTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Set up a hypothetical vorticity fields with uniform longitude and latitude grids
        to test the functions in basis.py and wrapper.py
        '''
        
        # Define physical constants
        lapse_rate = 0.0098 # Unit: K/m
        p0 = 1000. # Ground pressure level. Unit: hPa
        self.scale_height = 7000. # Unit: m
        self.kmax = 49
        self.dz = 1000.
        self.cp = 1004.
        self.dry_gas_constant = 287.
        self.planet_radius = 6.378e+6 # Unit: m        

        # Define parameters
        nlat, nlon, nlev = 241, 480, 30
        wavenum_k, wavenum_l = 6., 1.
        
        xlon = np.linspace(0, 2. * pi, nlon, endpoint=False)
        ylat = np.linspace(-pi/2., pi/2., nlat, endpoint=True)
        zlev = np.linspace(0, self.dz * (self.kmax-1), nlev, endpoint=True)
        plev = p0 * np.exp(-zlev/self.scale_height)
        
        u_field, v_field, t_field = \
            np.zeros((nlev, nlat, nlon)), np.zeros((nlev, nlat, nlon)), \
            np.zeros((nlev, nlat, nlon))

        # Create hypothetical wind and temperature fields to test the object methods
        psi = np.cos(3. * pi * zlev / (self.kmax - 1) * self.dz)
        streamfunction = psi[:, np.newaxis, np.newaxis] \
                         * np.exp(1j*(wavenum_k * xlon[np.newaxis, np.newaxis, :] 
                                  + wavenum_l * ylat[np.newaxis, :, np.newaxis] + pi/6.))
        self.u_field = np.real(1j * wavenum_l / self.planet_radius * streamfunction)
        self.v_field = np.real(-1j * wavenum_k / (self.planet_radius 
                                                  * np.cos(ylat[np.newaxis, :, np.newaxis]))
                                                  * streamfunction)
        self.t_field = np.exp(-zlev[:, np.newaxis, np.newaxis] * lapse_rate) \
                       * (pi/2. - np.abs(ylat[np.newaxis, :, np.newaxis])) * np.real(streamfunction)
                       
        # Compute the potential temperature to check whether the interpolation method 
        # returns a bounded potential temperature field
        self.theta_field = self.t_field * (plev[:, np.newaxis, np.newaxis]/p0)**(-self.dry_gas_constant / self.cp)

        # Create a QGField object for testing
        self.qgfield = QGField(xlon, ylat, plev, self.u_field, self.v_field, self.t_field,
                               kmax=self.kmax,
                               dz=self.dz,
                               scale_height=self.scale_height,
                               cp=self.cp,
                               dry_gas_constant=self.dry_gas_constant,
                               planet_radius=self.planet_radius)
        

    def test_interpolate_fields(self):
        '''
        Check that the input fields are interpolated onto a grid of correct dimension and 
        the interpolated values are bounded.
        '''
    
        qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
            self.qgfield.interpolate_fields()
            
        kmax, nlat, nlon = self.qgfield.kmax, self.qgfield.nlat, self.qgfield.nlon
        
        # Check that the dimensions of the interpolated fields are correct
        self.assertEqual(qgpv.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_u.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_v.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_theta.shape, (kmax, nlat, nlon))
        self.assertEqual(static_stability.shape, (kmax,))
        
        # Check that at the interior grid points, the interpolated fields are bounded
        self.assertTrue((interpolated_u[1:-1, 1:-1, 1:-1].max() <= self.u_field.max()) &
                        (interpolated_u[1:-1, 1:-1, 1:-1].max() >= self.u_field.min())) 
        self.assertTrue((interpolated_u[1:-1, 1:-1, 1:-1].min() <= self.u_field.max()) &
                        (interpolated_u[1:-1, 1:-1, 1:-1].min() >= self.u_field.min())) 
        self.assertTrue((interpolated_v[1:-1, 1:-1, 1:-1].max() <= self.v_field.max()) &
                         (interpolated_v[1:-1, 1:-1, 1:-1].max() >= self.v_field.min()))
        self.assertTrue((interpolated_v[1:-1, 1:-1, 1:-1].min() <= self.v_field.max()) &
                        (interpolated_v[1:-1, 1:-1, 1:-1].min() >= self.v_field.min()))
        self.assertTrue((interpolated_theta[1:-1, 1:-1, 1:-1].max() <= self.theta_field.max()) &
                         (interpolated_theta[1:-1, 1:-1, 1:-1].max() >= self.theta_field.min()))
        self.assertTrue((interpolated_theta[1:-1, 1:-1, 1:-1].min() <= self.theta_field.max()) &
                        (interpolated_theta[1:-1, 1:-1, 1:-1].min() >= self.theta_field.min()))




if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(basisTestCase)
    unittest.TextTestRunner(verbosity=4).run(suite)
