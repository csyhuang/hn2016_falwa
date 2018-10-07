import unittest
import numpy as np
from math import pi
from hn2016_falwa.oopinterface import QGField
import os

class basisTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Set up a hypothetical vorticity fields with uniform longitude and latitude grids to test the functions in basis.py and wrapper.py
        '''

        # Define physical constants
        p0 = 1000.  # Ground pressure level. Unit: hPa
        self.scale_height = 7000.  # Unit: m
        self.kmax = 49
        self.dz = 1000.
        self.cp = 1004.
        self.dry_gas_constant = 287.
        self.planet_radius = 6.378e+6  # Unit: m

        # Define parameters
        nlev, nlat, nlon = 12, 31, 60

        xlon = np.linspace(0, 2. * pi, nlon, endpoint=False)
        ylat = np.linspace(-pi/2., pi/2., nlat, endpoint=True)
        plev = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200,
                         100, 10, 1])

        dir_path = os.path.dirname(__file__)
        self.u_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [nlev, nlat, nlon]
        )
        self.v_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [nlev, nlat, nlon]
        )
        self.t_field = np.reshape(
            np.loadtxt(dir_path + '/test_data/demo_u.txt'),
            [nlev, nlat, nlon]
        )

        # Compute the potential temperature to check whether the interpolation
        # method returns a bounded potential temperature field
        self.theta_field = self.t_field * (plev[:, np.newaxis, np.newaxis]/p0)**(-self.dry_gas_constant / self.cp)

        # Create a QGField object for testing
        self.qgfield = QGField(
            xlon, ylat, plev, self.u_field, self.v_field, self.t_field,
            kmax=self.kmax,
            dz=self.dz,
            scale_height=self.scale_height,
            cp=self.cp,
            dry_gas_constant=self.dry_gas_constant,
            planet_radius=self.planet_radius)

    def test_interpolate_fields(self):
        '''
        Check that the input fields are interpolated onto a grid of correct dimension and the interpolated values are bounded.
        '''

        qgpv, interpolated_u, interpolated_v, interpolated_theta, static_stability = \
            self.qgfield.interpolate_fields()

        kmax, nlat, nlon = \
            self.qgfield.kmax, self.qgfield.nlat, self.qgfield.nlon

        # Check that the dimensions of the interpolated fields are correct
        self.assertEqual(qgpv.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_u.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_v.shape, (kmax, nlat, nlon))
        self.assertEqual(interpolated_theta.shape, (kmax, nlat, nlon))
        self.assertEqual(static_stability.shape, (kmax,))

        # Check that at the interior grid points, the interpolated fields 
        # are bounded
        self.assertTrue(
            (interpolated_u[1:-1, 1:-1, 1:-1].max() <= self.u_field.max()) &
            (interpolated_u[1:-1, 1:-1, 1:-1].max() >= self.u_field.min())
        )
        self.assertTrue(
            (interpolated_u[1:-1, 1:-1, 1:-1].min() <= self.u_field.max()) &
            (interpolated_u[1:-1, 1:-1, 1:-1].min() >= self.u_field.min())
            )
        self.assertTrue(
            (interpolated_v[1:-1, 1:-1, 1:-1].max() <= self.v_field.max()) &
            (interpolated_v[1:-1, 1:-1, 1:-1].max() >= self.v_field.min())
        )
        self.assertTrue(
            (interpolated_v[1:-1, 1:-1, 1:-1].min() <= self.v_field.max()) &
            (interpolated_v[1:-1, 1:-1, 1:-1].min() >= self.v_field.min())
        )
        self.assertTrue(
            (interpolated_theta[1:-1, 1:-1, 1:-1].max() <=
                self.theta_field.max()) &
            (interpolated_theta[1:-1, 1:-1, 1:-1].max() >=
                self.theta_field.min())
        )
        self.assertTrue(
            (interpolated_theta[1:-1, 1:-1, 1:-1].min() <=
             self.theta_field.max()) &
            (interpolated_theta[1:-1, 1:-1, 1:-1].min() >=
                self.theta_field.min())
        )
        self.assertTrue(np.isnan(qgpv).sum() == 0)
        self.assertTrue((qgpv == float('Inf')).sum() == 0)

    def test_compute_reference_states(self):
        '''
        Check that the output reference states are of correct dimension, and the QGPV reference state is non-decreasing.
        '''
        qref_north_hem, uref_north_hem, ptref_north_hem = \
            self.qgfield.compute_reference_states(
                northern_hemisphere_results_only=True
            )
        kmax, nlat, nlon = \
            self.qgfield.kmax, self.qgfield.nlat, self.qgfield.nlon

        # Check dimension of the input field
        self.assertTrue(qref_north_hem.shape == (kmax, nlat//2+1))
        self.assertTrue(uref_north_hem.shape == (kmax, nlat//2+1))
        self.assertTrue(ptref_north_hem.shape == (kmax, nlat//2+1))
        # Check that qref is monotonically increasing (in the interior)
        self.assertTrue(
            (np.diff(qref_north_hem, axis=-1)[:, 1:-1] >= 0.).all()
        )


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(basisTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
