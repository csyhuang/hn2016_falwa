from math import pi

import numpy as np
from scipy.interpolate import interp1d

from hn2016_falwa import utilities
from hn2016_falwa import basis
from hn2016_falwa.constant import *
from interpolate_fields import interpolate_fields
from compute_reference_states import compute_reference_states
from compute_lwa_and_barotropic_fluxes import compute_lwa_and_barotropic_fluxes


class QGField(object):

    """
    Local wave activity and flux analysis in quasi-geostrophic framework
    that can be used to reproduce the results in:
    Nakamura and Huang, Atmospheric Blocking as a Traffic Jam in the Jet Stream, Science (2018).
    Note that topography is assumed flat in this object.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    xlon : numpy.array
           Array of evenly-spaced longitude (in degree) of size nlon.
    ylat : numpy.array
           Array of evenly-spaced latitude (in degree) of size nlat.
    plev : numpy.
           Array of pressure level (in hPa) of size nlev.
    u_field : numpy.ndarray
           Three-dimensional array of zonal wind field (in m/s) of dimension [nlev, nlat, nlon].
    v_field : numpy.ndarray
           Three-dimensional array of meridional wind field (in m/s) of dimension [nlev, nlat, nlon].
    t_field : numpy.ndarray
           Three-dimensional array of temperature field (in K) of dimension [nlev, nlat, nlon].
    kmax : int, optional
           Dimension of uniform pseudoheight grids used for interpolation.
    maxit : int, optional
           Number of iteration by the Successive over-relaxation (SOR) solver to compute the reference states.
    dz : float, optional
           Size of uniform pseudoheight grids (in meters).
    prefactor : float, optional
           Vertical air density summed over height.
    npart : int, optional
           Number of partitions used to compute equivalent latitude.
           If not initialized, it will be set to nlat.
    tol : float, optional
           Tolerance that defines the convergence of solution in SOR solver.
    rjac : float, optional
           Spectral radius of the Jacobi iteration in the SOR solver.
    scale_height : float, optional
           Scale height of the atmosphere in meters. Default = 7000.
    cp : float, optional
           Heat capacity of dry air in J/kg-K. Default = 1004.
    dry_gas_constant : float, optional
           Gas constant for dry air in J/kg-K. Default = 287.
    omega : float, optional
           Rotation rate of the earth in 1/s. Default = 7.29e-5.
    planet_radius : float, optional
           Radius of the planet in meters.
           Default = 6.378e+6 (Earth's radius).


    Examples
    --------
    >>> test_object = QGField(xlon, ylat, plev, u_field, v_field, t_field)

    """

    def __init__(self, xlon, ylat, plev,
                 u_field, v_field, t_field,
                 kmax=49, maxit=100000, dz=1000., prefactor=6500.,
                 npart=None, tol=1.e-5, rjac=0.95,
                 scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS):

        """Create a QGField object.
        This only initialize the attributes of the object. Analysis and
        computation are done by calling various methods.
        """

        # Check if ylat is in ascending order and include the equator
        if np.diff(ylat)[0] < 0:
            raise TypeError("ylat must be in ascending order")
        if (ylat.size % 2 == 0) & (sum(ylat == 0.0) == 0):
            # Even grid
            self.need_latitude_interpolation = True
            self.ylat_no_equator = ylat
            self.ylat = np.linspace(-90., 90., ylat.size+1, endpoint=True)
            self.equator_idx = \
                np.argwhere(self.ylat == 0)[0][0] + 1
            # Fortran indexing starts from 1
        elif sum(ylat == 0) == 1:
            # Odd grid
            self.need_latitude_interpolation = False
            self.ylat_no_equator = None
            self.ylat = ylat
            self.equator_idx = np.argwhere(ylat == 0)[0][0] + 1 # Fortran indexing starts from 1
        else:
            raise TypeError(
                "There are more than 1 grid point with latitude 0."
            )
        self.clat = np.abs(np.cos(np.deg2rad(self.ylat)))

        # === Check if plev is in decending order ===
        if np.diff(plev)[0] > 0:
            raise TypeError("plev must be in decending order")
        else:
            self.plev = plev
        self.xlon = xlon

        # === Check the shape of wind/temperature fields ===
        self.nlev = plev.size
        self.nlat = ylat.size
        self.nlon = xlon.size
        expected_dimension = (self.nlev, self.nlat, self.nlon)
        if u_field.shape != expected_dimension:
            raise TypeError(
                "Incorrect dimension of u_field. Expected dimension: {}"
                .format(expected_dimension)
            )
        if v_field.shape != expected_dimension:
            raise TypeError(
                "Incorrect dimension of v_field. Expected dimension: {}"
                .format(expected_dimension)
            )
        if t_field.shape != expected_dimension:
            raise TypeError(
                "Incorrect dimension of t_field. Expected dimension: {}"
                .format(expected_dimension)
            )

        # === Do Interpolation on latitude grid if needed ===
        if self.need_latitude_interpolation:
            interp_u = interp1d(
                self.ylat_no_equator, u_field, axis=1, fill_value="extrapolate"
            )
            interp_v = interp1d(
                self.ylat_no_equator, v_field, axis=1, fill_value="extrapolate"
            )
            interp_t = interp1d(
                self.ylat_no_equator, t_field, axis=1, fill_value="extrapolate"
            )
            self.u_field = interp_u(self.ylat)
            self.v_field = interp_v(self.ylat)
            self.t_field = interp_t(self.ylat)
        else:
            self.u_field = u_field
            self.v_field = v_field
            self.t_field = t_field

        # === To be computed ===
        self.dphi = np.deg2rad(180./(self.nlat-1))
        self.dlambda = np.deg2rad(self.xlon[1] - self.xlon[0])

        if npart is None:
            self.npart = self.nlat
        else:
            self.npart = npart
        self.height = np.array([i * dz for i in range(kmax)])

        # === Parameters ===
        self.kmax = kmax
        self.maxit = maxit
        self.dz = dz
        self.prefactor = prefactor
        self.tol = tol
        self.rjac = rjac

        # === Constants ===
        self.scale_height = scale_height
        self.cp = cp
        self.dry_gas_constant = dry_gas_constant
        self.omega = omega
        self.planet_radius = planet_radius

        # === Variables that will be computed in methods ===
        self._qgpv_temp = None
        self._interpolated_u_temp = None
        self._interpolated_v_temp = None
        self._interpolated_theta_temp = None
        self._static_stability = None

        # Computation from computer_reference_states
        self._qref_stemp = None
        self._uref_stemp = None
        self._ptref_stemp = None
        self._qref_ntemp = None
        self._uref_ntemp = None
        self._ptref_ntemp = None
        self._qref = None
        self._uref = None
        self._ptref = None
        self._qgpv = None
        self._interpolated_u = None
        self._interpolated_v = None
        self._interpolated_theta = None
        self.northern_hemisphere_results_only = False

        # Computation from compute_lwa_and_barotropic_fluxes
        self._adv_flux_f1 = None
        self._adv_flux_f2 = None
        self._adv_flux_f3 = None
        self._convergence_zonal_advective_flux = None
        self._meridional_heat_flux = None
        self._lwa_baro = None
        self._u_baro = None
        self._lwa = None
        self._divergence_eddy_momentum_flux = None

    def _interp_back(
        self,
        field,
        interp_from,
        interp_to,
        which_axis=1
    ):
        """
        Internal function to interpolate the results from odd grid to even grid.
        If the initial input to the QGField object is an odd grid, error will be raised.
        """
        # print('For debugging. field.shape = {}'.format(field.shape))
        # print('For debugging. interp_from.shape = {}'.format(interp_from.shape))
        # print('For debugging. interp_to.shape = {}'.format(interp_to.shape))

        if self.ylat_no_equator is None:
            raise TypeError("No need for such interpolation.")
        else:
            return interp1d(
                interp_from, field, axis=which_axis, bounds_error=False,
                fill_value='extrapolate'
            )(interp_to)

    def interpolate_fields(self):
        """
        Interpolate zonal wind, maridional wind, and potential temperature field onto the uniform pseudoheight grids, and compute QGPV on the same grids.

        Returns
        -------

        qgpv : numpy.ndarray
            Three-dimensional array of quasi-geostrophic potential vorticity (QGPV) with dimension = [kmax, nlat, nlon]

        interpolated_u : numpy.ndarray
            Three-dimensional array of interpolated zonal wind with dimension = [kmax, nlat, nlon]

        interpolated_v : numpy.ndarray
            Three-dimensional array of interpolated meridional wind with dimension = [kmax, nlat, nlon]

        interpolated_theta : numpy.ndarray
            Three-dimensional array of interpolated potential temperature with dimension = [kmax, nlat, nlon]

        static_stability : numpy.array
            One-dimension array of interpolated static stability with dimension = kmax


        Examples
        --------

        >>> qgpv, interpolated_u, interpolated_v,
            interpolated_theta, static_stability
            = test_object.interpolate_fields()

        """

        if self._qref_ntemp is None:

            # === Interpolate fields and obtain qgpv ===
            self._qgpv_temp, \
                self._interpolated_u_temp, \
                self._interpolated_v_temp, \
                self._interpolated_theta_temp, \
                self._static_stability = \
                interpolate_fields(
                    np.swapaxes(self.u_field, 0, 2),
                    np.swapaxes(self.v_field, 0, 2),
                    np.swapaxes(self.t_field, 0, 2),
                    self.plev,
                    self.height,
                    self.planet_radius,
                    self.omega,
                    self.dz,
                    self.scale_height,
                    self.dry_gas_constant,
                    self.cp
                )

            self._qgpv = np.swapaxes(self._qgpv_temp, 0, 2)
            self._interpolated_u = np.swapaxes(self._interpolated_u_temp, 0, 2)
            self._interpolated_v = np.swapaxes(self._interpolated_v_temp, 0, 2)
            self._interpolated_theta = np.swapaxes(
                self._interpolated_theta_temp, 0, 2
            )

        return self.qgpv, self.interpolated_u, self.interpolated_v, \
               self.interpolated_theta, self.static_stability

    def _return_interp_variables(self, variable, interp_axis, northern_hemisphere_results_only=True):
        if self.need_latitude_interpolation:
            if northern_hemisphere_results_only:
                return self._interp_back(
                    variable, self.ylat[-(self.nlat//2+1):],
                    self.ylat_no_equator[-(self.nlat//2):],
                    which_axis=interp_axis)
            else:
                return self._interp_back(variable, self.ylat, self.ylat_no_equator, which_axis=interp_axis)
        else:
            return variable

    @property
    def qgpv(self):
        if self._qgpv is None:
            raise ValueError('QGPV field is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._qgpv, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_u(self):
        if self._interpolated_u is None:
            raise ValueError('interpolated_u is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_u, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_v(self):
        if self._interpolated_v is None:
            raise ValueError('interpolated_v is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_v, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_theta(self):
        if self._interpolated_theta is None:
            raise ValueError('interpolated_theta is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_theta, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def static_stability(self):
        """
        Retrieve the interpolated static stability.
        """
        return self._static_stability

    def _compute_reference_state_wrapper(self, qgpv, u, theta):
        return compute_reference_states(
            qgpv,
            u,
            theta,
            self._static_stability,
            self.equator_idx,
            self.npart,
            self.maxit,
            self.planet_radius,
            self.omega,
            self.dz,
            self.tol,
            self.scale_height,
            self.dry_gas_constant,
            self.cp,
            self.rjac,
        )

    def compute_reference_states(self, northern_hemisphere_results_only=True):

        """
        Compute the local wave activity and reference states of QGPV, zonal wind and potential temperature using a more stable inversion algorithm applied in Nakamura and Huang (2018, Science). The equation to be invert is equation (22) in supplementary materials of Huang and Nakamura (2017, GRL). In this version, only values in the Northern Hemisphere is computed.


        Parameters
        ----------
        northern_hemisphere_results_only : bool
               If true, arrays of size [kmax, nlat//2+1] will be returned. Otherwise, arrays of size [kmax, nlat] will be returned. This parameter is present since the current version (v0.3.1) of the package only return analysis in the northern hemisphere. Default: True.

        Returns
        -------

        qref : numpy.ndarray
            Two-dimensional array of reference state of quasi-geostrophic potential vorticity (QGPV) with dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False, or dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True

        uref : numpy.ndarray
            Two-dimensional array of reference state of zonal wind (Uref) with dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False, or dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True

        ptref : numpy.ndarray
            Two-dimensional array of reference state of potential temperature (Theta_ref) with dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False, or dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True

        Examples
        --------

        >>> qref, uref, ptref = test_object.compute_reference_states()

        """
        self.northern_hemisphere_results_only = \
            northern_hemisphere_results_only

        if self._qgpv_temp is None:
            self.interpolate_fields()

        # === Compute reference states in Northern Hemisphere ===
        self._qref_ntemp, self._uref_ntemp, self._ptref_ntemp = self._compute_reference_state_wrapper(
            qgpv=self._qgpv_temp, u=self._interpolated_u_temp, theta=self._interpolated_theta_temp)

        # === Compute reference states in Southern Hemisphere ===
        self._qref_stemp, self._uref_stemp, self._ptref_stemp = self._compute_reference_state_wrapper(
            qgpv=self._qgpv_temp[:, ::-1, :],
            u=self._interpolated_u_temp[:, ::-1, :],
            theta=self._interpolated_theta_temp[:, ::-1, :])

        qref_ntemp_right_unit = \
            self._qref_ntemp * 2 * self.omega * np.sin(np.deg2rad(self.ylat[(self.equator_idx - 1):, np.newaxis]))
        qref_stemp_right_unit = \
            self._qref_stemp[::-1, :] * 2 * self.omega * np.sin(
                np.deg2rad(self.ylat[:self.equator_idx, np.newaxis]))

        if self.northern_hemisphere_results_only:
            self._qref = np.swapaxes(qref_ntemp_right_unit, 0, 1)
            self._uref = np.swapaxes(self._uref_ntemp, 0, 1)
            self._ptref = np.swapaxes(self._ptref_ntemp, 0, 1)
        else:
            self._qref = \
                np.hstack((np.swapaxes(qref_stemp_right_unit[:, :], 0, 1),
                           np.swapaxes(qref_ntemp_right_unit[1:, :], 0, 1)))
            self._uref = \
                np.hstack((np.swapaxes(self._uref_stemp[::-1, :], 0, 1),
                           np.swapaxes(self._uref_ntemp[1:, :], 0, 1)))
            self._ptref = \
                np.hstack((np.swapaxes(self._ptref_stemp[::-1, :], 0, 1),
                           np.swapaxes(self._ptref_ntemp[1:, :], 0, 1)))

        return self.qref, self.uref, self.ptref

    @property
    def qref(self):
        """
        Return reference state of QGPV (Qref).
        """
        if self._qref is None:
            raise ValueError('qref is not computed yet.')
        return self._return_interp_variables(variable=self._qref, interp_axis=1, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def uref(self):
        """
        Return reference state of zonal wind (Uref).
        """
        if self._uref is None:
            raise ValueError('uref field is not computed yet.')
        return self._return_interp_variables(variable=self._uref, interp_axis=1, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def ptref(self):
        """
        Return reference state of potential temperature (\Theta_ref).
        """
        if self._ptref is None:
            raise ValueError('ptref field is not computed yet.')
        return self._return_interp_variables(variable=self._ptref, interp_axis=1, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    def _compute_lwa_and_barotropic_fluxes_wrapper(self, qgpv, u, v, theta):
        return compute_lwa_and_barotropic_fluxes(
            qgpv,
            u,
            v,
            theta,
            self._qref_ntemp,
            self._uref_ntemp,
            self._ptref_ntemp,
            self.planet_radius,
            self.omega,
            self.dz,
            self.scale_height,
            self.dry_gas_constant,
            self.cp,
            self.prefactor)

    def compute_lwa_and_barotropic_fluxes(
        self, northern_hemisphere_results_only=True
    ):

        """
        Compute barotropic components of local wave activity and flux terms in eqs.(2) and (3) in Nakamura and Huang (Science, 2018).

        Parameters
        ----------
        northern_hemisphere_results_only : bool
               If true, arrays of size [kmax, nlat//2] will be returned. Otherwise, arrays of size [kmax, nlat] will be returned. This parameter is present since the current version (v0.3.1) of the package only return analysis in the northern hemisphere. Default: True.


        Returns
        -------

        adv_flux_f1 : numpy.ndarray
            Two-dimensional array of the second-order eddy term in zonal advective flux,
            i.e. F1 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        adv_flux_f2 : numpy.ndarray
            Two-dimensional array of the third-order eddy term in zonal advective flux,
            i.e. F2 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        adv_flux_f3 : numpy.ndarray
            Two-dimensional array of the remaining term in zonal advective flux,
            i.e. F3 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        convergence_zonal_advective_flux : numpy.ndarray
            Two-dimensional array of the convergence of zonal advective flux,
            i.e. -div(F1+F2+F3) in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        divergence_eddy_momentum_flux : numpy.ndarray
            Two-dimensional array of the divergence of eddy momentum flux,
            i.e. (II) in equation 2 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        meridional_heat_flux : numpy.ndarray
            Two-dimensional array of the low-level meridional heat flux,
            i.e. (III) in equation 2 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        lwa_baro : np.ndarray
            Two-dimensional array of barotropic local wave activity (with cosine weighting). Dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        u_baro     : np.ndarray
            Two-dimensional array of barotropic zonal wind (without cosine weighting). Dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        lwa : np.ndarray
            Three-dimensional array of barotropic local wave activity Dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False.


        Examples
        --------

        >>> adv_flux_f1, adv_flux_f2, adv_flux_f3, convergence_zonal_advective_flux,
            divergence_eddy_momentum_flux, meridional_heat_flux,
            lwa_baro, u_baro, lwa = test_object.compute_lwa_and_barotropic_fluxes()

        """

        self.northern_hemisphere_results_only = northern_hemisphere_results_only

        if self._qgpv_temp is None:
            self.interpolate_fields()

        if self._uref_ntemp is None:
            self.compute_reference_states()

        # === Compute barotropic flux terms (NHem) ===
        lwa_nhem, astarbaro_nhem, ua1baro_nhem, ubaro_nhem, ua2baro_nhem,\
            ep1baro_nhem, ep2baro_nhem, ep3baro_nhem, ep4_nhem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                self._qgpv_temp,
                self._interpolated_u_temp,
                self._interpolated_v_temp,
                self._interpolated_theta_temp)

        # === Compute barotropic flux terms (SHem) ===
        lwa_shem, astarbaro_shem, ua1baro_shem, ubaro_shem, ua2baro_shem,\
            ep1baro_shem, ep2baro_shem, ep3baro_shem, ep4_shem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                -self._qgpv_temp[:, ::-1, :],
                self._interpolated_u_temp[:, ::-1, :],
                self._interpolated_v_temp[:, ::-1, :],
                self._interpolated_theta_temp[:, ::-1, :])

        # *** Northern Hemisphere ***
        # Compute divergence of the meridional eddy momentum flux
        meri_flux_nhem_temp = np.zeros_like(ep2baro_nhem)
        meri_flux_nhem_temp[:, 1:-1] = (ep2baro_nhem[:, 1:-1] - ep3baro_nhem[:, 1:-1]) / \
            (2 * self.planet_radius * self.dphi *
             np.cos(np.deg2rad(self.ylat[-self.equator_idx + 1:-1])))
        # Compute convergence of the zonal LWA flux
        zonal_adv_flux_nhem_sum = np.swapaxes((ua1baro_nhem + ua2baro_nhem + ep1baro_nhem), 0, 1)
        convergence_zonal_advective_flux_nhem = \
            utilities.zonal_convergence(
                zonal_adv_flux_nhem_sum,
                np.cos(np.deg2rad(self.ylat[-self.equator_idx:])),
                self.dlambda,
                planet_radius=self.planet_radius
            )

        # *** Southern Hemisphere ***
        # Compute divergence of the meridional eddy momentum flux
        meri_flux_shem_temp = np.zeros_like(ep2baro_shem)
        meri_flux_shem_temp[:, 1:-1] = (ep2baro_shem[:, 1:-1] - ep3baro_shem[:, 1:-1]) / \
            (2 * self.planet_radius * self.dphi *
             np.cos(np.deg2rad(self.ylat[-self.equator_idx + 1:-1])))
        # Compute convergence of the zonal LWA flux
        zonal_adv_flux_shem_sum = np.swapaxes((ua1baro_shem + ua2baro_shem + ep1baro_shem), 0, 1)  # axes swapped
        convergence_zonal_advective_flux_shem = \
            utilities.zonal_convergence(
                zonal_adv_flux_shem_sum,
                np.cos(np.deg2rad(self.ylat[-self.equator_idx:])),
                self.dlambda,
                planet_radius=self.planet_radius
            )

        if northern_hemisphere_results_only:
            self._adv_flux_f1 = np.swapaxes(ua1baro_nhem, 0, 1)
            self._adv_flux_f2 = np.swapaxes(ua2baro_nhem, 0, 1)
            self._adv_flux_f3 = np.swapaxes(ep1baro_nhem, 0, 1)
            self._convergence_zonal_advective_flux = convergence_zonal_advective_flux_nhem
            self._meridional_heat_flux = np.swapaxes(ep4_nhem, 0, 1)
            self._lwa_baro = np.swapaxes(astarbaro_nhem, 0, 1)
            self._u_baro = np.swapaxes(ubaro_nhem, 0, 1)
            self._lwa = np.swapaxes(lwa_nhem, 0, 2)
            self._divergence_eddy_momentum_flux = \
                np.swapaxes(meri_flux_nhem_temp, 0, 1)
        else:
            # Flip component in southern hemisphere
            self._adv_flux_f1 = np.vstack((np.swapaxes(ua1baro_shem[:, ::-1], 0, 1),
                                           np.swapaxes(ua1baro_nhem[:, 1:], 0, 1)))

            self._adv_flux_f2 = np.vstack((np.swapaxes(ua2baro_shem[:, ::-1], 0, 1),
                                           np.swapaxes(ua2baro_nhem[:, 1:], 0, 1)))

            self._adv_flux_f3 = np.vstack((np.swapaxes(ep1baro_shem[:, ::-1], 0, 1),
                                           np.swapaxes(ep1baro_nhem[:, 1:], 0, 1)))

            # Axes already swapped for convergence zonal advective flux
            self._convergence_zonal_advective_flux = np.vstack((convergence_zonal_advective_flux_shem[::-1, :],
                                                                convergence_zonal_advective_flux_nhem[1:, :]))

            self._meridional_heat_flux = \
                np.vstack((np.swapaxes(ep4_shem[:, ::-1], 0, 1),
                           np.swapaxes(ep4_nhem[:, 1:], 0, 1)))

            self._lwa_baro = \
                np.vstack((np.swapaxes(astarbaro_shem[:, ::-1], 0, 1),
                           np.swapaxes(astarbaro_nhem[:, 1:], 0, 1)))

            self._u_baro = np.vstack((np.swapaxes(ubaro_shem[:, ::-1], 0, 1),
                                      np.swapaxes(ubaro_nhem[:, 1:], 0, 1)))

            self._lwa = np.concatenate((np.swapaxes(lwa_shem[:, ::-1], 0, 2),
                                        np.swapaxes(lwa_nhem[:, 1:], 0, 2)), axis=1)

            self._divergence_eddy_momentum_flux = np.vstack((np.swapaxes(meri_flux_shem_temp[:, ::-1], 0, 1),
                                                             np.swapaxes(meri_flux_nhem_temp[:, 1:], 0, 1)))

        return self.adv_flux_f1, self.adv_flux_f2, self.adv_flux_f3, self.convergence_zonal_advective_flux,\
               self.divergence_eddy_momentum_flux, self.meridional_heat_flux, self.lwa_baro, self.u_baro, self.lwa

    @property
    def adv_flux_f1(self):
        if self._adv_flux_f1 is None:
            raise ValueError('adv_flux_f1 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f1, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def adv_flux_f2(self):
        if self._adv_flux_f2 is None:
            raise ValueError('adv_flux_f2 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f2, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def adv_flux_f3(self):
        if self._adv_flux_f3 is None:
            raise ValueError('adv_flux_f3 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f3, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def convergence_zonal_advective_flux(self):
        if self._convergence_zonal_advective_flux is None:
            raise ValueError('convergence_zonal_advective_flux is not computed yet.')
        return self._return_interp_variables(variable=self._convergence_zonal_advective_flux, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def divergence_eddy_momentum_flux(self):
        if self._divergence_eddy_momentum_flux is None:
            raise ValueError('divergence_eddy_momentum_flux is not computed yet.')
        return self._return_interp_variables(variable=self._divergence_eddy_momentum_flux, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def meridional_heat_flux(self):
        if self._meridional_heat_flux is None:
            raise ValueError('meridional_heat_flux is not computed yet.')
        return self._return_interp_variables(variable=self._meridional_heat_flux, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def lwa_baro(self):
        if self._lwa_baro is None:
            raise ValueError('lwa_baro is not computed yet.')
        return self._return_interp_variables(variable=self._lwa_baro, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def u_baro(self):
        if self._u_baro is None:
            raise ValueError('u_baro is not computed yet.')
        return self._return_interp_variables(variable=self._u_baro, interp_axis=0, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    @property
    def lwa(self):
        if self._lwa is None:
            raise ValueError('lwa is not computed yet.')
        return self._return_interp_variables(variable=self._lwa, interp_axis=1, northern_hemisphere_results_only=self.northern_hemisphere_results_only)

    def get_latitude_dim(self):
        """
        Return the latitude dimension of the input data.
        """
        if self.need_latitude_interpolation:
            return self.ylat_no_equator.size
        else:
            return self.nlat


def curl_2d(ufield, vfield, clat, dlambda, dphi, planet_radius=6.378e+6):
    """
    Assuming regular latitude and longitude [in degree] grid, compute the curl
    of velocity on a pressure level in spherical coordinates.
    """

    ans = np.zeros_like(ufield)
    ans[1:-1, 1:-1] = (vfield[1:-1, 2:] - vfield[1:-1, :-2])/(2.*dlambda) - \
                      (ufield[2:, 1:-1] * clat[2:, np.newaxis] -
                       ufield[:-2, 1:-1] * clat[:-2, np.newaxis])/(2.*dphi)
    ans[0, :] = 0.0
    ans[-1, :] = 0.0
    ans[1:-1, 0] = ((vfield[1:-1, 1] - vfield[1:-1, -1]) / (2. * dlambda) -
                    (ufield[2:, 0] * clat[2:] -
                     ufield[:-2, 0] * clat[:-2]) / (2. * dphi))
    ans[1:-1, -1] = ((vfield[1:-1, 0] - vfield[1:-1, -2]) / (2. * dlambda) -
                     (ufield[2:, -1] * clat[2:] -
                      ufield[:-2, -1] * clat[:-2]) / (2. * dphi))
    ans[1:-1, :] = ans[1:-1, :] / planet_radius / clat[1:-1, np.newaxis]
    return ans


class BarotropicField(object):

    """
    An object that deals with barotropic (2D) wind and/or PV fields

    Parameters
    ----------
        xlon : np.array
            Longitude array in degree with dimension = nlon.

        ylat : np.array
            Latitude array in degree with dimension = nlat.

        area : np.ndarray
            Differential area at each lon-lat grid points with dimension (nlat,nlon). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface. Dimension = [nlat, nlon]

        dphi : np.array
            Differential length element along the lat grid with dimension = nlat.

        pv_field : np.ndarray
            Absolute vorticity field with dimension [nlat x nlon]. If 'pv_field=None': pv_field is expected to be computed with u,v,t field.


    Example
    ---------
    >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)

    """

    def __init__(self, xlon, ylat, pv_field, area=None, dphi=None,
                 n_partitions=None, planet_radius=6.378e+6):

        """
        Create a BarotropicField object.

        Parameters
        ----------
            xlon : np.array
                Longitude array in degree with dimension = nlon.

            ylat : np.array
                Latitude array in degree with dimension = nlat.

            area : np.ndarray
                Differential area at each lon-lat grid points with dimension (nlat,nlon). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface. Dimension = [nlat, nlon]

            dphi : np.array
                Differential length element along the lat grid with dimension = nlat.

            pv_field : np.ndarray
                Absolute vorticity field with dimension = [nlat, nlon].
                If none, pv_field is expected to be computed with u,v,t field.

        """

        self.xlon = xlon
        self.ylat = ylat
        self.clat = np.abs(np.cos(np.deg2rad(ylat)))
        self.nlon = xlon.size
        self.nlat = ylat.size
        self.planet_radius = planet_radius
        if dphi is None:
            self.dphi = pi/(self.nlat-1) * np.ones((self.nlat))
        else:
            self.dphi = dphi

        if area is None:
            self.area = 2. * pi * planet_radius ** 2 * \
                        (np.cos(ylat[:, np.newaxis] * pi/180.) * self.dphi[:, np.newaxis])\
                        / float(self.nlon)*np.ones((self.nlat, self.nlon))
        else:
            self.area = area

        self.pv_field = pv_field

        if n_partitions is None:
            self.n_partitions = self.nlat
        else:
            self.n_partitions = n_partitions

        # Quantities that are computed with the methods below
        self.eqvlat = None
        self.lwa = None

    def _compute_eqvlat(self):
        """
        Internal function. Compute equivalent latitude if it has not been computed yet.
        """
        self.eqvlat, _ = basis.eqvlat(
            self.ylat, self.pv_field, self.area, self.n_partitions,
            planet_radius=self.planet_radius
        )
        return self.eqvlat

    def _compute_lwa(self):
        """
        Internal function. Compute equivalent latitude if it has not been computed yet.
        """
        if self.eqvlat is None:
            self.eqvlat = self.equivalent_latitudes()

        if self.lwa is None:
            self.lwa, dummy = basis.lwa(
                self.nlon, self.nlat, self.pv_field, self.eqvlat,
                self.planet_radius * self.clat * self.dphi
            )
        return self.lwa

    @property
    def equivalent_latitudes(self):
        """
        Return the computd quivalent latitude with the *pv_field* stored in the object.

        Return
        ----------
        An numpy array with dimension (nlat) of equivalent latitude array.

        Example
        ----------
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> eqv_lat = barofield1.equivalent_latitudes

        """
        if self.eqvlat is None:
            return self._compute_eqvlat()
        return self.eqvlat

    @property
    def lwa(self):

        """
        Compute the finite-amplitude local wave activity based on the *equivalent_latitudes* and the *pv_field* stored in the object.

        Return
        ----------
        An 2-D numpy array with dimension [nlat,nlon] of local wave activity values.

        Example
        ----------
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> lwa = barofield1.lwa

        """
        if self.lwa is None:
            return self._compute_lwa()
        return self.lwa
