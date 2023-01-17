"""
------------------------------------------
File name: oopinterface.py
Author: Clare Huang
"""
from typing import Optional
import math
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgetrf, dgetri

from hn2016_falwa import utilities
from hn2016_falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA

from hn2016_falwa import interpolate_fields, interpolate_fields_direct_inv, compute_qref_and_fawa_first,\
    matrix_b4_inversion, matrix_after_inversion, upward_sweep, compute_flux_dirinv, compute_reference_states,\
    compute_lwa_and_barotropic_fluxes
from collections import namedtuple


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
           If it is a masked array, the value ylat.data will be used.
    plev : numpy.
           Array of pressure level (in hPa) of size nlev.
    u_field : numpy.ndarray
           Three-dimensional array of zonal wind field (in m/s) of dimension [nlev, nlat, nlon].
           If it is a masked array, the value u_field.data will be used.
    v_field : numpy.ndarray
           Three-dimensional array of meridional wind field (in m/s) of dimension [nlev, nlat, nlon].
           If it is a masked array, the value v_field.data will be used.
    t_field : numpy.ndarray
           Three-dimensional array of temperature field (in K) of dimension [nlev, nlat, nlon].
           If it is a masked array, the value t_field.data will be used.
    kmax : int, optional
           Dimension of uniform pseudoheight grids used for interpolation.
    maxit : int, optional
           Number of iteration by the Successive over-relaxation (SOR) solver to compute the reference states.
    dz : float, optional
           Size of uniform pseudoheight grids (in meters).
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
    eq_boundary_index: int, optional
           The improved inversion algorithm of reference states allow modification of equatorward boundary
           to be the absolute vorticity. This parameter specify the location of grid point (from equator)
           which will be used as boundary. Default = 5.

    Examples
    --------
    >>> test_object = QGField(xlon, ylat, plev, u_field, v_field, t_field)

    """

    def __init__(self, xlon, ylat, plev, u_field, v_field, t_field, kmax=49, maxit=100000, dz=1000., npart=None,
                 tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, prefactor=None, eq_boundary_index=5):

        """
        Create a QGField object.
        This only initialize the attributes of the object. Analysis and
        computation are done by calling various methods.
        """

        # === Check whether the input field is masked array. If so, turn them to normal array ===
        if np.ma.is_masked(ylat) or isinstance(ylat, np.ma.core.MaskedArray):
            warnings.warn(
                'ylat is a masked array of dimension {dim} with {num} masked elements and fill value {fill}. '
                .format(dim=ylat.shape, num=ylat.mask.sum(), fill=ylat.fill_value))
            ylat = ylat.data

        if np.ma.is_masked(u_field) or isinstance(u_field, np.ma.core.MaskedArray):
            warnings.warn(
                'u_field is a masked array of dimension {dim} with {num} masked elements and fill value {fill}. '
                .format(dim=u_field.shape, num=u_field.mask.sum(), fill=u_field.fill_value))
            u_field = u_field.data

        if np.ma.is_masked(v_field) or isinstance(v_field, np.ma.core.MaskedArray):
            warnings.warn(
                'v_field is a masked array of dimension {dim} with {num} masked elements and fill value {fill}. '
                .format(dim=v_field.shape, num=v_field.mask.sum(), fill=v_field.fill_value))
            v_field = v_field.data

        if np.ma.is_masked(t_field) or isinstance(t_field, np.ma.core.MaskedArray):
            warnings.warn(
                't_field is a masked array of dimension {dim} with {num} masked elements and fill value {fill}. '
                .format(dim=t_field.shape, num=t_field.mask.sum(), fill=t_field.fill_value))
            t_field = t_field.data

        # === Check if ylat is in ascending order and include the equator ===
        self._check_and_flip_ylat(ylat)

        # === Check the validity of plev ===
        self._check_valid_plev(plev, scale_height, kmax, dz)

        # === Initialize longitude grid ===
        self.xlon = xlon

        # === Check the shape of wind/temperature fields ===
        self.nlev = plev.size
        self.nlat = ylat.size
        self.nlon = xlon.size
        expected_dimension = (self.nlev, self.nlat, self.nlon)
        self._check_dimension_of_fields(field=u_field, field_name='u_field', expected_dim=expected_dimension)
        self._check_dimension_of_fields(field=v_field, field_name='v_field', expected_dim=expected_dimension)
        self._check_dimension_of_fields(field=t_field, field_name='t_field', expected_dim=expected_dimension)

        # === Do Interpolation on latitude grid if needed ===
        if self.need_latitude_interpolation:
            interp_u = interp1d(self.ylat_no_equator, u_field, axis=1, fill_value="extrapolate")
            interp_v = interp1d(self.ylat_no_equator, v_field, axis=1, fill_value="extrapolate")
            interp_t = interp1d(self.ylat_no_equator, t_field, axis=1, fill_value="extrapolate")
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
        self.tol = tol
        self.rjac = rjac
        self.eq_boundary_index = eq_boundary_index

        # === Constants ===
        self.scale_height = scale_height
        self.cp = cp
        self.dry_gas_constant = dry_gas_constant
        self.omega = omega
        self.planet_radius = planet_radius
        self._compute_prefactor()  # Compute normalization prefactor

        # Modification on Oct 19, 2021 - deprecation of prefactor (it should be computed from kmax and dz)
        if prefactor is not None:
            warnings.warn(
                "The optional input prefactor will be deprecated since it can be determined directly from " +
                "kmax and dz. Given your input kmax = {kmax} and dz = {dz}, ".format(kmax=self.kmax, dz=self.dz) +
                "the computed normalization prefactor is {prefactor}. ".format(prefactor=self.prefactor) +
                "Your input value {input} would be ignored.".format(input=prefactor))

        # === Variables that will be computed in methods ===
        self._qgpv_temp = None
        self._interpolated_u_temp = None
        self._interpolated_v_temp = None
        self._interpolated_avort_temp = None
        self._interpolated_theta_temp = None
        self._static_stability = None
        self._static_stability_n = None
        self._static_stability_s = None
        self._tn0 = None
        self._ts0 = None

        # === Temporary solution for direct inv routine
        self._f_qref, self._f_u, self._f_tref, self._f_ubar, self._f_tbar, self._f_fawa, self._f_ckref = \
            None, None, None, None, None, None, None

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
        self._northern_hemisphere_results_only: Optional[bool] = None  # Leave it uninitialized first.

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

        # Temporary solution for GRL computation
        self._ua1baro_nhem = None
        self._ua2baro_nhem = None
        self._ep1baro_nhem = None
        self._ep2baro_nhem = None
        self._ep3baro_nhem = None
        self._ep4_nhem = None

    def _compute_prefactor(self):
        """
        Private function. Compute prefactor for normalization by evaluating
            int^{z=kmax*dz}_0 e^{-z/H} dz
        using rectangular rule consistent with the integral evaluation in compute_lwa_and_barotropic_fluxes.f90.
        TODO: evaluate numerical integration scheme used in the fortran module.
        """
        self._prefactor = sum([math.exp(-k * self.dz / self.scale_height) * self.dz for k in range(1, self.kmax-1)])

    @property
    def prefactor(self):
        return self._prefactor

    def _check_valid_plev(self, plev, scale_height, kmax, dz):
        """
        Private function. Check the validity of plev to see
            1. if plev is in decending order
            2. if kmax is valid given the max pseudoheight in the input data

        Parameters
        ----------
        plev : numpy.ndarray
               Array of pressure level (in hPa) of size nlev.
        scale_height : float, optional
               Scale height of the atmosphere in meters. Default = 7000.
        kmax : int, optional
               Dimension of uniform pseudoheight grids used for interpolation.
        dz : float, optional
               Size of uniform pseudoheight grids (in meters).
        """
        # Check if plev is in decending order
        if np.diff(plev)[0] > 0:
            raise TypeError("plev must be in decending order (i.e. from ground level to aloft)")
        else:
            self.plev = plev

        # Check if kmax is valid given the max pseudoheight in the input data
        hmax = -scale_height*np.log(plev[-1]/P_GROUND)
        if hmax < (kmax-1) * dz:
            raise ValueError('Input kmax = {} but the maximum valid kmax'.format(kmax) +
                             '(constrainted by the vertical grid of your input data) is {}'.format(int(hmax//dz)+1))

    def _check_and_flip_ylat(self, ylat):
        """
        Private function. Check if ylat is in ascending order and include the equator. If not, create a new grid with
        odd number of grid points that include the equator.

        Parameters
        ----------
        ylat : numpy.array
               Array of evenly-spaced latitude (in degree) of size nlat.
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

    @staticmethod
    def _check_dimension_of_fields(field, field_name, expected_dim):
        """
        Private function. Check if the field of a specific field_name has the expected dimension expected_dim.
        """
        if field.shape != expected_dim:
            raise TypeError(
                "Incorrect dimension of {}. Expected dimension: {}"
                .format(field_name, expected_dim)
            )

    def _interp_back(self, field, interp_from, interp_to, which_axis=1):
        """
        Private function to interpolate the results from odd grid to even grid.
        If the initial input to the QGField object is an odd grid, error will be raised.
        """
        if self.ylat_no_equator is None:
            raise TypeError("No need for such interpolation.")
        else:
            return interp1d(
                interp_from, field, axis=which_axis, bounds_error=False,
                fill_value='extrapolate'
            )(interp_to)

    def _return_interp_variables(self, variable, interp_axis, northern_hemisphere_results_only=True):
        """
        Private function to return interpolated variables from odd grid back to even grid if originally
        the data was given on an odd grid.

        Parameters
        ----------
            variable(numpy.ndarray): the variable to be interpolated

            interp_axis(int): the index of axis for interpolation

            northern_hemisphere_results_only(bool): whether only to return northern hemispheric results.
                Will be deprecated in upcoming release.

        Returns
        -------
            The interpolated variable(numpy.ndarray)
        """
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

    def _compute_reference_state_wrapper(self, qgpv, u, theta):
        """
        Private function to call the fortran subroutine compute_reference_states that returns variable of
        dimension [nlat, kmax]. Swapping of axes is needed for other computation.

        Parameters
        ----------
            qgpv(numpy.ndarray): QGPV

            u(numpy.ndarray): 3D zonal wind

            theta(numpy.ndarray): 3D potential temperature

            num_of_iter(int): number of iteration when solving the eliptic equation

        Returns
        -------
            Qref(numpy.ndarray): Reference state of QGPV of dimension [nlat, kmax]

            Uref(numpy.ndarray): Reference state of zonal wind of dimension [nlat, kmax]

            PTref(numpy.ndarray): Reference state of potential temperature of dimension [nlat, kmax]
        """
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

    def _compute_lwa_and_barotropic_fluxes_wrapper(self, qgpv, u, v, theta, qref_temp, uref_temp, ptref_temp):
        """
        Private function. Wrapper to call the fortran subroutine compute_lwa_and_barotropic_fluxes.
        """
        return compute_lwa_and_barotropic_fluxes(
            qgpv, u, v, theta, qref_temp, uref_temp, ptref_temp,
            self.planet_radius, self.omega, self.dz, self.scale_height, self.dry_gas_constant, self.cp, self.prefactor)

    def interpolate_fields(self):

        """
        Interpolate zonal wind, maridional wind, and potential temperature field onto the uniform pseudoheight grids,
        and compute QGPV on the same grids. This returns named tuple called "Interpolated_fields" that consists of
        5 elements as listed below.

        Returns
        -------
        QGPV : numpy.ndarray
            Three-dimensional array of quasi-geostrophic potential vorticity (QGPV) with dimension = [kmax, nlat, nlon]

        U : numpy.ndarray
            Three-dimensional array of interpolated zonal wind with dimension = [kmax, nlat, nlon]

        V : numpy.ndarray
            Three-dimensional array of interpolated meridional wind with dimension = [kmax, nlat, nlon]

        Theta : numpy.ndarray
            Three-dimensional array of interpolated potential temperature with dimension = [kmax, nlat, nlon]

        Static_stability : numpy.array
            One-dimension array of interpolated static stability with dimension = kmax


        Examples
        --------

        >>> interpolated_fields = test_object.interpolate_fields()
        >>> interpolated_fields.QGPV  # This is to access the QGPV field

        """

        if self._qref_ntemp is None:

            # === Interpolate fields and obtain qgpv ===
            self._qgpv_temp, \
                self._interpolated_u_temp, \
                self._interpolated_v_temp, \
                self._interpolated_avort_temp, \
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

        # Construct a named tuple # TODO: add absolute vorticity here? But only after testing
        Interpolated_fields = namedtuple('Interpolated_fields', ['QGPV', 'U', 'V', 'Theta', 'Static_stability'])
        interpolated_fields = Interpolated_fields(
            self.qgpv,
            self.interpolated_u,
            self.interpolated_v,
            self.interpolated_theta,
            self.static_stability)
        return interpolated_fields

    def compute_reference_states(self, northern_hemisphere_results_only=False):

        """
        Compute the local wave activity and reference states of QGPV, zonal wind and potential temperature using a more
        stable inversion algorithm applied in Nakamura and Huang (2018, Science). The equation to be invert is
        equation (22) in supplementary materials of Huang and Nakamura (2017, GRL).

        This function returns named tuple called "Reference_states" that consists of 3 elements:

        Parameters
        ----------
        northern_hemisphere_results_only : bool
           If true, arrays of size [kmax, nlat//2+1] will be returned. Otherwise, arrays of size [kmax, nlat] will be
           returned. Default: False.

        Returns
        -------
        Qref : numpy.ndarray
            Two-dimensional array of reference state of quasi-geostrophic potential vorticity (QGPV) with
            dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False, or
            dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True

        Uref : numpy.ndarray
            Two-dimensional array of reference state of zonal wind (Uref) with dimension = [kmax, nlat, nlon]
            if northern_hemisphere_results_only=False, or dimension = [kmax, nlat//2+1, nlon] if
            northern_hemisphere_results_only=True

        PTref : numpy.ndarray
            Two-dimensional array of reference state of potential temperature (Theta_ref) with
            dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False, or
            dimension = [kmax, nlat//2+1, nlon] if northern_hemisphere_results_only=True

        Examples
        --------

        >>> qref, uref, ptref = test_object.compute_reference_states()

        """
        if self._northern_hemisphere_results_only is None:
            self._northern_hemisphere_results_only = northern_hemisphere_results_only
        else:
            if northern_hemisphere_results_only != self._northern_hemisphere_results_only:
                warnings.warn(
                    f"In compute_reference_states, input argument northern_hemisphere_results_only = " +
                    f"{northern_hemisphere_results_only} but self._northern_hemisphere_results_only has been " +
                    f"initialized to {self._northern_hemisphere_results_only} already. The following computation will " +
                    f"use self._northern_hemisphere_results_only = {self._northern_hemisphere_results_only}.")

        if self._qgpv_temp is None:
            self.interpolate_fields()

        # === Compute reference states in Northern Hemisphere ===
        self._qref_ntemp, self._uref_ntemp, self._ptref_ntemp, num_of_iter = self._compute_reference_state_wrapper(
            qgpv=self._qgpv_temp, u=self._interpolated_u_temp, theta=self._interpolated_theta_temp)
        if num_of_iter >= self.maxit:
            raise ValueError("The reference state does not converge for Northern Hemisphere.")
        # *** Convert Qref to the right unit
        qref_ntemp_right_unit = \
            self._qref_ntemp * 2 * self.omega * np.sin(np.deg2rad(self.ylat[(self.equator_idx - 1):, np.newaxis]))

        # === Compute reference states in Southern Hemisphere ===
        if not self._northern_hemisphere_results_only:
            self._qref_stemp, self._uref_stemp, self._ptref_stemp, num_of_iter = self._compute_reference_state_wrapper(
                qgpv=-self._qgpv_temp[:, ::-1, :],
                u=self._interpolated_u_temp[:, ::-1, :],
                theta=self._interpolated_theta_temp[:, ::-1, :])
            if num_of_iter >= self.maxit:
                raise ValueError("The reference state does not converge for Southern Hemisphere.")

            # *** Convert Qref to the right unit
            qref_stemp_right_unit = self._qref_stemp[::-1, :] * 2 * self.omega * np.sin(
                np.deg2rad(self.ylat[:self.equator_idx, np.newaxis]))

        if self._northern_hemisphere_results_only:
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

        # Construct a named tuple
        Reference_states = namedtuple('Reference_states', ['Qref', 'Uref', 'PTref'])
        reference_states = Reference_states(
            self.qref,
            self.uref,
            self.ptref)
        return reference_states

    def compute_lwa_and_barotropic_fluxes(self, northern_hemisphere_results_only=False):

        """
        Compute barotropic components of local wave activity and flux terms in eqs.(2) and (3) in
        Nakamura and Huang (Science, 2018). It returns a named tuple called "LWA_and_fluxes" that consists of
        9 elements as listed below. The discretization scheme that is used in the numerical integration is outlined
        in the Supplementary materials of Huang and Nakamura (GRL, 2017).

        Parameters
        ----------
        northern_hemisphere_results_only : bool
           If true, arrays of size [kmax, nlat//2+1] will be returned. Otherwise, arrays of size [kmax, nlat] will be
           returned. Default: False. If this variable has been initialized when calling compute_reference_states,
           the initialized value will be used. This variable will be deprecated in the next major release.

        Returns
        -------
        adv_flux_f1 : numpy.ndarray
            Two-dimensional array of the second-order eddy term in zonal advective flux,
            i.e. F1 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True,
            or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        adv_flux_f2 : numpy.ndarray
            Two-dimensional array of the third-order eddy term in zonal advective flux,
            i.e. F2 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True,
            or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        adv_flux_f3 : numpy.ndarray
            Two-dimensional array of the remaining term in zonal advective flux,
            i.e. F3 in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True,
            or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        convergence_zonal_advective_flux : numpy.ndarray
            Two-dimensional array of the convergence of zonal advective flux,
            i.e. -div(F1+F2+F3) in equation 3 of NH18, with dimension = [nlat//2+1, nlon] if
            northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        divergence_eddy_momentum_flux : numpy.ndarray
            Two-dimensional array of the divergence of eddy momentum flux,
            i.e. (II) in equation 2 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True,
            or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        meridional_heat_flux : numpy.ndarray
            Two-dimensional array of the low-level meridional heat flux,
            i.e. (III) in equation 2 of NH18, with dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True,
            or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        lwa_baro : np.ndarray
            Two-dimensional array of barotropic local wave activity (with cosine weighting).
            Dimension = [nlat//2+1, nlon] if northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if
            northern_hemisphere_results_only=False.

        u_baro : np.ndarray
            Two-dimensional array of barotropic zonal wind (without cosine weighting). Dimension = [nlat//2+1, nlon] if
            northern_hemisphere_results_only=True, or dimension = [nlat, nlon] if northern_hemisphere_results_only=False.

        lwa : np.ndarray
            Three-dimensional array of barotropic local wave activity Dimension = [kmax, nlat//2+1, nlon] if
            northern_hemisphere_results_only=True, or dimension = [kmax, nlat, nlon] if northern_hemisphere_results_only=False.

        Examples
        --------

        >>> adv_flux_f1, adv_flux_f2, adv_flux_f3, convergence_zonal_advective_flux,
            divergence_eddy_momentum_flux, meridional_heat_flux,
            lwa_baro, u_baro, lwa = test_object.compute_lwa_and_barotropic_fluxes()
        """

        # Check if previous steps have been done.
        if self._qgpv_temp is None:
            self.interpolate_fields()

        if self._uref_ntemp is None:
            self.compute_reference_states(northern_hemisphere_results_only)

        if northern_hemisphere_results_only != self._northern_hemisphere_results_only:
            warnings.warn(
                f"In compute_lwa_and_barotropic_fluxes, input argument northern_hemisphere_results_only = " +
                f"{northern_hemisphere_results_only} but self._northern_hemisphere_results_only has been " +
                f"initialized to {self._northern_hemisphere_results_only} already. The following computation will " +
                f"use self._northern_hemisphere_results_only = {self._northern_hemisphere_results_only}.")

        # === Compute barotropic flux terms (NHem) ===
        lwa_nhem, astarbaro_nhem, ua1baro_nhem, ubaro_nhem, ua2baro_nhem,\
            ep1baro_nhem, ep2baro_nhem, ep3baro_nhem, ep4_nhem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                self._qgpv_temp,
                self._interpolated_u_temp,
                self._interpolated_v_temp,
                self._interpolated_theta_temp,
                self._qref_ntemp,
                self._uref_ntemp,
                self._ptref_ntemp)

        # === Access barotropic components of ua1, ua2, ep1, ep2, ep3, ep4: for the use of nhn GLR paper only ===
        self._ua1baro_nhem = ua1baro_nhem
        self._ua2baro_nhem = ua2baro_nhem
        self._ep1baro_nhem = ep1baro_nhem
        self._ep2baro_nhem = ep2baro_nhem
        self._ep3baro_nhem = ep3baro_nhem
        self._ep4_nhem = ep4_nhem

        # === Compute barotropic flux terms (SHem) ===
        if not self._northern_hemisphere_results_only:
            lwa_shem, astarbaro_shem, ua1baro_shem, ubaro_shem, ua2baro_shem,\
                ep1baro_shem, ep2baro_shem, ep3baro_shem, ep4_shem = \
                self._compute_lwa_and_barotropic_fluxes_wrapper(
                    -self._qgpv_temp[:, ::-1, :],
                    self._interpolated_u_temp[:, ::-1, :],
                    self._interpolated_v_temp[:, ::-1, :],
                    self._interpolated_theta_temp[:, ::-1, :],
                    self._qref_stemp,
                    self._uref_stemp,
                    self._ptref_stemp)

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
        if not self._northern_hemisphere_results_only:
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

        if self._northern_hemisphere_results_only:
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

            # Negative sign for southern hemisphere upon flipping (via Coriolis parameter)
            self._meridional_heat_flux = \
                np.vstack((np.swapaxes(-ep4_shem[:, ::-1], 0, 1),
                           np.swapaxes(ep4_nhem[:, 1:], 0, 1)))

            self._lwa_baro = \
                np.vstack((np.swapaxes(astarbaro_shem[:, ::-1], 0, 1),
                           np.swapaxes(astarbaro_nhem[:, 1:], 0, 1)))

            self._u_baro = np.vstack((np.swapaxes(ubaro_shem[:, ::-1], 0, 1),
                                      np.swapaxes(ubaro_nhem[:, 1:], 0, 1)))

            self._lwa = np.concatenate((np.swapaxes(lwa_shem[:, ::-1], 0, 2),
                                        np.swapaxes(lwa_nhem[:, 1:], 0, 2)), axis=1)

            self._divergence_eddy_momentum_flux = np.vstack((np.swapaxes(-meri_flux_shem_temp[:, ::-1], 0, 1),
                                                             np.swapaxes(meri_flux_nhem_temp[:, 1:], 0, 1)))

        # Construct a named tuple
        LWA_and_fluxes = namedtuple(
            'LWA_and_fluxes',
            ['adv_flux_f1', 'adv_flux_f2', 'adv_flux_f3', 'convergence_zonal_advective_flux',
             'divergence_eddy_momentum_flux', 'meridional_heat_flux', 'lwa_baro', 'u_baro', 'lwa'])
        lwa_and_fluxes = LWA_and_fluxes(
            self.adv_flux_f1, self.adv_flux_f2, self.adv_flux_f3, self.convergence_zonal_advective_flux,
            self.divergence_eddy_momentum_flux, self.meridional_heat_flux, self.lwa_baro, self.u_baro, self.lwa)
        return lwa_and_fluxes

    # *** Added in Release 0.6.0 ***
    # The following internal functions are used to compute results in NHN (2022, GRL):
    # - _interpolate_field_dirinv
    # - _compute_qref_fawa_and_bc
    # - _compute_lwa_flux_dirinv
    # They will be refactored in the upcoming releases.
    def _interpolate_field_dirinv(self):
        """
        Added for NHN 2022 GRL

        .. versionadded:: 0.6.0
        """
        self._qgpv_temp, \
        self._interpolated_u_temp, \
        self._interpolated_v_temp, \
        self._interpolated_avort_temp, \
        self._interpolated_theta_temp, \
        self._static_stability_n, \
        self._static_stability_s,\
        self._tn0, self._ts0 = \
            interpolate_fields_direct_inv(
                self.kmax,
                self.nlat // 2 + self.nlat % 2,
                np.swapaxes(self.u_field, 0, 2),
                np.swapaxes(self.v_field, 0, 2),
                np.swapaxes(self.t_field, 0, 2),
                self.plev,
                self.planet_radius,
                self.omega,
                self.dz,
                self.scale_height,
                self.dry_gas_constant,
                self.cp)

        self._check_nan("self._qgpv_temp", self._qgpv_temp)
        self._check_nan("self._interpolated_u_temp", self._interpolated_u_temp)
        self._check_nan("self._interpolated_v_temp", self._interpolated_v_temp)
        self._check_nan("self._interpolated_avort_temp", self._interpolated_avort_temp)
        self._check_nan("self._interpolated_theta_temp", self._interpolated_theta_temp)
        self._check_nan("self._static_stability_n", self._static_stability_n)
        self._check_nan("self._static_stability_s", self._static_stability_s)
        self._check_nan("self._tn0", self._tn0)
        self._check_nan("self._ts0", self._ts0)

        return self._qgpv_temp, self._interpolated_u_temp, self._interpolated_v_temp,  self._interpolated_avort_temp, \
        self._interpolated_theta_temp, self._static_stability_n, self._static_stability_s, self._tn0, self._ts0

    @staticmethod
    def _check_nan(name, var):
        nan_num = np.count_nonzero(np.isnan(var))
        if nan_num > 0:
            print(f"num of nan in {name}: {np.count_nonzero(np.isnan(var))}.")

    def _compute_qref_fawa_and_bc(self):
        """
        Added for NHN 2022 GRL

        .. versionadded:: 0.6.0
        """
        # ans = compute_qref_and_fawa_first(
            # pv, uu, vort, pt, nd, nnd, jb, jd, aa, omega, dz, h, rr, cp)
        ans = compute_qref_and_fawa_first(
            pv=self._qgpv_temp,
            uu=self._interpolated_u_temp,
            vort=self._interpolated_avort_temp,
            pt=self._interpolated_theta_temp,
            tn0=self._tn0,
            nd=self.nlat//2 + self.nlat % 2,  # 91
            nnd=self.nlat,                    # 181
            jb=self.eq_boundary_index,        # 5
            jd=self.nlat//2 + self.nlat % 2 - self.eq_boundary_index,  # 86 TODO fix its formula
            a=self.planet_radius,
            omega=self.omega,
            dz=self.dz,
            h=self.scale_height,
            rr=self.dry_gas_constant,
            cp=self.cp)

        qref_over_cor, ubar, tbar, fawa, ckref, tjk, sjk = ans  # unpack tuple

        self._check_nan("qref_over_cor", qref_over_cor)
        self._check_nan("ubar", ubar)
        self._check_nan("tbar", tbar)
        self._check_nan("fawa", fawa)
        self._check_nan("ckref", ckref)
        self._check_nan("tjk", tjk)
        self._check_nan("sjk", sjk)

        for k in range(self.kmax-1, 1, -1):  # Fortran indices
            ans = matrix_b4_inversion(
                k=k,
                jmax=self.nlat,
                jb=self.eq_boundary_index,  # 5
                jd=self.nlat // 2 + self.nlat % 2 - self.eq_boundary_index,  # 86
                z=np.arange(0, self.kmax*self.dz, self.dz),
                statn=self._static_stability_n,
                qref=qref_over_cor,
                ckref=ckref,
                a=self.planet_radius,
                om=self.omega,
                dz=self.dz,
                h=self.scale_height,
                rr=self.dry_gas_constant,
                cp=self.cp,
                sjk=sjk,
                tjk=tjk)
            qjj, djj, cjj, rj, tj = ans

            # TODO: The inversion algorithm  is the bottleneck of the computation
            # SciPy is very slow compared to MKL in Fortran...
            lu, piv, info = dgetrf(qjj)
            qjj, info = dgetri(lu, piv)

            _ = matrix_after_inversion(
                k=k,
                qjj=qjj,
                djj=djj,
                cjj=cjj,
                tj=tj,
                rj=rj,
                sjk=sjk,
                tjk=tjk)

        tref, qref, u = upward_sweep(
            jmax=self.nlat,
            jb=self.eq_boundary_index,
            sjk=sjk,
            tjk=tjk,
            ckref=ckref,
            tb=self._tn0,
            qref_over_cor=qref_over_cor,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            h=self.scale_height,
            rr=self.dry_gas_constant,
            cp=self.cp)

        return qref, u, tref, fawa, ubar, tbar  # uref = u

    def _compute_lwa_flux_dirinv(self, qref, uref, tref):
        """
        Added for NHN 2022 GRL

        .. versionadded:: 0.6.0
        """
        ans = compute_flux_dirinv(pv=self._qgpv_temp, uu=self._interpolated_u_temp, vv=self._interpolated_v_temp,
                                  pt=self._interpolated_theta_temp, tn0=self._tn0,
                                  qref=qref, uref=uref, tref=tref,
                                  jb=self.eq_boundary_index, a=self.planet_radius, om=self.omega,
                                  dz=self.dz, h=self.scale_height, rr=self.dry_gas_constant, cp=self.cp,
                                  prefac=self.prefactor)
        # astarbaro, ubaro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4, astar1, astar2 = ans
        return ans

    @property
    def qgpv(self):
        """
        Quasi-geostrophic potential vorticity on the regular pseudoheight grids.
        """
        if self._qgpv is None:
            raise ValueError('QGPV field is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._qgpv, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_u(self):
        """
        Zonal wind on the regular pseudoheight grids.
        """
        if self._interpolated_u is None:
            raise ValueError('interpolated_u is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_u, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_v(self):
        """
        Meridional wind on the regular pseudoheight grids.
        """
        if self._interpolated_v is None:
            raise ValueError('interpolated_v is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_v, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def interpolated_theta(self):
        """
        Potential temperature on the regular pseudoheight grids.
        """
        if self._interpolated_theta is None:
            raise ValueError('interpolated_theta is not present in the QGField object.')
        return self._return_interp_variables(
            variable=self._interpolated_theta, interp_axis=1, northern_hemisphere_results_only=False)

    @property
    def static_stability(self):
        """
        The interpolated static stability.
        """
        return self._static_stability

    @property
    def qref(self):
        """
        Reference state of QGPV (Qref).
        """
        if self._qref is None:
            raise ValueError('qref is not computed yet.')
        return self._return_interp_variables(variable=self._qref, interp_axis=1,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def uref(self):
        """
        Reference state of zonal wind (Uref).
        """
        if self._uref is None:
            raise ValueError('uref field is not computed yet.')
        return self._return_interp_variables(variable=self._uref, interp_axis=1,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def ptref(self):
        """
        Reference state of potential temperature (\\Theta_ref).
        """
        if self._ptref is None:
            raise ValueError('ptref field is not computed yet.')
        return self._return_interp_variables(variable=self._ptref, interp_axis=1,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def adv_flux_f1(self):
        """
        Two-dimensional array of the second-order eddy term in zonal advective flux, i.e. F1 in equation 3 of NH18
        """
        if self._adv_flux_f1 is None:
            raise ValueError('adv_flux_f1 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f1, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def adv_flux_f2(self):
        """
        Two-dimensional array of the third-order eddy term in zonal advective flux, i.e. F2 in equation 3 of NH18
        """
        if self._adv_flux_f2 is None:
            raise ValueError('adv_flux_f2 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f2, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def adv_flux_f3(self):
        """
        Two-dimensional array of the remaining term in zonal advective flux, i.e. F3 in equation 3 of NH18
        """
        if self._adv_flux_f3 is None:
            raise ValueError('adv_flux_f3 is not computed yet.')
        return self._return_interp_variables(variable=self._adv_flux_f3, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def convergence_zonal_advective_flux(self):
        """
        Two-dimensional array of the convergence of zonal advective flux, i.e. -div(F1+F2+F3) in equation 3 of NH18
        """
        if self._convergence_zonal_advective_flux is None:
            raise ValueError('convergence_zonal_advective_flux is not computed yet.')
        return self._return_interp_variables(variable=self._convergence_zonal_advective_flux, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def divergence_eddy_momentum_flux(self):
        """
        Two-dimensional array of the divergence of eddy momentum flux, i.e. (II) in equation 2 of NH18
        """
        if self._divergence_eddy_momentum_flux is None:
            raise ValueError('divergence_eddy_momentum_flux is not computed yet.')
        return self._return_interp_variables(variable=self._divergence_eddy_momentum_flux, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def meridional_heat_flux(self):
        """
        Two-dimensional array of the low-level meridional heat flux, i.e. (III) in equation 2 of NH18
        """
        if self._meridional_heat_flux is None:
            raise ValueError('meridional_heat_flux is not computed yet.')
        return self._return_interp_variables(variable=self._meridional_heat_flux, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def lwa_baro(self):
        """
        Two-dimensional array of barotropic local wave activity (with cosine weighting).
        """
        if self._lwa_baro is None:
            raise ValueError('lwa_baro is not computed yet.')
        return self._return_interp_variables(variable=self._lwa_baro, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def u_baro(self):
        """
        Two-dimensional array of barotropic zonal wind (without cosine weighting).
        """
        if self._u_baro is None:
            raise ValueError('u_baro is not computed yet.')
        return self._return_interp_variables(variable=self._u_baro, interp_axis=0,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    @property
    def lwa(self):
        """
        Three-dimensional array of barotropic local wave activity
        """
        if self._lwa is None:
            raise ValueError('lwa is not computed yet.')
        return self._return_interp_variables(variable=self._lwa, interp_axis=1,
                                             northern_hemisphere_results_only=self._northern_hemisphere_results_only)

    def get_latitude_dim(self):
        """
        Return the latitude dimension of the input data.
        """
        if self.need_latitude_interpolation:
            return self.ylat_no_equator.size
        else:
            return self.nlat

