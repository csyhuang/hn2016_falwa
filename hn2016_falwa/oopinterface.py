"""
------------------------------------------
File name: oopinterface.py
Author: Clare Huang
"""
from typing import Tuple, Optional, Union, NamedTuple
from abc import ABC, abstractmethod
import math
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg.lapack import dgetrf, dgetri

from hn2016_falwa import utilities
from hn2016_falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA
from hn2016_falwa.data_storage import InterpolatedFieldsStorage, DomainAverageStorage, ReferenceStatesStorage, \
    LWAStorage, BarotropicFluxTermsStorage, OutputBarotropicFluxTermsStorage

# *** Import f2py modules ***
from hn2016_falwa import interpolate_fields, interpolate_fields_direct_inv, compute_qref_and_fawa_first,\
    matrix_b4_inversion, matrix_after_inversion, upward_sweep, compute_flux_dirinv_nshem, compute_reference_states,\
    compute_lwa_and_barotropic_fluxes
from collections import namedtuple


class QGFieldBase(ABC):

    """
    Local wave activity and flux analysis in the quasi-geostrophic framework.

    .. warning::
        This is an abstract class that defines the public interface but does
        not define any boundary conditions for the reference state computation.
        Instanciate via the specific child classes :py:class:`QGFieldNH18` or
        :py:class:`QGFieldNHN22` to select the desired boundary conditions.

    Topography is assumed flat in this object.

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
    northern_hemisphere_results_only : bool, optional
           whether only to return northern hemispheric results. Default = False
    """

    def __init__(self, xlon, ylat, plev, u_field, v_field, t_field, kmax=49, maxit=100000, dz=1000., npart=None,
                 tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=False):

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

        # === Coordinate-related ===
        self.dphi = np.deg2rad(180./(self.nlat-1))
        self.dlambda = np.deg2rad(self.xlon[1] - self.xlon[0])
        self.slat = np.sin(np.deg2rad(ylat))  # sin latitude
        self.clat = np.cos(np.deg2rad(ylat))  # sin latitude
        self.npart = npart if npart is not None else self.nlat
        self.kmax = kmax
        self.height = np.array([i * dz for i in range(kmax)])

        # === Moved here in v0.7.0 ===
        self._northern_hemisphere_results_only = northern_hemisphere_results_only

        # === Other parameters ===
        self.maxit = maxit
        self.dz = dz
        self.tol = tol
        self.rjac = rjac

        # === Constants ===
        self.scale_height = scale_height
        self.cp = cp
        self.dry_gas_constant = dry_gas_constant
        self.omega = omega
        self.planet_radius = planet_radius
        self._compute_prefactor()  # Compute normalization prefactor

        # === qgpv, u, v, avort, theta encapsulated in InterpolatedFieldsStorage ===
        self._interpolated_field_storage = InterpolatedFieldsStorage(
            pydim=(self.kmax, self.nlat, self.nlon),
            fdim=(self.nlon, self.nlat, self.kmax),
            swapaxis_1=0,
            swapaxis_2=2,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)
        # Global averaged quantities (TODO: encalsulate them later)
        self._domain_average_storage = DomainAverageStorage(
            pydim=self.kmax,
            fdim=self.kmax,
            swapaxis_1=0,
            swapaxis_2=2,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)

        # Reference states
        lat_dim = self.nlat // 2 + 1 if self.northern_hemisphere_results_only else self.nlat
        self._reference_states_storage = ReferenceStatesStorage(
            pydim=(self.kmax, lat_dim),
            fdim=(lat_dim, self.kmax),
            swapaxis_1=0,
            swapaxis_2=1,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)

        # LWA storage (3D)
        self._lwa_storage = LWAStorage(
            pydim=(self.kmax, lat_dim, self.nlon),
            fdim=(self.nlon, lat_dim, self.kmax),
            swapaxis_1=0,
            swapaxis_2=2,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)

        # barotropic flux term storage (2D)
        self._barotropic_flux_terms_storage = BarotropicFluxTermsStorage(
            pydim=(lat_dim, self.nlon),
            fdim=(self.nlon, lat_dim),
            swapaxis_1=0,
            swapaxis_2=1,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)

        # output barotropic flux term storage (2D)
        self._output_barotropic_flux_terms_storage = OutputBarotropicFluxTermsStorage(
            pydim=(lat_dim, self.nlon),
            fdim=(self.nlon, lat_dim),
            swapaxis_1=0,
            swapaxis_2=1,
            northern_hemisphere_results_only=self.northern_hemisphere_results_only)

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
        self.plev = plev
        self.zlev = -scale_height * np.log(plev/P_GROUND)

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

    def _return_interp_variables(self, variable, interp_axis):
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
            if self.northern_hemisphere_results_only:
                return self._interp_back(
                    variable, self.ylat[-(self.nlat//2+1):],
                    self.ylat_no_equator[-(self.nlat//2):],
                    which_axis=interp_axis)
            else:
                return self._interp_back(variable, self.ylat, self.ylat_no_equator, which_axis=interp_axis)
        else:
            return variable

    def _compute_lwa_and_barotropic_fluxes_wrapper(self, qgpv, u, v, theta, qref_temp, uref_temp, ptref_temp):
        """
        Private function. Wrapper to call the fortran subroutine compute_lwa_and_barotropic_fluxes.
        """
        return compute_lwa_and_barotropic_fluxes(
            qgpv, u, v, theta, qref_temp, uref_temp, ptref_temp,
            self.planet_radius, self.omega, self.dz, self.scale_height, self.dry_gas_constant, self.cp, self.prefactor)

    def interpolate_fields(self, return_named_tuple: bool = True) -> Optional[NamedTuple]:

        """
        Interpolate zonal wind, maridional wind, and potential temperature field onto the uniform pseudoheight grids,
        and compute QGPV on the same grids. This returns named tuple called "Interpolated_fields" that consists of
        5 elements as listed below.

        Parameters
        ----------
        return_named_tuple : bool
           Whether to returned a named tuple with variables in python indexing. Default: True. If False, nothing will be
           returned from this method. The variables can be retrieved from the QGField object after all computation is
           finished. This may save run time in some use case.

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

        # Return a named tuple
        Interpolated_fields_to_return = namedtuple(
            'Interpolated_fields', ['QGPV', 'U', 'V', 'Theta', 'Static_stability'])

        interpolated_fields_tuple = self._interpolate_fields(Interpolated_fields_to_return, return_named_tuple)

        # TODO: warn that for NHN22, static stability returned would be a tuple of ndarray
        if return_named_tuple:
            return interpolated_fields_tuple

    @abstractmethod
    def _interpolate_fields(self, Interpolated_fields_to_return: NamedTuple, return_named_tuple: bool) -> Optional[NamedTuple]:
        """
        The specific interpolation procedures w.r.t the particular procedures in the paper will be implemented here.
        """

    def compute_reference_states(self, return_named_tuple: bool = True, northern_hemisphere_results_only=None) -> Optional[NamedTuple]:

        """
        Compute the local wave activity and reference states of QGPV, zonal wind and potential temperature using a more
        stable inversion algorithm applied in Nakamura and Huang (2018, Science). The equation to be invert is
        equation (22) in supplementary materials of Huang and Nakamura (2017, GRL).

        The parameter `northern_hemisphere_results_only` is deprecated and has no effect.

        Parameters
        ----------
        return_named_tuple : bool
           Whether to returned a named tuple with variables in python indexing. Default: True. If False, nothing will be
           returned from this method. The variables can be retrieved from the QGField object after all computation is
           finished. This may save run time in some use case.

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

        if northern_hemisphere_results_only:
            warnings.warn(
                f"""
                Since v0.7.0, northern_hemisphere_results_only is initialized at the creation of QGField instance.
                The value of self.northern_hemisphere_results_only = {self.northern_hemisphere_results_only} but
                your input here is northern_hemisphere_results_only = {northern_hemisphere_results_only}. 
                Please remove this input argument from the method 'compute_reference_states'.
                """)

        if self.qgpv is None:
            raise ValueError("QGField.interpolate_fields has to be called before QGField.compute_reference_states.")

        self._compute_reference_states()

        # *** Return a named tuple ***
        if return_named_tuple:
            Reference_states = namedtuple('Reference_states', ['Qref', 'Uref', 'PTref'])
            reference_states = Reference_states(
                self.qref,
                self.uref,
                self.ptref)
            return reference_states

    @abstractmethod
    def _compute_reference_states(self):
        """
        Reference state computation with boundary conditions and procedures specified in the paper will be
        implemented here.
        """

    def compute_lwa_and_barotropic_fluxes(self, return_named_tuple: bool = True, northern_hemisphere_results_only=None):

        """
        Compute barotropic components of local wave activity and flux terms in eqs.(2) and (3) in
        Nakamura and Huang (Science, 2018). It returns a named tuple called "LWA_and_fluxes" that consists of
        9 elements as listed below. The discretization scheme that is used in the numerical integration is outlined
        in the Supplementary materials of Huang and Nakamura (GRL, 2017).

        The parameter `northern_hemisphere_results_only` is deprecated and has no effect.

        Note that flux computation for NHN22 is still experimental.

        Parameters
        ----------
        return_named_tuple : bool
           Whether to returned a named tuple with variables in python indexing. Default: True. If False, nothing will be
           returned from this method. The variables can be retrieved from the QGField object after all computation is
           finished. This may save run time in some use case.

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

        if northern_hemisphere_results_only:
            warnings.warn(
                f"""
                Since v0.7.0, northern_hemisphere_results_only is initialized at the creation of QGField instance.
                The value of self.northern_hemisphere_results_only = {self.northern_hemisphere_results_only} but
                your input here is northern_hemisphere_results_only = {northern_hemisphere_results_only}. 
                Please remove this input argument from the method 'compute_lwa_and_barotropic_fluxes'.
                """)

        # Check if previous steps have been done.
        if self.qgpv is None:
            raise ValueError("QGField.interpolate_fields has to be called before QGField.compute_reference_states.")

        # TODO: need a check for reference states computed. If not, throw an error.
        self._compute_intermediate_flux_terms()

        # *** Compute named fluxes in NH18 ***
        clat = self.clat[-self.equator_idx:] if self.northern_hemisphere_results_only else self.clat
        self._output_barotropic_flux_terms_storage.divergence_eddy_momentum_flux = \
            np.swapaxes(
                (self._barotropic_flux_terms_storage.ep2baro - self._barotropic_flux_terms_storage.ep3baro) / \
                (2 * self.planet_radius * self.dphi * clat), 0, 1)

        zonal_adv_flux_sum = np.swapaxes((
            self._barotropic_flux_terms_storage.ua1baro
            + self._barotropic_flux_terms_storage.ua2baro
            + self._barotropic_flux_terms_storage.ep1baro), 0, 1)
        self._output_barotropic_flux_terms_storage.convergence_zonal_advective_flux = \
            utilities.zonal_convergence(
                field=zonal_adv_flux_sum,
                clat=clat,
                dlambda=self.dlambda,
                planet_radius=self.planet_radius)
        self._output_barotropic_flux_terms_storage.adv_flux_f1 = \
            self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.ua1baro)
        self._output_barotropic_flux_terms_storage.adv_flux_f2 = \
            self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.ua2baro)
        self._output_barotropic_flux_terms_storage.adv_flux_f3 = \
            self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.ep1baro)
        self._output_barotropic_flux_terms_storage.meridional_heat_flux = \
            self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.ep4)

        # *** Return the named tuple ***
        if return_named_tuple:
            LWA_and_fluxes = namedtuple(
                'LWA_and_fluxes',
                ['adv_flux_f1', 'adv_flux_f2', 'adv_flux_f3', 'convergence_zonal_advective_flux',
                 'divergence_eddy_momentum_flux', 'meridional_heat_flux', 'lwa_baro', 'u_baro', 'lwa'])
            lwa_and_fluxes = LWA_and_fluxes(
                self._output_barotropic_flux_terms_storage.adv_flux_f1,
                self._output_barotropic_flux_terms_storage.adv_flux_f2,
                self._output_barotropic_flux_terms_storage.adv_flux_f3,
                self._output_barotropic_flux_terms_storage.convergence_zonal_advective_flux,
                self._output_barotropic_flux_terms_storage.divergence_eddy_momentum_flux,
                self._output_barotropic_flux_terms_storage.meridional_heat_flux,
                self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.lwa_baro),
                self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.u_baro),
                self._lwa_storage.fortran_to_python(self._lwa_storage.lwa))
            return lwa_and_fluxes

    @abstractmethod
    def _compute_intermediate_flux_terms(self):
        """
        Compute ua1, ua2, ep1, ep2, ep3, ep4 depending on which BC protocol to use.
        """

    @staticmethod
    def _check_nan(name, var):
        nan_num = np.count_nonzero(np.isnan(var))
        if nan_num > 0:
            print(f"num of nan in {name}: {np.count_nonzero(np.isnan(var))}.")

    # *** Fixed properties (since creation of instance) ***
    @property
    def prefactor(self):
        """Normalization constant for vertical weighted-averaged integration"""
        return self._prefactor

    @property
    def ylat_ref_states(self) -> np.array:
        """
        Latitude dimension of reference state
        """
        if self.northern_hemisphere_results_only:
            return self.ylat[-(self.nlat//2+1):]
        return self.ylat

    @property
    def northern_hemisphere_results_only(self) -> bool:
        """
        Even though a global field is required for input, whether ref state and fluxes are computed for
        northern hemisphere only
        """
        return self._northern_hemisphere_results_only

    # *** Derived physical quantities ***
    @property
    def qgpv(self):
        """
        Quasi-geostrophic potential vorticity on the regular pseudoheight grids.
        """
        if self._interpolated_field_storage.qgpv is None:
            raise ValueError('QGPV field is not present in the QGField object.')
        return self._return_interp_variables(variable=self._interpolated_field_storage.fortran_to_python(
            self._interpolated_field_storage.qgpv), interp_axis=1)

    @property
    def interpolated_u(self):
        """
        Zonal wind on the regular pseudoheight grids.
        """
        if self._interpolated_field_storage.interpolated_u is None:
            raise ValueError('interpolated_u is not present in the QGField object.')
        return self._return_interp_variables(variable=self._interpolated_field_storage.fortran_to_python(
            self._interpolated_field_storage.interpolated_u), interp_axis=1)

    @property
    def interpolated_v(self):
        """
        Meridional wind on the regular pseudoheight grids.
        """
        if self._interpolated_field_storage.interpolated_v is None:
            raise ValueError('interpolated_v is not present in the QGField object.')
        return self._return_interp_variables(variable=self._interpolated_field_storage.fortran_to_python(
            self._interpolated_field_storage.interpolated_v), interp_axis=1)

    @property
    def interpolated_theta(self):
        """
        Potential temperature on the regular pseudoheight grids.
        """
        if self._interpolated_field_storage.interpolated_theta is None:
            raise ValueError('interpolated_theta is not present in the QGField object.')
        return self._return_interp_variables(variable=self._interpolated_field_storage.fortran_to_python(
            self._interpolated_field_storage.interpolated_theta), interp_axis=1)

    @property
    @abstractmethod
    def static_stability(self) -> Union[np.array, Tuple[np.array, np.array]]:
        """
        The interpolated static stability.
        """

    @property
    def qref(self):
        """
        Reference state of QGPV (Qref).
        """
        if self._reference_states_storage.qref is None:
            raise ValueError('qref is not computed yet.')
        return self._return_interp_variables(
            variable=self._reference_states_storage.qref_correct_unit(
                self.ylat_ref_states, self.omega), interp_axis=1)

    @property
    def uref(self):
        """
        Reference state of zonal wind (Uref).
        """
        if self._reference_states_storage.uref is None:
            raise ValueError('uref is not computed yet.')
        return self._return_interp_variables(
            variable=self._reference_states_storage.fortran_to_python(self._reference_states_storage.uref), interp_axis=1)

    @property
    def ptref(self):
        """
        Reference state of potential temperature (\\Theta_ref).
        """
        if self._reference_states_storage.ptref is None:
            raise ValueError('ptref is not computed yet.')
        return self._return_interp_variables(
            variable=self._reference_states_storage.fortran_to_python(self._reference_states_storage.ptref), interp_axis=1)

    @property
    def adv_flux_f1(self):
        """
        Two-dimensional array of the second-order eddy term in zonal advective flux, i.e. F1 in equation 3 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.adv_flux_f1,
            interp_axis=0)

    @property
    def adv_flux_f2(self):
        """
        Two-dimensional array of the third-order eddy term in zonal advective flux, i.e. F2 in equation 3 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.adv_flux_f2,
            interp_axis=0)

    @property
    def adv_flux_f3(self):
        """
        Two-dimensional array of the remaining term in zonal advective flux, i.e. F3 in equation 3 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.adv_flux_f3,
            interp_axis=0)

    @property
    def convergence_zonal_advective_flux(self):
        """
        Two-dimensional array of the convergence of zonal advective flux, i.e. -div(F1+F2+F3) in equation 3 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.convergence_zonal_advective_flux,
            interp_axis=0)

    @property
    def divergence_eddy_momentum_flux(self):
        """
        Two-dimensional array of the divergence of eddy momentum flux, i.e. (II) in equation 2 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.divergence_eddy_momentum_flux,
            interp_axis=0)

    @property
    def meridional_heat_flux(self):
        """
        Two-dimensional array of the low-level meridional heat flux, i.e. (III) in equation 2 of NH18
        """
        return self._return_interp_variables(
            variable=self._output_barotropic_flux_terms_storage.meridional_heat_flux,
            interp_axis=0)

    @property
    def lwa_baro(self):
        """
        Two-dimensional array of barotropic local wave activity (with cosine weighting).
        """
        if self._barotropic_flux_terms_storage.lwa_baro is None:
            raise ValueError('lwa_baro is not computed yet.')
        return self._return_interp_variables(
            variable=self._barotropic_flux_terms_storage.fortran_to_python(
                self._barotropic_flux_terms_storage.lwa_baro),
            interp_axis=0)

    @property
    def u_baro(self):
        """
        Two-dimensional array of barotropic zonal wind (without cosine weighting).
        """
        if self._barotropic_flux_terms_storage.u_baro is None:
            raise ValueError('u_baro is not computed yet.')
        return self._return_interp_variables(
            variable=self._barotropic_flux_terms_storage.fortran_to_python(
                self._barotropic_flux_terms_storage.u_baro),
            interp_axis=0)

    @property
    def lwa(self):
        """
        Three-dimensional array of local wave activity
        """
        if self._lwa_storage.lwa is None:
            raise ValueError('lwa is not computed yet.')
        return self._return_interp_variables(
            variable=self._lwa_storage.fortran_to_python(self._lwa_storage.lwa),
            interp_axis=1)

    def get_latitude_dim(self):
        """
        Return the latitude dimension of the input data.
        """
        if self.need_latitude_interpolation:
            return self.ylat_no_equator.size
        else:
            return self.nlat


class QGFieldNH18(QGFieldBase):
    """
    Procedures and reference state computation with the set of boundary conditions of NH18:

        Nakamura, N., & Huang, C. S. (2018). Atmospheric blocking as a traffic jam in the jet stream. Science, 361(6397), 42-47.
        https://www.science.org/doi/10.1126/science.aat0721

    See the documentation of :py:class:`QGField` for the public interface.
    There are no additional arguments for this class.

    .. versionadded:: 0.7.0

    Examples
    --------
    :doc:`notebooks/demo_script_for_nh2018`
    """

    def _interpolate_fields(self, Interpolated_fields_to_return, return_named_tuple) -> Optional[NamedTuple]:
        """
        .. versionadded:: 0.7.0
        """
        self._interpolated_field_storage.qgpv, \
            self._interpolated_field_storage.interpolated_u, \
            self._interpolated_field_storage.interpolated_v, \
            self._interpolated_field_storage.interpolated_avort, \
            self._interpolated_field_storage.interpolated_theta, \
            self._domain_average_storage.static_stability = interpolate_fields(  # f2py module
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
                self.cp)

        if return_named_tuple:
            interpolated_fields = Interpolated_fields_to_return(
                self.qgpv,
                self.interpolated_u,
                self.interpolated_v,
                self.interpolated_theta,
                self._domain_average_storage.static_stability)
            return interpolated_fields

    def _compute_reference_states(self):
        """
        .. versionadded:: 0.7.0
        """
        # *** Compute reference states in Northern Hemisphere using SOR ***
        self._reference_states_storage.qref_nhem, \
            self._reference_states_storage.uref_nhem, \
            self._reference_states_storage.ptref_nhem, num_of_iter = \
            self._compute_reference_state_wrapper(
                qgpv=self._interpolated_field_storage.qgpv,
                u=self._interpolated_field_storage.interpolated_u,
                theta=self._interpolated_field_storage.interpolated_theta)

        if num_of_iter >= self.maxit:
            raise ValueError("The reference state does not converge for Northern Hemisphere.")

        # === Compute reference states in Southern Hemisphere ===
        if not self.northern_hemisphere_results_only:
            self._reference_states_storage.qref_shem, \
                self._reference_states_storage.uref_shem, \
                self._reference_states_storage.ptref_shem, num_of_iter = \
                self._compute_reference_state_wrapper(
                    qgpv=-self._interpolated_field_storage.qgpv[:, ::-1, :],
                    u=self._interpolated_field_storage.interpolated_u[:, ::-1, :],
                    theta=self._interpolated_field_storage.interpolated_theta[:, ::-1, :])

            if num_of_iter >= self.maxit:
                raise ValueError("The reference state does not converge for Southern Hemisphere.")

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
            self._domain_average_storage.static_stability,
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

    def _compute_intermediate_flux_terms(self):
        """
        The flux term computation from NH18 is currently shared by both interface.
        .. versionadded:: 0.7.0
        """
        # === Compute barotropic flux terms (NHem) ===
        self._lwa_storage.lwa_nhem, \
            self._barotropic_flux_terms_storage.lwa_baro_nhem, \
            self._barotropic_flux_terms_storage.ua1baro_nhem, \
            self._barotropic_flux_terms_storage.u_baro_nhem, \
            self._barotropic_flux_terms_storage.ua2baro_nhem, \
            self._barotropic_flux_terms_storage.ep1baro_nhem, \
            self._barotropic_flux_terms_storage.ep2baro_nhem, \
            self._barotropic_flux_terms_storage.ep3baro_nhem, \
            self._barotropic_flux_terms_storage.ep4_nhem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                self._interpolated_field_storage.qgpv,
                self._interpolated_field_storage.interpolated_u,
                self._interpolated_field_storage.interpolated_v,
                self._interpolated_field_storage.interpolated_theta,
                self._reference_states_storage.qref_nhem,
                self._reference_states_storage.uref_nhem,
                self._reference_states_storage.ptref_nhem)

        # === Compute barotropic flux terms (SHem) ===
        # TODO: check signs!
        if not self.northern_hemisphere_results_only:
            self._lwa_storage.lwa_shem, \
                self._barotropic_flux_terms_storage.lwa_baro_shem, \
                self._barotropic_flux_terms_storage.ua1baro_shem, \
                self._barotropic_flux_terms_storage.u_baro_shem, \
                self._barotropic_flux_terms_storage.ua2baro_shem, \
                self._barotropic_flux_terms_storage.ep1baro_shem, \
                self._barotropic_flux_terms_storage.ep2baro_shem, \
                self._barotropic_flux_terms_storage.ep3baro_shem, \
                ep4_shem = \
                self._compute_lwa_and_barotropic_fluxes_wrapper(
                    -self._interpolated_field_storage.qgpv[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_u[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_v[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_theta[:, ::-1, :],
                    self._reference_states_storage.qref_shem[::-1, :],
                    self._reference_states_storage.uref_shem[::-1, :],
                    self._reference_states_storage.ptref_shem[::-1, :])
            self._barotropic_flux_terms_storage.ep4_shem = -ep4_shem

    @property
    def static_stability(self) -> np.array:
        """
        The interpolated static stability.
        """
        return self._domain_average_storage.static_stability


class QGField(QGFieldNH18):
    """
    This class is equivalent to `QGFieldNH18` for backward compatibility.
    `QGField` will be deprecated in upcoming release. See documentation in `QGFieldNH18`.
    """


class QGFieldNHN22(QGFieldBase):
    """
    Procedures and reference state computation with the set of boundary conditions of NHN22:

        Neal et al (2022). The 2021 Pacific Northwest heat wave and associated blocking: meteorology and the role of an
        upstream cyclone as a diabatic source of wave activity.
        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL097699

    Note that barotropic flux term computation from this class occasionally experience numerical instability, so
    please use with caution.

    See the documentation of :py:class:`QGField` for the public interface.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    eq_boundary_index: int, optional
        The improved inversion algorithm of reference states allow modification of equatorward boundary
        to be the absolute vorticity. This parameter specify the location of grid point (from equator)
        which will be used as boundary. The results in NHN22 is produced by using 1 deg latitude data and
        eq_boundary_index = 5, i.e. using a latitude domain from 5 deg to the pole. Default = 5 here.

    Examples
    --------
    Notebook: :doc:`notebooks/nhn22_reference_states`
    """
    def __init__(self, xlon, ylat, plev, u_field, v_field, t_field, kmax=49, maxit=100000, dz=1000., npart=None,
                 tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS,
                 northern_hemisphere_results_only=False, eq_boundary_index=5):
        super().__init__(xlon, ylat, plev, u_field, v_field, t_field, kmax, maxit, dz, npart, tol, rjac, scale_height,
                         cp, dry_gas_constant, omega, planet_radius, northern_hemisphere_results_only)

        # === Latitude domain boundary ===
        self._eq_boundary_index = eq_boundary_index
        self._jd = self.nlat // 2 + self.nlat % 2 - self.eq_boundary_index

    def _interpolate_fields(self, Interpolated_fields_to_return, return_named_tuple) -> Optional[NamedTuple]:
        """
        .. versionadded:: 0.7.0
        """
        self._interpolated_field_storage.qgpv, \
            self._interpolated_field_storage.interpolated_u, \
            self._interpolated_field_storage.interpolated_v, \
            self._interpolated_field_storage.interpolated_avort, \
            self._interpolated_field_storage.interpolated_theta, \
            self._domain_average_storage.static_stability_n, \
            self._domain_average_storage.static_stability_s, \
            self._domain_average_storage.tn0, self._domain_average_storage.ts0 = interpolate_fields_direct_inv(  # f2py module
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

        if return_named_tuple:
            interpolated_fields = Interpolated_fields_to_return(
                self.qgpv,
                self.interpolated_u,
                self.interpolated_v,
                self.interpolated_theta,
                (self._domain_average_storage.static_stability_s, self._domain_average_storage.static_stability_n))
            return interpolated_fields

    def _compute_reference_states(self):
        """
        Added for NHN 2022 GRL

        .. versionadded:: 0.6.0
        """

        # === Compute reference states in Northern Hemisphere ===
        self._reference_states_storage.qref_nhem, \
            self._reference_states_storage.uref_nhem, \
            self._reference_states_storage.ptref_nhem, \
                fawa, ubar, tbar = \
            self._compute_reference_states_nhn22_hemispheric_wrapper(
                qgpv=self._interpolated_field_storage.qgpv,
                u=self._interpolated_field_storage.interpolated_u,
                avort=self._interpolated_field_storage.interpolated_avort,
                theta=self._interpolated_field_storage.interpolated_theta,
                t0=self._domain_average_storage.tn0)

        if not self.northern_hemisphere_results_only:
            # === Compute reference states in Southern Hemisphere ===
            self._reference_states_storage.qref_shem, \
                self._reference_states_storage.uref_shem, \
                self._reference_states_storage.ptref_shem, \
                fawa, ubar, tbar = \
                self._compute_reference_states_nhn22_hemispheric_wrapper(
                    qgpv=-self._interpolated_field_storage.qgpv[:, ::-1, :],
                    u=self._interpolated_field_storage.interpolated_u[:, ::-1, :],
                    avort=self._interpolated_field_storage.interpolated_avort[:, ::-1, :],
                    theta=self._interpolated_field_storage.interpolated_theta[:, ::-1, :],
                    t0=self._domain_average_storage.ts0)

    def _compute_reference_states_nhn22_hemispheric_wrapper(self, qgpv, u, avort, theta, t0):
        """
        Wrapper to a series of operation using direct inversion algorithm to solve reference state.
        """
        qref_over_sin, ubar, tbar, fawa, ckref, tjk, sjk = compute_qref_and_fawa_first(
            pv=qgpv,
            uu=u,
            vort=avort,
            pt=theta,
            tn0=t0,
            nd=self.nlat//2 + self.nlat % 2,  # 91
            nnd=self.nlat,                    # 181
            jb=self.eq_boundary_index,        # 5
            jd=self.jd,
            a=self.planet_radius,
            omega=self.omega,
            dz=self.dz,
            h=self.scale_height,
            rr=self.dry_gas_constant,
            cp=self.cp)

        self._check_nan("qref_over_sin", qref_over_sin)
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
                jd=self.jd,
                z=np.arange(0, self.kmax*self.dz, self.dz),
                statn=self._domain_average_storage.static_stability_n,
                qref=qref_over_sin,
                ckref=ckref,
                sjk=sjk,
                a=self.planet_radius,
                om=self.omega,
                dz=self.dz,
                h=self.scale_height,
                rr=self.dry_gas_constant,
                cp=self.cp)
            qjj, djj, cjj, rj = ans

            # TODO: The inversion algorithm  is the bottleneck of the computation
            # SciPy is very slow compared to MKL in Fortran...
            lu, piv, info = dgetrf(qjj)
            qjj, info = dgetri(lu, piv)

            _ = matrix_after_inversion(
                k=k,
                qjj=qjj,
                djj=djj,
                cjj=cjj,
                rj=rj,
                sjk=sjk,
                tjk=tjk)

        tref, qref, uref = upward_sweep(
            jmax=self.nlat,
            jb=self.eq_boundary_index,
            sjk=sjk,
            tjk=tjk,
            ckref=ckref,
            tb=self._domain_average_storage.tn0,
            qref_over_cor=qref_over_sin,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            h=self.scale_height,
            rr=self.dry_gas_constant,
            cp=self.cp)

        # return qref, uref, tref, fawa, ubar, tbar
        return qref_over_sin / (2. * self.omega), uref, tref, fawa, ubar, tbar

    @property
    def static_stability(self) -> Tuple[np.array, np.array]:
        """
        The interpolated static stability.
        """
        if self.northern_hemisphere_results_only:
            return self._domain_average_storage.static_stability_n
        else:
            return self._domain_average_storage.static_stability_s, self._domain_average_storage.static_stability_n

    @property
    def eq_boundary_index(self):
        return self._eq_boundary_index

    @property
    def jd(self):
        return self._jd

    def _compute_intermediate_flux_terms(self):
        """
        Intermediate flux term computation for NHN 2022 GRL. Note that numerical instability is observed occasionally,
        so please used with caution.

        .. versionadded:: 0.7.0
        """

        # Turn qref back to correct unit

        ylat_input = self.ylat[-self.equator_idx:] if self.northern_hemisphere_results_only else self.ylat
        qref_correct_unit = self._reference_states_storage.qref_correct_unit(
            ylat=ylat_input, omega=self.omega, python_indexing=False)

        # === Compute barotropic flux terms (NHem) ===
        self._barotropic_flux_terms_storage.lwa_baro_nhem, \
            self._barotropic_flux_terms_storage.u_baro_nhem, \
            urefbaro, \
            self._barotropic_flux_terms_storage.ua1baro_nhem, \
            self._barotropic_flux_terms_storage.ua2baro_nhem, \
            self._barotropic_flux_terms_storage.ep1baro_nhem, \
            self._barotropic_flux_terms_storage.ep2baro_nhem, \
            self._barotropic_flux_terms_storage.ep3baro_nhem, \
            self._barotropic_flux_terms_storage.ep4_nhem, \
            astar1, \
            astar2 = \
            compute_flux_dirinv_nshem(
                pv=self._interpolated_field_storage.qgpv,
                uu=self._interpolated_field_storage.interpolated_u,
                vv=self._interpolated_field_storage.interpolated_v,
                pt=self._interpolated_field_storage.interpolated_theta,
                tn0=self._domain_average_storage.tn0,
                qref=qref_correct_unit[-self.equator_idx:],
                uref=self._reference_states_storage.uref_nhem,
                tref=self._reference_states_storage.ptref_nhem,
                jb=self.eq_boundary_index,
                is_nhem=True,
                a=self.planet_radius,
                om=self.omega,
                dz=self.dz,
                h=self.scale_height,
                rr=self.dry_gas_constant,
                cp=self.cp,
                prefac=self.prefactor)
        self._lwa_storage.lwa_nhem = np.abs(astar1 + astar2)

        # === Compute barotropic flux terms (SHem) ===
        # TODO: check signs!
        if not self.northern_hemisphere_results_only:
            self._barotropic_flux_terms_storage.lwa_baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.u_baro[:, :self.equator_idx], \
                urefbaro, \
                self._barotropic_flux_terms_storage.ua1baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.ua2baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.ep1baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.ep2baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.ep3baro[:, :self.equator_idx], \
                self._barotropic_flux_terms_storage.ep4[:, :self.equator_idx], \
                astar1, \
                astar2 = \
                compute_flux_dirinv_nshem(
                    pv=self._interpolated_field_storage.qgpv,
                    uu=self._interpolated_field_storage.interpolated_u,
                    vv=self._interpolated_field_storage.interpolated_v,
                    pt=self._interpolated_field_storage.interpolated_theta,
                    tn0=self._domain_average_storage.ts0,
                    qref=qref_correct_unit[:self.equator_idx],
                    uref=self._reference_states_storage.uref_shem,
                    tref=self._reference_states_storage.ptref_shem,
                    jb=self.eq_boundary_index,
                    is_nhem=False,
                    a=self.planet_radius,
                    om=self.omega,
                    dz=self.dz,
                    h=self.scale_height,
                    rr=self.dry_gas_constant,
                    cp=self.cp,
                    prefac=self.prefactor)
            self._lwa_storage.lwa[:, :self.equator_idx, :] = np.abs(astar1 + astar2)

    def _compute_lwa_flux_dirinv(self, qref, uref, tref):
        """
        Added for NHN 2022 GRL

        .. versionadded:: 0.6.0
        """
        ans = compute_flux_dirinv_nshem(
            pv=self._interpolated_field_storage.qgpv,
            uu=self._interpolated_field_storage.interpolated_u,
            vv=self._interpolated_field_storage.interpolated_v,
            pt=self._interpolated_field_storage.interpolated_theta,
            tn0=self._domain_average_storage.tn0,
            qref=qref,
            uref=uref,
            tref=tref,
            jb=self.eq_boundary_index,
            is_nhem=True,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz, h=self.scale_height, rr=self.dry_gas_constant, cp=self.cp, prefac=self.prefactor)
        # astarbaro, u_baro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4baro, astar1, astar2 = ans
        return ans
