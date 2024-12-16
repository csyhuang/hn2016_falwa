"""
------------------------------------------
File name: oopinterface.py
Author: Clare Huang
"""
from typing import Tuple, Optional, Union, NamedTuple, Type
from abc import ABC, abstractmethod
import math
import warnings
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.linalg.lapack import dgetrf, dgetri

from falwa import utilities
from falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA
from falwa.data_storage import InterpolatedFieldsStorage, DomainAverageStorage, ReferenceStatesStorage, \
    LWAStorage, BarotropicFluxTermsStorage, OutputBarotropicFluxTermsStorage

# *** Import f2py modules ***
from falwa import compute_qgpv, compute_qgpv_direct_inv, compute_qref_and_fawa_first, \
    matrix_b4_inversion, matrix_after_inversion, upward_sweep, compute_flux_dirinv_nshem, compute_reference_states, \
    compute_lwa_and_layerwise_fluxes, compute_lwa_only_nhn22
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
    data_on_evenly_spaced_pseudoheight_grid : bool, optional
        whether the input data sits on an evenly spaced pseudoheight grid. Default = False
        If Ture, the method interpolate_fields (i.e. vertical interpolation) would not do vertical interpolation,
        but only calculate potential temperature, QGPV and static stability. New in version 1.3.
    raise_error_for_nonconvergence : bool, optional
        If Uref solver does not have a solution, with this set to False, user can call compute_lwa_only after calling
        computer_reference_states. If set to True, an error is raised and the computation would halt.
    """

    def __init__(self, xlon, ylat, plev, u_field, v_field, t_field, kmax=49, maxit=100000, dz=1000., npart=None,
                 tol=1.e-5, rjac=0.95, scale_height=SCALE_HEIGHT, cp=CP, dry_gas_constant=DRY_GAS_CONSTANT,
                 omega=EARTH_OMEGA, planet_radius=EARTH_RADIUS, northern_hemisphere_results_only=False,
                 data_on_evenly_spaced_pseudoheight_grid=False, raise_error_for_nonconvergence=True):

        """
        Create a QGField object.
        This only initialize the attributes of the object. Analysis and
        computation are done by calling various methods.
        """

        # === Warning setting ===
        self._raise_error_for_nonconvergence: bool = raise_error_for_nonconvergence
        self._nonconvergent_uref: bool = False  # This will be set to True once Uref is computed

        # === Variables related to the Vertical grid ===
        self._data_on_evenly_spaced_pseudoheight_grid = data_on_evenly_spaced_pseudoheight_grid
        if self._data_on_evenly_spaced_pseudoheight_grid:
            self.plev = plev
            self.kmax = plev.size
            self._plev_to_height = -scale_height * np.log(plev / P_GROUND)
            self.height = -scale_height * np.log(plev / P_GROUND)
            self.dz = np.diff(self.height)[0]
        else:
            # === Check the validity of plev ===
            self._check_valid_plev(plev, scale_height, kmax, dz)  # initialized self.plev and self._plev_to_height
            self.kmax = kmax
            self._plev_to_height = -scale_height * np.log(plev / P_GROUND)
            self.height = np.array([i * dz for i in range(self.kmax)])
            self.dz = dz

        # === Check whether the input field is masked array. If so, turn them to normal array ===
        u_field = self._convert_masked_data(u_field, "u_field")
        v_field = self._convert_masked_data(v_field, "v_field")
        theta_field = self._convert_masked_data(t_field, "t_field") * np.exp(
            dry_gas_constant / cp * self._plev_to_height[:, np.newaxis, np.newaxis] / scale_height)

        # === Check if ylat is in ascending order and include the equator ===
        ylat = self._convert_masked_data(ylat, "ylat")
        self._input_ylat, self.need_latitude_interpolation, self._ylat, self.equator_idx, self._clat = \
            self._check_and_flip_ylat(ylat)

        # === Initialize longitude grid ===
        self.xlon = xlon

        # === Check the shape of wind/temperature fields ===
        self.nlev = plev.size
        self.nlat = ylat.size
        self.nlon = xlon.size
        expected_dimension = (self.nlev, self.nlat, self.nlon)
        self._check_dimension_of_fields(field=u_field, field_name='u_field', expected_dim=expected_dimension)
        self._check_dimension_of_fields(field=v_field, field_name='v_field', expected_dim=expected_dimension)
        self._check_dimension_of_fields(field=theta_field, field_name='theta_field', expected_dim=expected_dimension)

        # === Do Interpolation on latitude grid if needed ===
        if self.need_latitude_interpolation:
            interp_u = interp1d(self._input_ylat, u_field, axis=1, fill_value="extrapolate")
            interp_v = interp1d(self._input_ylat, v_field, axis=1, fill_value="extrapolate")
            interp_t = interp1d(self._input_ylat, theta_field, axis=1, fill_value="extrapolate")
            self.u_field = interp_u(self._ylat)
            self.v_field = interp_v(self._ylat)
            self.theta_field = interp_t(self._ylat)
        else:
            self.u_field = u_field
            self.v_field = v_field
            self.theta_field = theta_field

        # === Coordinate-related ===
        self._nlat_analysis = self._ylat.size  # This is the number of latitude grid point used in analysis
        self._eq_boundary_index = 0  # Latitude domain boundary. Will be updated in QGFieldNHN22.__init__
        self._jd = self._nlat_analysis // 2 + self._nlat_analysis % 2 - self._eq_boundary_index
        self.dphi = np.deg2rad(180. / (self._nlat_analysis - 1))  # F90 code: dphi = pi/float(nlat-1)
        self.dlambda = np.deg2rad(360. / self.nlon)  # F90 code: dlambda = 2*pi/float(nlon)
        self.npart = npart if npart is not None else self._nlat_analysis

        # === Moved here in v0.7.0 ===
        self._northern_hemisphere_results_only = northern_hemisphere_results_only

        # === Other parameters ===
        self.maxit = maxit
        self.tol = tol
        self.rjac = rjac

        # === Constants ===
        self.scale_height = scale_height
        self.cp = cp
        self.dry_gas_constant = dry_gas_constant
        self.omega = omega
        self.planet_radius = planet_radius
        self._compute_prefactor()  # Compute normalization prefactor
        self._initialize_storage()  # Create storage instances to store variables

    def _initialize_storage(self):
        """
        Create storage instances to store output variables
        """
        # === qgpv, u, v, avort, theta encapsulated in InterpolatedFieldsStorage ===
        self._interpolated_field_storage = InterpolatedFieldsStorage(
            pydim=(self.kmax, self._nlat_analysis, self.nlon),
            fdim=(self.nlon, self._nlat_analysis, self.kmax),
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
        lat_dim = self.equator_idx if self.northern_hemisphere_results_only else self._nlat_analysis
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

    def _compute_static_stability_func(self):
        """
        Private function to compute hemispheric static stability from input pressure grids.
        TODO: add more description

        Returns
        -------
        """
        # Total area
        csm = self._clat[:self._jd].sum()

        # (Hemispheric) global potential temperature mean per pressure level
        t0_s = np.mean(self.theta_field[:, :self._jd, :] * self._clat[np.newaxis, :self._jd, np.newaxis], axis=-1)\
            .sum(axis=-1) / csm  # SHem
        t0_n = np.mean(self.theta_field[:, -self._jd:, :] * self._clat[np.newaxis, -self._jd:, np.newaxis], axis=-1)\
            .sum(axis=-1) / csm  # NHem

        # Create an interpolation function
        uni_spline_s = UnivariateSpline(x=self._plev_to_height, y=t0_s)
        uni_spline_n = UnivariateSpline(x=self._plev_to_height, y=t0_n)
        return uni_spline_s, uni_spline_n, uni_spline_s.derivative(), uni_spline_n.derivative()

    def _compute_prefactor(self):
        """
        Private function. Compute prefactor for normalization by evaluating
            int^{z=kmax*dz}_0 e^{-z/H} dz
        using rectangular rule consistent with the integral evaluation in compute_lwa_and_barotropic_fluxes.f90.
        TODO: evaluate numerical integration scheme used in the fortran module.
        """
        self._prefactor = sum([math.exp(-k * self.dz / self.scale_height) * self.dz for k in range(1, self.kmax - 1)])

    @staticmethod
    def _convert_masked_data(variable: np.ndarray, varname: str):
        if np.ma.is_masked(variable) or isinstance(variable, np.ma.core.MaskedArray):
            warnings.warn(
                '{var} is a masked array of dimension {dim} with {num} masked elements and fill value {fill}. '
                .format(var=varname, dim=variable.shape, num=variable.mask.sum(), fill=variable.fill_value))
            variable = variable.data
        return variable

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
        self._plev_to_height = -scale_height * np.log(plev / P_GROUND)

        # Check if kmax is valid given the max pseudoheight in the input data
        hmax = -scale_height * np.log(plev[-1] / P_GROUND)
        if hmax < (kmax - 1) * dz:
            raise ValueError('Input kmax = {} but the maximum valid kmax'.format(kmax) +
                             '(constrainted by the vertical grid of your input data) is {}'.format(int(hmax // dz) + 1))

    @staticmethod
    def _check_and_flip_ylat(ylat):
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
        # Save ylat input by user first
        _input_ylat = ylat
        if (ylat.size % 2 == 0) & (sum(ylat == 0.0) == 0):
            # Even grid
            need_latitude_interpolation = True
            _ylat = np.linspace(-90., 90., ylat.size + 1, endpoint=True)
            equator_idx = \
                np.argwhere(_ylat == 0)[0][0] + 1
            # Fortran indexing starts from 1
        elif sum(ylat == 0) == 1:
            # Odd grid
            need_latitude_interpolation = False
            _ylat = ylat
            equator_idx = np.argwhere(ylat == 0)[0][0] + 1  # Fortran indexing starts from 1
        else:
            raise TypeError(
                "There are more than 1 grid point with latitude 0."
            )
        _clat = np.abs(np.cos(np.deg2rad(_ylat)))
        return _input_ylat, need_latitude_interpolation, _ylat, equator_idx, _clat

    @property
    def ylat(self):
        """
        This is ylat grid input by user.
        """
        return self._input_ylat

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
        if self._input_ylat is None:
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
                    variable, self._ylat[-(self._nlat_analysis // 2 + 1):],
                    self._input_ylat[-(self.nlat // 2):],
                    which_axis=interp_axis)
            else:
                return self._interp_back(variable, self._ylat, self._input_ylat, which_axis=interp_axis)
        else:
            return variable

    def _vertical_average(self, var_3d, lowest_layer_index=1, height_axis=-1):
        dc = self.dz / self.prefactor
        var_baro = np.sum(
            var_3d[:, :, lowest_layer_index:]
            * np.exp(-self.height[np.newaxis, np.newaxis, lowest_layer_index:] / self.scale_height) * dc,
            axis=height_axis)
        return var_baro

    def compute_layerwise_lwa_fluxes(self, qgpv, u, v, theta, ncforce, qref_temp, uref_temp, ptref_temp):
        """
        Private function. Compute layerwise lwa flux in Fortran except the stretching term, which will be calculated in
        python.
        Shall be in parallel with compute_lwa_and_barotropic_fluxes.
        Now the input are in fortran indices...

        Parameters
        ----------
        qgpv
        u
        v
        theta
        ncforce
        qref_temp
        uref_temp
        ptref_temp

        Returns
        -------

        """
        astar, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4 = compute_lwa_and_layerwise_fluxes(
            pv=qgpv,
            uu=u,
            vv=v,
            pt=theta,
            ncforce=ncforce,
            qref=qref_temp,
            uref=uref_temp,
            tref=ptref_temp,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            h=self.scale_height,
            r=self.dry_gas_constant,
            cp=self.cp,
            prefactor=self.prefactor)

        # *** Compute stretching term layerwise ***

    @abstractmethod
    def _compute_lwa_and_barotropic_fluxes_wrapper(self, *args):
        """
        Private function. It computes layerwise fluxes (via F2PY modules) first, and then do vertical integration.
        Returns variable in the following order:
            astar, astar_baro, ua1_baro, u_baro, ua2_baro, ep1_baro, ep2_baro, ep3_baro, ep4, ncforce_baro
        """

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
        # === Computed static stability function d(\tilde{theta})/dz (z) ===
        t0_s_func, t0_n_func, static_stability_func_s, static_stability_func_n = self._compute_static_stability_func()

        # === Interpolate onto evenly spaced pseudoheight grid ===
        if self._data_on_evenly_spaced_pseudoheight_grid:
            print("No need to do interpolation. Directly initialize")
            interpolated_u = self.u_field
            interpolated_v = self.v_field
            interpolated_theta = self.theta_field
        else:
            print("Do scipy interpolation")
            interpolated_u = self._vertical_interpolation(self.u_field, kind="linear", axis=0)
            interpolated_v = self._vertical_interpolation(self.v_field, kind="linear", axis=0)
            interpolated_theta = self._vertical_interpolation(self.theta_field, kind="linear", axis=0)
        self._interpolated_field_storage.interpolated_u = np.swapaxes(interpolated_u, 0, 2)
        self._interpolated_field_storage.interpolated_v = np.swapaxes(interpolated_v, 0, 2)
        self._interpolated_field_storage.interpolated_theta = np.swapaxes(interpolated_theta, 0, 2)

        # Return a named tuple
        interpolated_fields_to_return: Type[namedtuple] = namedtuple(
            'Interpolated_fields', ['QGPV', 'U', 'V', 'Theta', 'Static_stability'])

        interpolated_fields_tuple = self._compute_qgpv(
            interpolated_fields_to_return, return_named_tuple,
            t0_s=t0_s_func(self.height), t0_n=t0_n_func(self.height),
            stat_s=static_stability_func_s(self.height), stat_n=static_stability_func_n(self.height))

        # TODO: warn that for NHN22, static stability returned would be a tuple of ndarray
        if return_named_tuple:
            return interpolated_fields_tuple

    def _vertical_interpolation(self, variable, kind, axis=0):
        return interp1d(
            self._plev_to_height, variable, axis=axis, bounds_error=False, kind=kind, fill_value='extrapolate')(
            self.height)

    @abstractmethod
    def _compute_qgpv(
            self, interpolated_fields_to_return: NamedTuple, return_named_tuple: bool,
            t0_n: np.ndarray, t0_s: np.ndarray, stat_n: np.ndarray, stat_s: np.ndarray) -> Optional[NamedTuple]:
        """
        The specific interpolation procedures w.r.t the particular procedures in the paper will be implemented here.
        """

    def compute_reference_states(self, return_named_tuple: bool = True, northern_hemisphere_results_only=None) -> \
    Optional[NamedTuple]:

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

        # === Return a named tuple ===
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

    def compute_lwa_only(self) -> None:
        """
        New in version 1.3. After calling compute_reference_state, if the reference state Uref was not solved properly,
        this method would compute LWA (and its barotropic component) based on `QGPV` and `Qref` obtained from previous
        step.

        Note with caution that, the implementation of this method in `QGFieldNH18` behaves slightly differently from
        `compute_lwa_and_barotropic_fluxes` in the Southern Hemisphere (while it produces the same results for the
        Northern Hemisphere) probably due to indexing difference.

        To retrieve LWA and barotropic fluxes computed:
        --------

        >>> QGField.compute_lwa_only()
        >>> lwa_baro = QGField.lwa_baro  # barotropic LWA
        >>> u_baro = QGField.u_baro    # barotropic U
        >>> lwa = QGField.lwa       # 3-D LWA
        """

        ylat_input = self._ylat[-self.equator_idx:] if self.northern_hemisphere_results_only else self._ylat
        qref_correct_unit = self._reference_states_storage.qref_correct_unit(
            ylat=ylat_input, omega=self.omega, python_indexing=False)

        # === Compute barotropic flux terms (NHem) ===
        self._barotropic_flux_terms_storage.lwa_baro_nhem, \
            self._barotropic_flux_terms_storage.u_baro_nhem, \
            astar1, \
            astar2 = \
            compute_lwa_only_nhn22(
                pv=self._interpolated_field_storage.qgpv,
                uu=self._interpolated_field_storage.interpolated_u,
                qref=qref_correct_unit[-self.equator_idx:],
                jb=self._eq_boundary_index,
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
                astar1, \
                astar2 = \
                compute_lwa_only_nhn22(
                    pv=self._interpolated_field_storage.qgpv,
                    uu=self._interpolated_field_storage.interpolated_u,
                    qref=qref_correct_unit[:self.equator_idx],
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

    def compute_lwa_and_barotropic_fluxes(
            self, return_named_tuple: bool = True, northern_hemisphere_results_only=None, ncforce=None):

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

        ncforce : optional, np.ndarray
           This is the diabatic term output from climate model interpolated on even-pseudoheight grid, i.e.,
           the integrand of equation (11) in Lubis et al. "Importance of Cloud-Radiative Effects in Wintertime
           Atmospheric Blocking over the Euro-Atlantic Sector" (in prep). The integrated barotropic component
           of ncforce is accessible via `QGField.ncforce_baro`.

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

        if self._nonconvergent_uref:
            raise ValueError(
                """
                Uref was not properly solved. This method cannot be called. 
                Try compute_lwa_only instead if you deem appropriate.
                """)

        # TODO: need a check for reference states computed. If not, throw an error.
        self._compute_intermediate_barotropic_flux_terms(ncforce=ncforce)

        # === Compute named fluxes in NH18 ===
        clat = self._clat[-self.equator_idx:] if self.northern_hemisphere_results_only else self._clat
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
        self._output_barotropic_flux_terms_storage.ncforce_baro = \
            self._barotropic_flux_terms_storage.fortran_to_python(self._barotropic_flux_terms_storage.ncforce_baro)

        # === Return the named tuple ===
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
    def _compute_intermediate_barotropic_flux_terms(self, ncforce=None):
        """
        Compute all the barotropic components stored in self._barotropic_flux_terms_storage
        (ua1, ua2, ep1, ep2, ep3, ep4) depending on which BC protocol to use
        """

    @staticmethod
    def _check_nan(name, var):
        nan_num = np.count_nonzero(np.isnan(var))
        if nan_num > 0:
            print(f"num of nan in {name}: {np.count_nonzero(np.isnan(var))}.")

    # === Fixed properties (since creation of instance) ===
    @property
    def prefactor(self):
        """Normalization constant for vertical weighted-averaged integration"""
        return self._prefactor

    @property
    def eq_boundary_index(self):
        return self._eq_boundary_index

    @property
    def ylat_ref_states(self) -> np.array:
        """
        This is the reference state grid output to user
        """
        if self.northern_hemisphere_results_only:
            if self.need_latitude_interpolation:
                return self._input_ylat[-(self.nlat // 2):]
            else:
                return self._input_ylat[-(self.nlat // 2 + 1):]
        return self._input_ylat

    @property
    def ylat_ref_states_analysis(self) -> np.array:
        """
        Latitude dimension of reference state.
        This is input to ReferenceStatesStorage.qref_correct_unit.
        """
        if self.northern_hemisphere_results_only:
            return self._ylat[-(self._nlat_analysis // 2 + 1):]
        return self._ylat

    @property
    def northern_hemisphere_results_only(self) -> bool:
        """
        Even though a global field is required for input, whether ref state and fluxes are computed for
        northern hemisphere only
        """
        return self._northern_hemisphere_results_only

    # === Boolean related to the state of the input field ===
    @property
    def nonconvergent_uref(self) -> bool:
        """
        If True, `QGField.compute_lwa_and_barotropic_flux` cannot be called. If user deems appropriate to proceed with LWA calculation, call `QGField.compute_lwa_only` instead.

        Returns
        -------
        A boolean. Initial value is False. After calling QGField.compute_reference_state, if Uref cannot be solved, its value will be changed to True.
        """
        return self._nonconvergent_uref

    # === Derived physical quantities ===
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
                self.ylat_ref_states_analysis, self.omega), interp_axis=1)

    @property
    def uref(self):
        """
        Reference state of zonal wind (Uref).
        """
        if self._reference_states_storage.uref is None:
            raise ValueError('uref is not computed yet.')
        return self._return_interp_variables(
            variable=self._reference_states_storage.fortran_to_python(self._reference_states_storage.uref),
            interp_axis=1)

    @property
    def ptref(self):
        """
        Reference state of potential temperature (\\Theta_ref).
        """
        if self._reference_states_storage.ptref is None:
            raise ValueError('ptref is not computed yet.')
        return self._return_interp_variables(
            variable=self._reference_states_storage.fortran_to_python(self._reference_states_storage.ptref),
            interp_axis=1)

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
    def ncforce_baro(self):
        """
        If input `ncforce` of method compute_lwa_and_barotropic_fluxes is not None, this would return the
        corresponding barotropic component of non-conservative force contribution with respect to the wave
        activity budget equation in Lubis et al. Eq.(11).
        """
        if self._barotropic_flux_terms_storage.ncforce_baro is None:
            raise ValueError('ncforce_baro is not computed yet.')
        return self._return_interp_variables(
            variable=self._barotropic_flux_terms_storage.fortran_to_python(
                self._barotropic_flux_terms_storage.ncforce_baro),
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
        return self._input_ylat.size


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

    def _compute_qgpv(self, interpolated_fields_to_return, return_named_tuple, t0_n, t0_s, stat_n, stat_s) -> Optional[
        NamedTuple]:
        """
        .. versionadded:: 1.3.0
        """
        self._domain_average_storage.static_stability = 0.5 * (stat_s + stat_n)
        self._interpolated_field_storage.qgpv, \
            self._interpolated_field_storage.interpolated_avort = compute_qgpv(  # f2py module
            self._interpolated_field_storage.interpolated_u,
            self._interpolated_field_storage.interpolated_v,
            self._interpolated_field_storage.interpolated_theta,
            self.height,
            0.5 * (t0_s + t0_n),
            0.5 * (stat_s + stat_n),
            self.planet_radius,
            self.omega,
            self.dz,
            self.scale_height,
            self.dry_gas_constant,
            self.cp)
        if return_named_tuple:
            interpolated_fields = interpolated_fields_to_return(
                self.qgpv,
                self.interpolated_u,
                self.interpolated_v,
                self.interpolated_theta,
                self._domain_average_storage.static_stability)
            return interpolated_fields

    def _nonconvergence_notification(self, hemisphere: str = "Northern"):
        self._nonconvergent_uref = True
        display_str = f"The reference state does not converge for {hemisphere} Hemisphere."
        if self._raise_error_for_nonconvergence:
            raise ValueError(display_str)
        else:  # Issue just warning
            warnings.warn(display_str)

    def _compute_reference_states(self):
        """
        .. versionadded:: 0.7.0
        """
        # === Compute reference states in Northern Hemisphere using SOR ===
        self._reference_states_storage.qref_nhem, \
            self._reference_states_storage.uref_nhem, \
            self._reference_states_storage.ptref_nhem, num_of_iter = \
            self._compute_reference_state_wrapper(
                qgpv=self._interpolated_field_storage.qgpv,
                u=self._interpolated_field_storage.interpolated_u,
                theta=self._interpolated_field_storage.interpolated_theta)

        if num_of_iter >= self.maxit:
            self._nonconvergence_notification(hemisphere="Northern")

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
                self._nonconvergence_notification(hemisphere="Southern")

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
            pv=qgpv,
            uu=u,
            pt=theta,
            stat=self._domain_average_storage.static_stability,
            jd=self.equator_idx,
            npart=self.npart,
            maxits=self.maxit,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            eps=self.tol,
            h=self.scale_height,
            dphi=self.dphi,
            dlambda=self.dlambda,
            r=self.dry_gas_constant,
            cp=self.cp,
            rjac=self.rjac,
        )

    def _compute_lwa_and_barotropic_fluxes_wrapper(self, qgpv, u, v, theta, ncforce, qref_temp, uref_temp, ptref_temp):
        """
        Private function. It computes layerwise fluxes (via F2PY modules) first, and then do vertical integration.
        Returns variable in the following order:
            astar, astar_baro, ua1_baro, u_baro, ua2_baro, ep1_baro, ep2_baro, ep3_baro, ep4, ncforce_baro
        """
        astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4 = compute_lwa_and_layerwise_fluxes(
            pv=qgpv,
            uu=u,
            vv=v,
            pt=theta,
            ncforce=ncforce,
            qref=qref_temp,
            uref=uref_temp,
            tref=ptref_temp,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            h=self.scale_height,
            r=self.dry_gas_constant,
            cp=self.cp,
            prefactor=self.prefactor)
        astar = astar1 + astar2
        jd = uref_temp.shape[0]
        astar_baro = self._vertical_average(astar, lowest_layer_index=1)
        ua1_baro = self._vertical_average(ua1, lowest_layer_index=1)
        u_baro = self._vertical_average(u[:, -jd:, :], lowest_layer_index=1)
        ua2_baro = self._vertical_average(ua2, lowest_layer_index=1)
        ep1_baro = self._vertical_average(ep1, lowest_layer_index=1)
        ep2_baro = self._vertical_average(ep2, lowest_layer_index=1)
        ep3_baro = self._vertical_average(ep3, lowest_layer_index=1)
        ncforce_baro = self._vertical_average(ncforce3d, lowest_layer_index=1)
        return astar, astar_baro, ua1_baro, u_baro, ua2_baro, ep1_baro, ep2_baro, ep3_baro, ep4, ncforce_baro

    def _compute_intermediate_barotropic_flux_terms(self, ncforce=None):
        """
        The flux term computation from NH18 is currently shared by both interface.
        .. versionadded:: 0.7.0
        """
        if ncforce is None:
            ncforce = np.zeros_like(self._interpolated_field_storage.interpolated_theta)  # fortran indexing
        else:  # There is input
            ncforce = np.swapaxes(ncforce, 0, 2)
            assert ncforce.shape == self._interpolated_field_storage.interpolated_theta.shape

        # === Compute barotropic flux terms (NHem) ===
        self._lwa_storage.lwa_nhem, \
            self._barotropic_flux_terms_storage.lwa_baro_nhem, \
            self._barotropic_flux_terms_storage.ua1baro_nhem, \
            self._barotropic_flux_terms_storage.u_baro_nhem, \
            self._barotropic_flux_terms_storage.ua2baro_nhem, \
            self._barotropic_flux_terms_storage.ep1baro_nhem, \
            self._barotropic_flux_terms_storage.ep2baro_nhem, \
            self._barotropic_flux_terms_storage.ep3baro_nhem, \
            self._barotropic_flux_terms_storage.ep4_nhem, \
            self._barotropic_flux_terms_storage.ncforce_nhem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                self._interpolated_field_storage.qgpv,
                self._interpolated_field_storage.interpolated_u,
                self._interpolated_field_storage.interpolated_v,
                self._interpolated_field_storage.interpolated_theta,
                ncforce,
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
                ep4_shem, \
                self._barotropic_flux_terms_storage.ncforce_shem = \
                self._compute_lwa_and_barotropic_fluxes_wrapper(
                    -self._interpolated_field_storage.qgpv[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_u[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_v[:, ::-1, :],
                    self._interpolated_field_storage.interpolated_theta[:, ::-1, :],
                    -ncforce[:, ::-1, :],
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
                 northern_hemisphere_results_only=False, eq_boundary_index=5,
                 data_on_evenly_spaced_pseudoheight_grid=False):
        super().__init__(xlon, ylat, plev, u_field, v_field, t_field, kmax, maxit, dz, npart, tol, rjac, scale_height,
                         cp, dry_gas_constant, omega, planet_radius, northern_hemisphere_results_only,
                         data_on_evenly_spaced_pseudoheight_grid)

        # === Latitude domain boundary ===
        self._eq_boundary_index = eq_boundary_index
        self._jd = self._nlat_analysis // 2 + self._nlat_analysis % 2 - self.eq_boundary_index

    def _compute_qgpv(self, interpolated_fields_to_return, return_named_tuple, t0_s, t0_n, stat_s, stat_n) -> Optional[
        NamedTuple]:
        """
        .. versionadded:: 1.3.0
        """
        self._domain_average_storage.ts0 = t0_s
        self._domain_average_storage.tn0 = t0_n
        self._domain_average_storage.static_stability_s = stat_s
        self._domain_average_storage.static_stability_n = stat_n
        self._interpolated_field_storage.qgpv, \
            self._interpolated_field_storage.interpolated_avort = compute_qgpv_direct_inv(  # f2py module
            self.equator_idx,
            self._interpolated_field_storage.interpolated_u,
            self._interpolated_field_storage.interpolated_v,
            self._interpolated_field_storage.interpolated_theta,
            self.height,
            t0_s,
            t0_n,
            stat_s,
            stat_n,
            self.planet_radius,
            self.omega,
            self.dz,
            self.scale_height,
            self.dry_gas_constant,
            self.cp)

        if return_named_tuple:
            interpolated_fields = interpolated_fields_to_return(
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
            nd=self._nlat_analysis // 2 + self._nlat_analysis % 2,  # 91
            nnd=self._nlat_analysis,  # 181
            jb=self.eq_boundary_index,  # 5
            jd=self.jd,
            a=self.planet_radius,
            omega=self.omega,
            dz=self.dz,
            h=self.scale_height,
            dphi=self.dphi,
            dlambda=self.dlambda,
            rr=self.dry_gas_constant,
            cp=self.cp)

        self._check_nan("qref_over_sin", qref_over_sin)
        self._check_nan("ubar", ubar)
        self._check_nan("tbar", tbar)
        self._check_nan("fawa", fawa)
        self._check_nan("ckref", ckref)
        self._check_nan("tjk", tjk)
        self._check_nan("sjk", sjk)

        for k in range(self.kmax - 1, 1, -1):  # Fortran indices
            ans = matrix_b4_inversion(
                k=k,
                jmax=self._nlat_analysis,
                jb=self.eq_boundary_index,  # 5
                jd=self.jd,
                z=np.arange(0, self.kmax * self.dz, self.dz),
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
            jmax=self._nlat_analysis,
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
    def jd(self):
        return self._jd

    def _compute_lwa_and_barotropic_fluxes_wrapper(self, pv, uu, vv, pt, ncforce, tn0, qref, uref, tref, jb, is_nhem):
        astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4 = compute_flux_dirinv_nshem(
            pv=pv,
            uu=uu,
            vv=vv,
            pt=pt,
            ncforce=ncforce,
            tn0=tn0,
            qref=qref,
            uref=uref,
            tref=tref,
            jb=jb,
            is_nhem=is_nhem,
            a=self.planet_radius,
            om=self.omega,
            dz=self.dz,
            h=self.scale_height,
            rr=self.dry_gas_constant,
            cp=self.cp,
            prefac=self.prefactor)
        jd = uref.shape[0]
        astar1_baro = self._vertical_average(astar1, lowest_layer_index=1)
        astar2_baro = self._vertical_average(astar2, lowest_layer_index=1)
        astar_baro = astar1_baro + astar2_baro
        ua1_baro = self._vertical_average(ua1, lowest_layer_index=1)
        u_baro = self._vertical_average(uu[:, -jd:, :], lowest_layer_index=1)
        ua2_baro = self._vertical_average(ua2, lowest_layer_index=1)
        ep1_baro = self._vertical_average(ep1, lowest_layer_index=1)
        ep2_baro = self._vertical_average(ep2, lowest_layer_index=1)
        ep3_baro = self._vertical_average(ep3, lowest_layer_index=1)
        ncforce_baro = self._vertical_average(ncforce3d, lowest_layer_index=1)
        uref_baro = np.sum(
            uref[-jd:, 1:] * np.exp(-self.height[np.newaxis, 1:] / self.scale_height) * self.dz / self.prefactor,
            axis=-1)

        return astar_baro, u_baro, uref_baro, ua1_baro, ua2_baro, ep1_baro, ep2_baro, ep3_baro, ep4, \
            astar1, astar2, ncforce_baro

    def _compute_intermediate_barotropic_flux_terms(self, ncforce=None):
        """
        Intermediate flux term computation for NHN 2022 GRL. Note that numerical instability is observed occasionally,
        so please used with caution.

        Args:
            ncforce(numpy.ndarray, optional): non-conservative forcing already interpolated on regular grid
                of dimension (kmax, nlat, nlon)

        .. versionadded:: 0.7.0
        """

        # Turn qref back to correct unit

        ylat_input = self._ylat[-self.equator_idx:] if self.northern_hemisphere_results_only else self._ylat
        qref_correct_unit = self._reference_states_storage.qref_correct_unit(
            ylat=ylat_input, omega=self.omega, python_indexing=False)

        if ncforce is None:
            ncforce = np.zeros_like(self._interpolated_field_storage.interpolated_theta)  # fortran indexing
        else:  # There is input
            ncforce = np.swapaxes(ncforce, 0, 2)
            assert ncforce.shape == self._interpolated_field_storage.interpolated_theta.shape

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
            astar2, \
            self._barotropic_flux_terms_storage.ncforce_nhem = \
            self._compute_lwa_and_barotropic_fluxes_wrapper(
                pv=self._interpolated_field_storage.qgpv,
                uu=self._interpolated_field_storage.interpolated_u,
                vv=self._interpolated_field_storage.interpolated_v,
                pt=self._interpolated_field_storage.interpolated_theta,
                ncforce=ncforce,
                tn0=self._domain_average_storage.tn0,
                qref=qref_correct_unit[-self.equator_idx:],
                uref=self._reference_states_storage.uref_nhem,
                tref=self._reference_states_storage.ptref_nhem,
                jb=self.eq_boundary_index,
                is_nhem=True)
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
                astar2, \
                self._barotropic_flux_terms_storage.ncforce_baro[:, :self.equator_idx] = \
                self._compute_lwa_and_barotropic_fluxes_wrapper(
                    pv=self._interpolated_field_storage.qgpv,
                    uu=self._interpolated_field_storage.interpolated_u,
                    vv=self._interpolated_field_storage.interpolated_v,
                    pt=self._interpolated_field_storage.interpolated_theta,
                    ncforce=ncforce,
                    tn0=self._domain_average_storage.ts0,
                    qref=qref_correct_unit[:self.equator_idx],
                    uref=self._reference_states_storage.uref_shem,
                    tref=self._reference_states_storage.ptref_shem,
                    jb=self.eq_boundary_index,
                    is_nhem=False)
            self._lwa_storage.lwa[:, :self.equator_idx, :] = np.abs(astar1 + astar2)

    def _compute_lwa_flux_dirinv(self, qref, uref, tref):
        """
        Added for NHN 2022 GRL. Will deprecate soon.

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
