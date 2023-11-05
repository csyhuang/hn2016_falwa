"""
------------------------------------------
File name: xarrayinterface.py
Author: Christopher Polster
"""
import functools
import numpy as np
import xarray as xr

from falwa import __version__
from falwa.oopinterface import QGFieldNH18


def _is_ascending(arr):
    return np.all(np.diff(arr) > 0)

def _is_descending(arr):
    return np.all(np.diff(arr) < 0)

def _is_equator(x):
    return abs(x) < 1.0e-4

# Coordinate name lookup
_NAMES_PLEV = ["plev", "lev", "level", "isobaricInhPa"]
_NAMES_YLAT = ["ylat", "lat", "latitude"]
_NAMES_XLON = ["xlon", "lon", "longitude"]
_NAMES_TIME = ["time", "date", "datetime"]
# Wind and temperature name lookup
_NAMES_U = ["u", "U"]
_NAMES_V = ["v", "V"]
_NAMES_T = ["t", "T"]
# Budget terms name lookup
_NAMES_LWA  = ["lwa_baro"]
_NAMES_CZAF = ["convergence_zonal_advective_flux"]
_NAMES_DEMF = ["divergence_eddy_momentum_flux"]
_NAMES_MHF  = ["meridional_heat_flux"]


def _get_dataarray(data, names, user_names=None):
    name = _get_name(data, names, user_names=user_names)
    return data[name]


def _get_name(ds, names, user_names=None):
    # If the first name from the list of defaults is in the user-provided
    # dictionary, use the name provided there
    if user_names is not None and names[0] in user_names:
        name = user_names[names[0]]
        if name not in ds:
            raise KeyError(f"specified variable '{name}' not found")
        return name
    # Else, search in default list of names
    for name in names:
        if name in ds:
            return name
    raise KeyError(f"no matching variable for '{names[0]}' found")


class _MetadataServiceProvider:
    """Metadata services for the QGDataset

    The class provides metadata from its own registry and can be instanciated
    to provide additional metadata based on a template QGField object and
    user-provided information about additional non-core dimensions.

    Parameters
    ----------
    field : QGField
        Template QGField object to extract metadata from.
    other_coords : None | dict
        Mapping of dimension name to dimension coordinates of non-core
        dimensions. Entries must reflect order of dimensions.
    """

    def __init__(self, field, other_coords=None):
        self.field = field
        # Depend on dict to preserve ordering of dims (Python 3.7+)
        self.other_coords = dict(other_coords) if other_coords is not None else dict()

    @property
    def other_dims(self):
        """Names of non-core dimensions"""
        return tuple(self.other_coords.keys())

    @property
    def other_shape(self):
        """Shape of non-core dimensions"""
        return tuple(value.size for value in self.other_coords.values())

    @property
    def other_size(self):
        """Size of non-core dimensions"""
        return np.product(self.other_shape)

    # numpy convenience functions

    def shape(self, var):
        """Shape of a variable (non-core and core dims)"""
        shape = list(self.other_shape)
        # Get sizes of field dimensions from template field
        for name in self.info(var)["dim_names"]:
            shape.append(getattr(self.field, name).size)
        return tuple(shape)

    def flatten_other(self, arr):
        """Flatten the non-core dimensions of the array"""
        n = len(self.other_shape)
        assert arr.shape[:n] == self.other_shape, f"expected other shape of {self.other_shape}"
        return arr.reshape((self.other_size, *arr.shape[n:]))

    def restore_other(self, arr):
        """Un-flatten the non-core dimensions of the array"""
        assert arr.shape[0] == self.other_size, f"expected other size of {self.other_size}"
        return arr.reshape(self.other_shape + arr.shape[1:])

    # xarray convenience functions

    def dims(self, var):
        """Dimension names (non-core and core dims)"""
        return self.other_dims + self.info(var)["core_dims"]

    def coords(self, var):
        """Coordinate dictionary (non-core and core dims)"""
        coords = self.other_coords.copy()
        info = self.info(var)
        for dim, name in zip(info["core_dims"], info["dim_names"]):
            coords[dim] = getattr(self.field, name)
        return coords

    def as_dataarray(self, arr, var):
        """Create a DataArray from the input array as the given variable"""
        arr = np.asarray(arr)
        if arr.shape != self.shape(var):
            arr = self.restore_other(arr)
        assert arr.shape == self.shape(var)
        return xr.DataArray(
            arr,
            dims=self.dims(var),
            coords=self.coords(var),
            name=var,
            attrs=self.attrs(var)
        )

    def attrs(self, var=None):
        """Attributes for a Dataset (var=None) or a DataArray (var!=None)"""
        if var is not None:
            return self.info(var)["attrs"]
        return {
            "kmax": self.field.kmax,
            "dz": self.field.dz,
            "maxit": self.field.maxit,
            "tol": self.field.tol,
            "npart": self.field.npart,
            "rjac": self.field.rjac,
            "scale_height": self.field.scale_height,
            "cp": self.field.cp,
            "dry_gas_constant": self.field.dry_gas_constant,
            "omega": self.field.omega,
            "planet_radius": self.field.planet_radius,
            "prefactor": self.field.prefactor,
            "protocol": type(self.field).__name__,
            "package": f"hn2016_falwa {__version__}"
        }

    # General information from a variable registry
    # (must be kept up-to-date with oopinterface.QGField, see below)

    _VARS = dict()

    @classmethod
    def register_var(cls, var, core_dims, dim_names=None, attrs=None):
        """Add a new variable configuration to the registry

        Parameters
        ----------
        var : string
            Name of the variable in the registry.
        core_dims : Tuple[string]
            Core dimensions of the variable, i.e. the fundamental dimensions
            that a single field of this variable always has. Core dimensions
            must always be the last dimensions in the array.
        dim_names : Tuple[string], optional
            Name overrides for data access on the QGField template object.
        attrs : dict, optional
            Attributes for the variable, attached to any produced DataArray.
        """
        cls._VARS[var] = {
            "core_dims": core_dims,
            "dim_names": dim_names if dim_names is not None else core_dims,
            "attrs": attrs
        }

    @classmethod
    def info(cls, var):
        """Metadata information from the variable registry"""
        return cls._VARS[var]


# Interpolated fields
_MetadataServiceProvider.register_var("qgpv", ("height", "ylat", "xlon"))
_MetadataServiceProvider.register_var("interpolated_u", ("height", "ylat", "xlon"))
_MetadataServiceProvider.register_var("interpolated_v", ("height", "ylat", "xlon"))
_MetadataServiceProvider.register_var("interpolated_theta", ("height", "ylat", "xlon"))
_MetadataServiceProvider.register_var("static_stability", ("height",))
_MetadataServiceProvider.register_var("static_stability_n", ("height",))
_MetadataServiceProvider.register_var("static_stability_s", ("height",))
# Reference state fields (y-z cross section)
_MetadataServiceProvider.register_var("qref", ("height", "ylat"), dim_names=("height","ylat_ref_states"))
_MetadataServiceProvider.register_var("uref", ("height", "ylat"), dim_names=("height", "ylat_ref_states"))
_MetadataServiceProvider.register_var("ptref", ("height", "ylat"), dim_names=("height", "ylat_ref_states"))
# Column-averaged fields (x-y horizontal fields)
_MetadataServiceProvider.register_var("u_baro", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("lwa_baro", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("adv_flux_f1", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("adv_flux_f2", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("adv_flux_f3", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("convergence_zonal_advective_flux", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("divergence_eddy_momentum_flux", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
_MetadataServiceProvider.register_var("meridional_heat_flux", ("ylat", "xlon"), dim_names=("ylat_ref_states", "xlon"))
# 3-dimensional LWA (full x-y-z fields)
_MetadataServiceProvider.register_var("lwa", ("height", "ylat", "xlon"), dim_names=("height", "ylat_ref_states", "xlon"))


class _DataArrayCollector(property):
    # Getter properties for DataArray-based access to QGField properties.
    # Inherits from property, so instances are recognized as properties by
    # sphinx for the docs.

    def __init__(self, var):
        self.var = var
        self.__doc__ = (
            f"See :py:attr:`oopinterface.QGFieldBase.{self.var}`."
            "\n\nReturns\n-------\nxarray.DataArray"
        )

    def __get__(self, qgds, objtype=None):
        arr = np.asarray([getattr(field, self.var) for field in qgds.fields])
        return qgds.metadata.as_dataarray(arr, self.var)


class QGDataset:
    """A wrapper for multiple QGField objects with xarray in- and output.

    For each combination of timestep, ensemble member, etc. in the input data,
    a :py:class:`oopinterface.QGField` object is instanciated. The constructor
    will automatically flip latitude and pressure dimensions of the input data
    if necessary to meet the requirements of QGField.

    This wrapper class imitates the methods of QGField (but not the
    properties/attributes) and collects and re-organizes output data in xarray
    Datasets for convenience. All calculations are performed by the QGField
    routines.

    .. versionadded:: 0.6.1

    Parameters
    ----------
    da_u : xarray.DataArray | xarray.Dataset
        Input 3D fields of zonal wind. The 3D fields's dimensions must end with
        height, latitude and longitude. Other dimensions (e.g. time, ensemble
        member id) are preserved in the output datasets.
        Alternatively, a dataset can be given, from which `u`, `v` and `T`
        fields are then extracted. The `da_v` and `da_t` arguments can then be
        omitted or used as an override.
    da_v : xarray.DataArray, optional
        Input 3D fields of meridional wind. The 3D fields's dimensions must end
        with height, latitude and longitude. Other dimensions (e.g. time,
        ensemble member id) are preserved in the output datasets.
    da_t : xarray.DataArray, optional
        Input 3D fields of temperature. The 3D fields's dimensions must end
        with height, latitude and longitude. Other dimensions (e.g. time,
        ensemble member id) are preserved in the output datasets.
    var_names : dict, optional
        If the auto-detection of variable or coordinate names fails, provide
        a lookup table that maps `plev`, `ylat`, `xlon`, `u`, `v` and/or `t` to
        the names used in the dataset.
    qgfield : QGField class, optional
        The QGField class to use in the computation. Default:
        :py:class:`oopinterface.QGFieldNH18`.
    qgfield_args : tuple, optional
        Positional arguments given to the QGField constructor.
    qgfield_kwargs : dict, optional
        Keyword arguments given to the QGField constructor.

    Examples
    -------
    >>> data = xarray.open_dataset("path/to/some/uvt-data.nc")
    >>> qgds = QGDataset(data)
    >>> qgds.interpolate_fields()
    <xarray.Dataset> ...

    :doc:`notebooks/demo_script_for_nh2018_with_xarray`

    >>> data_u = xarray.load_dataset("path/to/some/u-data.nc")
    >>> data_v = xarray.load_dataset("path/to/some/v-data.nc")
    >>> data_t = xarray.load_dataset("path/to/some/t-data.nc")
    >>> qgds = QGDataset(data_u, data_v, data_t)
    """

    def __init__(self, da_u, da_v=None, da_t=None, *, var_names=None,
                 qgfield=QGFieldNH18, qgfield_args=None, qgfield_kwargs=None):
        if var_names is None:
            var_names = dict()

        # Check input data type first
        assert isinstance(da_u, xr.Dataset) or isinstance(da_u, xr.DataArray)
        assert da_v is None or isinstance(da_v, xr.DataArray) or isinstance(da_v, xr.Dataset)
        assert da_t is None or isinstance(da_t, xr.DataArray) or isinstance(da_t, xr.Dataset)

        # Also support construction from single-arg and mixed variants
        if isinstance(da_u, xr.Dataset):
            # Fill up missing DataArrays for v and t from the Dataset but give
            # priority to existing v and t fields from the args
            if da_v is None:
                da_v = _get_dataarray(da_u, _NAMES_V, var_names)
            elif isinstance(da_v, xr.Dataset):
                da_v = _get_dataarray(da_v, _NAMES_V, var_names)
            # else: assume da_v is dataarray so there is no issue

            if da_t is None:
                da_t = _get_dataarray(da_u, _NAMES_T, var_names)
            elif isinstance(da_t, xr.Dataset):
                da_t = _get_dataarray(da_t, _NAMES_T, var_names)
            # else: assume da_t is dataarray so there is no issue

            # Always take u
            da_u = _get_dataarray(da_u, _NAMES_U, var_names)

        # Assertions about da_u, da_v, da_t
        assert da_u is not None, "missing u field"
        assert da_v is not None, "missing v field"
        assert da_t is not None, "missing t field"
        # Assign standard names to the input fields
        da_u = da_u.rename("u")
        da_v = da_v.rename("v")
        da_t = da_t.rename("t")
        # Merge into one dataset and keep the reference. xarray will avoid
        # copying the data in the merge, so the operation should be relatively
        # cheap and fast. The merge further verifies that the coordinates of
        # the three DataArrays match.
        self._ds = xr.merge([da_u, da_v, da_t], join="exact", compat="equals")
        # QGField* configuration
        self._qgfield = qgfield
        self._qgfield_args = list() if qgfield_args is None else qgfield_args
        self._qgfield_kwargs = dict() if qgfield_kwargs is None else qgfield_kwargs
        # Extract spatial coordinates
        da_plev = _get_dataarray(self._ds.coords, _NAMES_PLEV, var_names)
        da_ylat = _get_dataarray(self._ds.coords, _NAMES_YLAT, var_names)
        da_xlon = _get_dataarray(self._ds.coords, _NAMES_XLON, var_names)
        # Check that field coordinates end in lev, lat, lon
        assert da_u.dims[-3] == da_plev.name, f"dimension -3 of input fields must be '{da_plev.name}' (plev)"
        assert da_u.dims[-2] == da_ylat.name, f"dimension -2 of input fields must be '{da_ylat.name}' (ylat)"
        assert da_u.dims[-1] == da_xlon.name, f"dimension -1 of input fields must be '{da_xlon.name}' (xlon)"
        assert da_u.dims == da_v.dims, f"dimensions of fields '{da_u.name}' (u) and '{da_v.name}' (v) don't match"
        assert da_u.dims == da_t.dims, f"dimensions of fields '{da_u.name}' (u) and '{da_t.name}' (t) don't match"
        # The input data may contain multiple time steps, ensemble members etc.
        # Flatten all these other dimensions so a single loop covers all
        # fields. These dimensions are restored in the output datasets.
        other_dims = da_u.dims[:-3]
        other_shape = tuple(da_u[dim].size for dim in other_dims)
        other_size = np.product(other_shape, dtype=np.int64)
        _shape = (other_size, *da_u.shape[-3:])
        # Extract value arrays and collapse all additional dimensions
        u = da_u.data.reshape(_shape)
        v = da_v.data.reshape(_shape)
        t = da_t.data.reshape(_shape)
        # Automatically determine how fields need to be flipped so they match
        # the requirements of QGField and extract coordinate values
        flip = []
        # Ensure that ylat is ascending
        ylat = da_ylat.values
        if not _is_ascending(ylat):
            ylat = np.flip(ylat)
            flip.append(-2)
        # Ensure that plev is descending
        plev = da_plev.values
        if not _is_descending(plev):
            plev = np.flip(plev)
            flip.append(-3)
        # Ordering of xlon doesn't matter here
        xlon = da_xlon.values
        # Create a QGField object for each combination of timestep, ensemble
        # member, etc.
        self._fields = []
        for u_field, v_field, t_field in zip(u, v, t):
            # Apply reordering to fields
            if flip:
                u_field = np.flip(u_field, axis=flip)
                v_field = np.flip(v_field, axis=flip)
                t_field = np.flip(t_field, axis=flip)
            field = self._qgfield(xlon, ylat, plev, u_field, v_field, t_field,
                                  *self._qgfield_args, **self._qgfield_kwargs)
            self._fields.append(field)
        # Make sure there is at least one field in the dataset
        assert self._fields, "empty input"
        # Tailored metadata access
        self.metadata = _MetadataServiceProvider(self._fields[0], other_coords={
            dim: self._ds.coords[dim] for dim in other_dims
        })

    @property
    def fields(self):
        """Access to the QGField objects created by the QGDataset.

        The :py:class:`.oopinterface.QGField` objects are stored in a flattened
        list.
        """
        return self._fields

    @property
    def attrs(self):
        """Metadata dictionary that is attached to output datasets."""
        return self.metadata.attrs()

    def interpolate_fields(self, return_dataset=True):
        """Call `interpolate_fields` on all contained fields.

        See :py:meth:`.oopinterface.QGFieldBase.interpolate_fields`.

        .. note::
            A QGField class may define static stability globally or
            hemispherically on each level. The output dataset contains a single
            variable for static stability in case of a global definition and
            two variables for static stability for a hemispheric definition
            (suffix ``_n`` for the northern hemisphere and ``_s`` for the
            southern hemisphere).

        Parameters
        ----------
        return_dataset : bool
            Whether to return the computed fields as a dataset.

        Returns
        -------
        xarray.Dataset or None
        """
        for field in self._fields:
            field.interpolate_fields(return_named_tuple=False)
        if return_dataset:
            data_vars = {
                "qgpv": self.qgpv,
                "interpolated_u": self.interpolated_u,
                "interpolated_v": self.interpolated_v,
                "interpolated_theta": self.interpolated_theta
            }
            # Stability property may contain multiple variables
            stability = self.static_stability
            if isinstance(stability, xr.DataArray):
                stability = (stability,)
            data_vars.update({ s.name: s for s in stability })
            return xr.Dataset(data_vars, attrs=self.attrs)

    # Accessors for individual field properties computed in interpolate_fields
    qgpv = _DataArrayCollector("qgpv")
    interpolated_u = _DataArrayCollector("interpolated_u")
    interpolated_v = _DataArrayCollector("interpolated_v")
    interpolated_theta = _DataArrayCollector("interpolated_theta")

    @property
    def static_stability(self):
        """See :py:attr:`oopinterface.QGFieldBase.static_stability`.

        Returns
        -------
        xr.Dataset | Tuple[xr.Dataset, xr.Dataset]
        """
        stability = np.asarray([getattr(field, "static_stability") for field in self._fields])
        if stability.ndim == 2:
            # One vertical profile of static stability per field: global
            return self.metadata.as_dataarray(stability, "static_stability")
        elif stability.ndim == 3 and stability.shape[-2] == 2:
            # Two vertical profiles of static stability per field: hemispheric
            return (
                self.metadata.as_dataarray(stability[:,0,:], "static_stability_n"),
                self.metadata.as_dataarray(stability[:,1,:], "static_stability_s")
            )
        else:
            raise ValueError(f"cannot process shape of returned static stability field: {stability.shape}")


    def compute_reference_states(self, return_dataset=True):
        """Call `compute_reference_states` on all contained fields.

        See :py:meth:`.oopinterface.QGFieldBase.compute_reference_states`.

        Parameters
        ----------
        return_dataset : bool
            Whether to return the computed fields as a dataset.

        Returns
        -------
        xarray.Dataset or None
        """
        for field in self._fields:
            field.compute_reference_states(return_named_tuple=False)
        if return_dataset:
            data_vars = {
                "qref": self.qref,
                "uref": self.uref,
                "ptref": self.ptref,
            }
            return xr.Dataset(data_vars, attrs=self.attrs)

    # Accessors for individual field properties computed in compute_reference_states
    qref = _DataArrayCollector("qref")
    uref = _DataArrayCollector("uref")
    ptref = _DataArrayCollector("ptref")

    def compute_lwa_and_barotropic_fluxes(self, return_dataset=True):
        """Call `compute_lwa_and_barotropic_fluxes` on all contained fields.

        See :py:meth:`.oopinterface.QGFieldBase.compute_lwa_and_barotropic_fluxes`.

        Parameters
        ----------
        return_dataset : bool
            Whether to return the computed fields as a dataset.

        Returns
        -------
        xarray.Dataset or None
        """
        for field in self._fields:
            field.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)
        if return_dataset:
            data_vars = {
                "adv_flux_f1": self.adv_flux_f1,
                "adv_flux_f2": self.adv_flux_f2,
                "adv_flux_f3": self.adv_flux_f3,
                "convergence_zonal_advective_flux": self.convergence_zonal_advective_flux,
                "divergence_eddy_momentum_flux": self.divergence_eddy_momentum_flux,
                "meridional_heat_flux": self.meridional_heat_flux,
                "lwa_baro": self.lwa_baro,
                "u_baro": self.u_baro,
                "lwa": self.lwa,
            }
            return xr.Dataset(data_vars, attrs=self.attrs)

    # Accessors for individual field properties computed in compute_lwa_and_barotropic_fluxes
    adv_flux_f1 = _DataArrayCollector("adv_flux_f1")
    adv_flux_f2 = _DataArrayCollector("adv_flux_f2")
    adv_flux_f3 = _DataArrayCollector("adv_flux_f3")
    convergence_zonal_advective_flux = _DataArrayCollector("convergence_zonal_advective_flux")
    divergence_eddy_momentum_flux = _DataArrayCollector("divergence_eddy_momentum_flux")
    meridional_heat_flux = _DataArrayCollector("meridional_heat_flux")
    lwa_baro = _DataArrayCollector("lwa_baro")
    u_baro = _DataArrayCollector("u_baro")
    lwa = _DataArrayCollector("lwa")



def integrate_budget(ds, var_names=None):
    """Compute the integrated LWA budget terms for the given data.

    Integrates the LWA tendencies from equation (2) of `NH18
    <https://doi.org/10.1126/science.aat0721>`_ in time (over the time interval
    covered by the input data). The residual (term IV) is determined by
    subtracting terms (I), (II) and (III) from the LWA difference between the
    last and first time step in the data. Uses
    :py:meth:`xarray.DataArray.integrate` for the time integration of the
    tendencies.

    See :py:meth:`QGDataset.compute_lwa_and_barotropic_fluxes`, which computes
    all required tendency terms as well as the LWA fields.

    .. versionadded:: 0.6.1

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the budget tendencies for the time integration
        interval.
    var_names : dict, optional
        The names of LWA and the tendency term variables are automatically
        detected. If the auto-detection fails, provide a lookup table that maps
        `time`, `lwa_baro`, `convergence_zonal_advective_flux`,
        `divergence_eddy_momentum_flux`, and/or `meridional_heat_flux` to the
        names used in the input dataset.

    Returns
    -------
    xarray.Dataset

    Examples
    -------
    >>> qgds = QGDataset(data)
    >>> ...
    >>> terms = qgds.compute_lwa_and_barotropic_fluxes()
    >>> integrate_budget(terms.isel({ "time": slice(5, 10) }))
    """
    name_time = _get_name(ds, _NAMES_TIME, var_names)
    name_lwa  = _get_name(ds, _NAMES_LWA,  var_names)
    name_czaf = _get_name(ds, _NAMES_CZAF, var_names)
    name_demf = _get_name(ds, _NAMES_DEMF, var_names)
    name_mhf  = _get_name(ds, _NAMES_MHF,  var_names)
    # Integration time interval covered by the data
    start = ds[name_time].values[0]
    stop = ds[name_time].values[-1]
    # Determine the change in LWA over the time interval
    dlwa = ds[name_lwa].sel({ name_time: stop }) - ds[name_lwa].sel({ name_time: start })
    # Integrate the known tendencies in time
    czaf = ds[name_czaf].integrate(coord=name_time, datetime_unit="s")
    demf = ds[name_demf].integrate(coord=name_time, datetime_unit="s")
    mhf  = ds[name_mhf].integrate(coord=name_time, datetime_unit="s")
    # Compute the residual from the difference between the explicitly computed
    # budget terms and the actual change in LWA
    res  = dlwa - czaf - demf - mhf
    # Include all 5 integrated budget terms in the output
    data_vars = {
        "delta_lwa": dlwa,
        "integrated_convergence_zonal_advective_flux": czaf,
        "integrated_divergence_eddy_momentum_flux": demf,
        "integrated_meridional_heat_flux": mhf,
        "residual": res
    }
    # Copy attributes from original dataset and add information about
    # integration interval (start and end timestamps as well as integration
    # time interval in seconds)
    attrs = dict(ds.attrs)
    attrs["integration_start"] = str(start)
    attrs["integration_stop"] = str(stop)
    attrs["integration_seconds"] = (stop - start) / np.timedelta64(1000000000)
    return xr.Dataset(data_vars, ds.coords, attrs)


def hemisphere_to_globe(ds, var_names=None):
    """Create a global dataset from a hemispheric one.

    Takes data from the given hemisphere, mirrors it to the other hemisphere
    and combines both hemispheres into a global dataset.

    If the meridional wind component is found in the dataset, its values will
    be negated on the created hemisphere. This results in identical fields of
    local wave activity on both hemispheres (since absolute vorticity is also
    the same except for the sign), making it possible to use
    `northern_hemisphere_only` in the methods of :py:class:`QGDataset` even if
    only southern hemisphere data is available. Discontinuities in the
    meridional wind and derived fields arise due to this at the equator but
    they generally have only a small effect on the outputs.

    .. versionadded:: 0.6.1

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset. Must contain the equator (0° latitude).
    var_names : dict, optional
        The names of the latitude and meridional wind fields are automatically
        detected. If the auto-detection of the latitude coordinate and/or the
        meridional wind component fails, provide a lookup table that maps
        `ylat`, and/or `v` to the names used in the dataset.

    Returns
    -------
    xarray.Dataset
    """
    # Determine if the northern or southern hemisphere is present
    ylat_name = _get_name(ds, _NAMES_YLAT, var_names)
    eq0 = _is_equator(ds[ylat_name][0])
    assert eq0 or _is_equator(ds[ylat_name][-1]), (
        "equator not found on the hemisphere; "
        "make sure latitudes either begin or end with 0° latitude"
    )
    # Flip the data along ylat and omit the equator which should not appear
    # twice in the output
    flipped_noeq = slice(None, 0, -1) if eq0 else slice(-2, None, -1)
    sd = ds.reindex({ ylat_name: ds[ylat_name][flipped_noeq] })
    # Latitudes are now on the other hemisphere
    sd[ylat_name] = -sd[ylat_name]
    # Also flip the meridional wind (if present in the dataset). This results
    # in mirrored LWA fields on both hemispheres, the discontinuities this
    # creates on the equator are acceptable.
    try:
        v_name = _get_name(ds, _NAMES_V, var_names)
        sd[v_name] = -sd[v_name]
    except KeyError:
        pass
    # Assemble global dataset
    return xr.concat([sd, ds] if eq0 else [ds, sd], dim=ylat_name)
