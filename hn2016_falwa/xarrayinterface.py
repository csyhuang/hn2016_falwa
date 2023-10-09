"""
------------------------------------------
File name: xarrayinterface.py
Author: Christopher Polster
"""
import functools
import numpy as np
import xarray as xr

from hn2016_falwa import __version__
from hn2016_falwa.oopinterface import QGFieldNH18


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


def _map_collect(f, xs, names, postprocess=None):
    out = { name: [] for name in names }
    for x in xs:
        for name, y in zip(names, f(x)):
            out[name].append(y)
    if postprocess is not None:
        for name in names:
            out[name] = postprocess(out[name])
    return out


class _DataArrayCollector(property):
    # Getter properties for DataArray-based access to QGField properties.
    # Inherits from property, so instances are recognized as properties by
    # sphinx for the docs.

    def __init__(self, name, dimnames, dimvars=None):
        self.name = name
        self.dimnames = dimnames
        self.dimvars = dimvars if dimvars is not None else dimnames
        self.__doc__ = (
            f"See :py:attr:`oopinterface.QGField.{name}`."
            "\n\nReturns\n-------\nxarray.DataArray"
        )

    def __get__(self, obj, objtype=None):
        fields = obj.fields
        data = np.asarray([getattr(field, self.name) for field in fields])
        coords = ({
            coord: getattr(fields[0], var)
            for coord, var in zip(self.dimnames, self.dimvars)
        })
        coords.update(obj._other_coords)
        dims = (*obj._other_dims, *self.dimnames)
        shape = (*obj._other_shape, *(getattr(fields[0], var).size for var in self.dimvars))
        return xr.DataArray(data.reshape(shape), coords, dims, self.name, obj.attrs)


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
        # Also support construction from single-arg and mixed variants
        if isinstance(da_u, xr.Dataset):
            # Fill up missing DataArrays for v and t from the Dataset but give
            # priority to existing v and t fields from the args
            if da_v is None:
                da_v = _get_dataarray(da_u, _NAMES_V, var_names)
            if da_t is None:
                da_t = _get_dataarray(da_u, _NAMES_T, var_names)
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
        self._other_dims = da_u.dims[:-3]
        self._other_shape = tuple(da_u[dim].size for dim in self._other_dims)
        self._other_size = np.product(self._other_shape, dtype=np.int64)
        _shape = (self._other_size, *da_u.shape[-3:])
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

    @property
    def fields(self):
        """Access to the QGField objects created by the QGDataset.

        The :py:class:`.oopinterface.QGField` objects are stored in a flattened
        list.
        """
        return self._fields

    @property
    def _other_coords(self):
        return {dim: self._ds[dim] for dim in self._other_dims}

    @property
    def attrs(self):
        """Metadata dictionary that is attached to output datasets."""
        field = self._fields[0]
        return {
            "kmax": field.kmax,
            "dz": field.dz,
            "maxit": field.maxit,
            "tol": field.tol,
            "npart": field.npart,
            "rjac": field.rjac,
            "scale_height": field.scale_height,
            "cp": field.cp,
            "dry_gas_constant": field.dry_gas_constant,
            "omega": field.omega,
            "planet_radius": field.planet_radius,
            "prefactor": field.prefactor,
            "protocol": self._qgfield.__name__,
            "package": f"hn2016_falwa {__version__}"
        }

    def interpolate_fields(self):
        """Collect the output of `interpolate_fields` in a dataset.

        See :py:meth:`.oopinterface.QGField.interpolate_fields`.

        .. note::
            A QGField class may define static stability globally or
            hemispherically on each level. The output dataset contains a single
            variable for static stability in case of a global definition and
            two variables for static stability for a hemispheric definition
            (suffix ``_n`` for the northern hemisphere and ``_s`` for the
            southern hemisphere).

        Returns
        -------
        xarray.Dataset
        """
        # Call interpolate_fields on all QGField objects
        out_fields = _map_collect(
            lambda field: field.interpolate_fields(),
            self._fields,
            ["qgpv", "interpolated_u", "interpolated_v", "interpolated_theta", "static_stability"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # TODO: fix the code below for even-number latitude grid point scenario
        ylat_output = _field.ylat_no_equator if _field.need_latitude_interpolation else _field.ylat
        # Prepare coordinate-related data for the output: interpolated data is
        # transferred onto the QG height grid, fields are functions of height,
        # latitude, longitude
        out_dims = (*self._other_dims, "height", "ylat", "xlon")
        out_shape = (*self._other_shape, _field.height.size, _field.ylat.size, _field.xlon.size)
        # Special case: static stability (global for NH18, hemispheric for NHN22)
        stability = out_fields["static_stability"]
        data_vars_stability = {}
        if stability.ndim == 2:
            # One vertical profile of static stability per field: global
            data_vars_stability["static_stability"] = (out_dims[:-2], stability.reshape(out_shape[:-2]))
        elif stability.ndim == 3 and stability.shape[-2] == 2:
            # Two vertical profiles of static stability per field: hemispheric
            data_vars_stability["static_stability_n"] = (out_dims[:-2], stability[:,0,:].reshape(out_shape[:-2]))
            data_vars_stability["static_stability_s"] = (out_dims[:-2], stability[:,1,:].reshape(out_shape[:-2]))
        else:
            raise ValueError(f"cannot process shape of returned static stability field: {stability.shape}")
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "qgpv": (out_dims, out_fields["qgpv"].reshape(out_shape)),
                "interpolated_u": (out_dims, out_fields["interpolated_u"].reshape(out_shape)),
                "interpolated_v": (out_dims, out_fields["interpolated_v"].reshape(out_shape)),
                "interpolated_theta": (out_dims, out_fields["interpolated_theta"].reshape(out_shape)),
                **data_vars_stability
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _field.ylat,
                "xlon": _field.xlon,
            },
            attrs=self.attrs
        )

    # Accessors for individual field properties computed in interpolate_fields
    qgpv = _DataArrayCollector(
        "qgpv",
        ["height", "ylat", "xlon"]
    )
    interpolated_u = _DataArrayCollector(
        "interpolated_u",
        ["height", "ylat", "xlon"]
    )
    interpolated_v = _DataArrayCollector(
        "interpolated_v",
        ["height", "ylat", "xlon"]
    )
    interpolated_theta = _DataArrayCollector(
        "interpolated_theta",
        ["height", "ylat", "xlon"]
    )

    def compute_reference_states(self):
        """Collect the output of `compute_reference_states` in a dataset.

        See :py:meth:`.oopinterface.QGField.compute_reference_states`.

        Returns
        -------
        xarray.Dataset
        """
        # Call compute_reference_states on all QGField objects
        out_fields = _map_collect(
            lambda field: field.compute_reference_states(),
            self._fields,
            ["qref", "uref", "ptref"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output
        _ylat = _field.ylat_ref_states
        # 2D data, function of height and latitude
        out_dims = (*self._other_dims, "height", "ylat")
        out_shape = (*self._other_shape, _field.height.size, _ylat.size)
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "qref": (out_dims, out_fields["qref"].reshape(out_shape)),
                "uref": (out_dims, out_fields["uref"].reshape(out_shape)),
                "ptref": (out_dims, out_fields["ptref"].reshape(out_shape)),
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _ylat,
            },
            attrs=self.attrs
        )

    # Accessors for individual field properties computed in compute_reference_states
    qref = _DataArrayCollector(
        "qref",
        ["height", "ylat"],
        ["height", "ylat_ref_states"]
    )
    uref = _DataArrayCollector(
        "uref",
        ["height", "ylat"],
        ["height", "ylat_ref_states"]
    )
    ptref = _DataArrayCollector(
        "ptref",
        ["height", "ylat"],
        ["height", "ylat_ref_states"]
    )

    def compute_lwa_and_barotropic_fluxes(self):
        """Collect the output of `compute_lwa_and_barotropic_fluxes` in a dataset.

        See :py:meth:`.oopinterface.QGField.compute_lwa_and_barotropic_fluxes`.

        Returns
        -------
        xarray.Dataset
        """
        # Call compute_lwa_and_barotropic_fluxes on all QGField objects
        out_fields = _map_collect(
            lambda field: field.compute_lwa_and_barotropic_fluxes(),
            self._fields,
            ["adv_flux_f1", "adv_flux_f2", "adv_flux_f3", "convergence_zonal_advective_flux",
                "divergence_eddy_momentum_flux", "meridional_heat_flux", "lwa_baro", "u_baro",
                "lwa"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output
        _ylat = _field.ylat_ref_states
        # 2D data, function of latitude and longitude
        out_dims_2d = (*self._other_dims, "ylat", "xlon")
        out_shape_2d = (*self._other_shape, _ylat.size, _field.xlon.size)
        # 3D data, function of height, latitude and longitude
        out_dims_3d = (*self._other_dims, "height", "ylat", "xlon")
        out_shape_3d = (*self._other_shape, _field.height.size, _ylat.size, _field.xlon.size)
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "adv_flux_f1": (out_dims_2d, out_fields["adv_flux_f1"].reshape(out_shape_2d)),
                "adv_flux_f2": (out_dims_2d, out_fields["adv_flux_f2"].reshape(out_shape_2d)),
                "adv_flux_f3": (out_dims_2d, out_fields["adv_flux_f3"].reshape(out_shape_2d)),
                "convergence_zonal_advective_flux": (out_dims_2d, out_fields["convergence_zonal_advective_flux"].reshape(out_shape_2d)),
                "divergence_eddy_momentum_flux": (out_dims_2d, out_fields["divergence_eddy_momentum_flux"].reshape(out_shape_2d)),
                "meridional_heat_flux": (out_dims_2d, out_fields["meridional_heat_flux"].reshape(out_shape_2d)),
                "lwa_baro": (out_dims_2d, out_fields["lwa_baro"].reshape(out_shape_2d)),
                "u_baro": (out_dims_2d, out_fields["u_baro"].reshape(out_shape_2d)),
                "lwa": (out_dims_3d, out_fields["lwa"].reshape(out_shape_3d)),
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _ylat,
                "xlon": _field.xlon,
            },
            attrs=self.attrs
        )

    # Accessors for individual field properties computed in compute_lwa_and_barotropic_fluxes
    adv_flux_f1 = _DataArrayCollector(
        "adv_flux_f1",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    adv_flux_f2 = _DataArrayCollector(
        "adv_flux_f2",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    adv_flux_f3 = _DataArrayCollector(
        "adv_flux_f3",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    convergence_zonal_advective_flux = _DataArrayCollector(
        "convergence_zonal_advective_flux",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    divergence_eddy_momentum_flux = _DataArrayCollector(
        "divergence_eddy_momentum_flux",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    meridional_heat_flux = _DataArrayCollector(
        "meridional_heat_flux",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    lwa_baro = _DataArrayCollector(
        "lwa_baro",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    u_baro = _DataArrayCollector(
        "u_baro",
        ["ylat", "xlon"],
        ["ylat_ref_states", "xlon"]
    )
    lwa = _DataArrayCollector(
        "lwa",
        ["height", "ylat", "xlon"],
        ["height", "ylat_ref_states", "xlon"]
    )



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
    >>> terms = qgds.compute_lwa_and_barotropic_fluxes()
    >>> compute_budget(terms.isel({ "time": slice(5, 10) }))
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
    be negated. This results in identical fields of local wave activity on both
    hemispheres (since absolute vorticity is also the same except for the
    sign), making it possible to use `northern_hemisphere_only` in the methods
    of :py:class:`QGDataset` even if only southern hemisphere data is
    available. Discontinuities in the meridional wind and derived fields arise
    due to this at the equator but they generally have only a small effect on
    the outputs.

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
