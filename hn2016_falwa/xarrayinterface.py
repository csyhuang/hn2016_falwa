"""
------------------------------------------
File name: xarrayinterface.py
Author: Christopher Polster
"""
import functools
import numpy as np
import xarray as xr

from hn2016_falwa import __version__
from hn2016_falwa.oopinterface import QGField


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


class QGDataset:
    """A wrapper for multiple QGField objects with xarray in- and output.

    Examines the given dataset and tries to extract `u`, `v`, and `T` fields
    based on the names of coordinates in the dataset. For each combination of
    timestep, ensemble member, etc., a :py:class:`oopinterface.QGField` object
    is instanciated. The constructor will automatically flip latitude and
    pressure dimensions of the input data if necessary to meet the requirements
    of QGField.

    This wrapper class imitates the methods of QGField (but not the
    properties/attributes) and collects and re-organizes output data in xarray
    Datasets for convenience. All calculations are performed by the QGField
    routines.

    .. versionadded:: 0.6.1

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset. Must contain 3D fields of zonal wind, meridional wind
        and temperature. The 3D fields's dimensions must end with height,
        latitude and longitude. Other dimensions (e.g. time, ensemble member
        id) are preserved in the output datasets.
    qgfield_args : tuple, optional
        Positional arguments given to the QGField constructor.
    qgfield_kwargs : dict, optional
        Keyword arguments given to the QGField constructor.
    var_names : dict, optional
        If the auto-detection of variable or coordinate names fails, provide
        a lookup table that maps `plev`, `ylat`, `xlon`, `u`, `v` and/or `t` to
        the names used in the dataset.

    Example
    -------
    >>> data = xarray.load_dataset("path/to/some/uvt-data.nc")
    >>> qgds = QGDataset(data)
    """

    def __init__(self, ds, qgfield_args=None, qgfield_kwargs=None, var_names=None):
        if var_names is None:
            var_names = dict()
        self._ds = ds
        self._qgfield_args   = list() if qgfield_args   is None else qgfield_args
        self._qgfield_kwargs = dict() if qgfield_kwargs is None else qgfield_kwargs
        # Find names of spatial coordinates
        self._plev_name = _get_name(ds, _NAMES_PLEV, var_names)
        self._ylat_name = _get_name(ds, _NAMES_YLAT, var_names)
        self._xlon_name = _get_name(ds, _NAMES_XLON, var_names)
        # Find names of wind and temperature fields
        self._u_name = _get_name(ds, _NAMES_U, var_names)
        self._v_name = _get_name(ds, _NAMES_V, var_names)
        self._t_name = _get_name(ds, _NAMES_T, var_names)
        # Shorthands for data arrays
        plev = ds[self._plev_name]
        ylat = ds[self._ylat_name]
        xlon = ds[self._xlon_name]
        u = ds[self._u_name]
        v = ds[self._v_name]
        t = ds[self._t_name]
        # Check that field coordinates end in lev, lat, lon
        assert u.dims[-3] == plev.name, f"dimension -3 of input fields must be '{plev.name}' (plev)"
        assert u.dims[-2] == ylat.name, f"dimension -2 of input fields must be '{ylat.name}' (ylat)"
        assert u.dims[-1] == xlon.name, f"dimension -1 of input fields must be '{xlon.name}' (xlon)"
        assert u.dims == v.dims, f"dimensions of fields '{u.name}' (u) and '{v.name}' (v) don't match"
        assert u.dims == t.dims, f"dimensions of fields '{u.name}' (u) and '{t.name}' (t) don't match"
        # The input data may contain multiple time steps, ensemble members etc.
        # Flatten all these other dimensions so a single loop covers all
        # fields. These dimensions are restored in the output datasets.
        self._other_dims = u.dims[:-3]
        self._other_shape = tuple(ds[dim].size for dim in self._other_dims)
        self._other_size = np.product(self._other_shape, dtype=np.int64)
        _shape = (self._other_size, *u.shape[-3:])
        # Extract value arrays and collapse all additional dimensions
        u = u.data.reshape(_shape)
        v = v.data.reshape(_shape)
        t = t.data.reshape(_shape)
        # Automatically determine how fields need to be flipped so they match
        # the requirements of QGField and extract coordinate values
        flip = []
        # Ensure that ylat is ascending
        ylat = ylat.values
        if not _is_ascending(ylat):
            ylat = np.flip(ylat)
            flip.append(-2)
        # Ensure that plev is descending
        plev = plev.values
        if not _is_descending(plev):
            plev = np.flip(plev)
            flip.append(-3)
        # Ordering of xlon doesn't matter here
        xlon = xlon.values
        # Create a QGField object for each combination of timestep, ensemble
        # member, etc.
        self._fields = []
        for u_field, v_field, t_field in zip(u, v, t):
            # Apply reordering to fields
            if flip:
                u_field = np.flip(u_field, axis=flip)
                v_field = np.flip(v_field, axis=flip)
                t_field = np.flip(t_field, axis=flip)
            field = QGField(xlon, ylat, plev, u_field, v_field, t_field,
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
        return { dim: self._ds[dim] for dim in self._other_dims}

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
            "package": f"hn2016_falwa {__version__}"
        }

    def interpolate_fields(self):
        """Collect the output of `interpolate_fields` in a dataset.

        See :py:meth:`.oopinterface.QGField.interpolate_fields`.

        Returns
        -------
        xarray.Dataset
        """
        # Call interpolate_fields on all QGField objects
        out_fields = _map_collect(
            lambda field: field.interpolate_fields(),
            self._fields,
            ["qgpv", "interpolated_u", "interpolated_v", "theta", "static_stability"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output: interpolated data is
        # transferred onto the QG height grid, fields are functions of height,
        # latitude, longitude
        out_dims = (*self._other_dims, "height", "ylat", "xlon")
        out_shape = (*self._other_shape, _field.height.size, _field.ylat.size, _field.xlon.size)
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "qgpv": (out_dims, out_fields["qgpv"].reshape(out_shape)),
                "interpolated_u": (out_dims, out_fields["interpolated_u"].reshape(out_shape)),
                "interpolated_v": (out_dims, out_fields["interpolated_v"].reshape(out_shape)),
                "theta": (out_dims, out_fields["theta"].reshape(out_shape)),
                "static_stability": (out_dims[:-2], out_fields["static_stability"].reshape(out_shape[:-2]))
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _field.ylat,
                "xlon": _field.xlon,
            },
            attrs=self.attrs
        )

    def compute_reference_states(self, northern_hemisphere_results_only=False):
        """Collect the output of `compute_reference_states` in a dataset.

        See :py:meth:`.oopinterface.QGField.compute_reference_states`.

        Returns
        -------
        xarray.Dataset
        """
        # Call compute_reference_states on all QGField objects
        out_fields = _map_collect(
            lambda field: field.compute_reference_states(northern_hemisphere_results_only),
            self._fields,
            ["qref", "uref", "ptref"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output
        if northern_hemisphere_results_only:
            _ylat = _field.ylat[(_field.equator_idx - 1):]            
        else:
            _ylat = _field.ylat
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

    def compute_lwa_and_barotropic_fluxes(self, northern_hemisphere_results_only=False):
        """Collect the output of `compute_lwa_and_barotropic_fluxes` in a dataset.

        See :py:meth:`.oopinterface.QGField.compute_lwa_and_barotropic_fluxes`.

        Returns
        -------
        xarray.Dataset
        """
        # Call compute_lwa_and_barotropic_fluxes on all QGField objects
        out_fields = _map_collect(
            lambda field: field.compute_lwa_and_barotropic_fluxes(northern_hemisphere_results_only),
            self._fields,
            ["adv_flux_f1", "adv_flux_f2", "adv_flux_f3", "convergence_zonal_advective_flux",
                "divergence_eddy_momentum_flux", "meridional_heat_flux", "lwa_baro", "u_baro",
                "lwa"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output
        if northern_hemisphere_results_only:
            _ylat = _field.ylat[(_field.equator_idx - 1):]            
        else:
            _ylat = _field.ylat
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

    # The new routines so far only seem to work for 1°-resolution data and the northern hemisphere

    def _interpolate_field_dirinv(self):
        # Call interpolate_field_dirinv on all QGField objects
        out_fields = _map_collect(
            lambda field: field._interpolate_field_dirinv(),
            self._fields,
            ["qgpv", "interpolated_u", "interpolated_v", "interpolated_avort",
                "interpolated_theta", "static_stability_n", "static_stability_s",
                "tn0", "ts0"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        # Prepare coordinate-related data for the output: interpolated data is
        # transferred onto the QG height grid.
        # 1D data, function of height only
        out_dims_h = (*self._other_dims, "height")
        out_shape_h = (*self._other_shape, _field.height.size)
        # 3D data, function of longitude, latitude and height (plev and xlon
        # dimensions are not swapped back in the dirinv routines currently)
        out_dims_xyh = (*self._other_dims, "xlon", "ylat", "height")
        out_shape_xyh = (*self._other_shape, _field.xlon.size, _field.ylat.size, _field.height.size)
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "qgpv": (out_dims_xyh, out_fields["qgpv"].reshape(out_shape_xyh)),
                "interpolated_u": (out_dims_xyh, out_fields["interpolated_u"].reshape(out_shape_xyh)),
                "interpolated_v": (out_dims_xyh, out_fields["interpolated_v"].reshape(out_shape_xyh)),
                "interpolated_avort": (out_dims_xyh, out_fields["interpolated_avort"].reshape(out_shape_xyh)),
                "interpolated_theta": (out_dims_xyh, out_fields["interpolated_theta"].reshape(out_shape_xyh)),
                "static_stability_n": (out_dims_h, out_fields["static_stability_n"].reshape(out_shape_h)),
                "static_stability_s": (out_dims_h, out_fields["static_stability_s"].reshape(out_shape_h)),
                "tn0": (out_dims_h, out_fields["tn0"].reshape(out_shape_h)),
                "ts0": (out_dims_h, out_fields["ts0"].reshape(out_shape_h)),
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _field.ylat,
                "xlon": _field.xlon,
            },
            attrs=self.attrs
        )

    def _compute_qref_fawa_and_bc(self):
        # Call _compute_qref_fawa_and_bc on all QGField objects
        out_fields = _map_collect(
            lambda field: field._compute_qref_fawa_and_bc(),
            self._fields,
            ["qref", "u", "tref", "fawa", "ubar", "tbar"],
            postprocess=np.asarray
        )
        # The output of _compute_qref_fawa_and_bc is currently not stored in
        # the QGField object and must be given to _compute_lwa_flux_dirinv
        # explicitly. Until a better solution is found in the QGField
        # implementation, apply a monkey patch here: add the outputs of
        # _compute_qref_fawa_and_bc as underscore-attributes to the QGField
        # objects so they can be retrieved later.
        for name, arrs in out_fields.items():
            for field, arr in zip(self._fields, arrs):
                setattr(field, "_temp_dirinv_" + name, arr)
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        _nlat = _field.nlat // 2 + _field.nlat % 2
        # Prepare coordinate-related data for the output: all outputs are
        # functions of latitude and height
        out_dims_yh = (*self._other_dims, "ylat", "height")
        out_shape_yh = (*self._other_shape, _nlat, _field.height.size)
        # Output fields u (=uref) and tref currently exclude the equator
        # boundary points and are padded here for more convenient, consistent
        # output array shapes.
        _pad = functools.partial(
            np.pad,
            pad_width=[(0, 0), (_field.eq_boundary_index, 0), (0, 0)],
            mode="constant",
            constant_values=np.nan
        )
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier. u is returned under the
        # name uref to avoid confusion about the content.
        return xr.Dataset(
            data_vars={
                "qref": (out_dims_yh, out_fields["qref"].reshape(out_shape_yh)),
                "uref": (out_dims_yh, _pad(out_fields["u"]).reshape(out_shape_yh)),
                "tref": (out_dims_yh, _pad(out_fields["tref"]).reshape(out_shape_yh)),
                "fawa": (out_dims_yh, out_fields["fawa"].reshape(out_shape_yh)),
                "ubar": (out_dims_yh, out_fields["ubar"].reshape(out_shape_yh)),
                "tbar": (out_dims_yh, out_fields["tbar"].reshape(out_shape_yh)),
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _field.ylat[-_nlat:],
            },
            attrs=self.attrs
        )
        return out_fields

    def _compute_lwa_flux_dirinv(self):
        # Call _compute_lwa_flux_dirinv on all QGField objects, use the monkey
        # patched attributes added in _compute_qref_fawa_and_bc
        out_fields = _map_collect(
            lambda field: field._compute_lwa_flux_dirinv(qref=field._temp_dirinv_qref, uref=field._temp_dirinv_u,
                                                         tref=field._temp_dirinv_tref),
            self._fields,
            ["astarbaro", "ubaro", "urefbaro", "ua1baro", "ua2baro", "ep1baro",
                "ep2baro", "ep3baro", "ep4", "astar1", "astar2"],
            postprocess=np.asarray
        )
        # Take the first field to extract coordinates and metadata
        _field = self.fields[0]
        _nlat = _field.nlat // 2 + _field.nlat % 2
        # Prepare coordinate-related data for the output:
        # 1D data, function of latitude only
        out_dims_y = (*self._other_dims, "ylat")
        out_shape_y = (*self._other_shape, _nlat)
        # 2D data, function of longitude and latitude
        out_dims_xy = (*self._other_dims, "xlon", "ylat")
        out_shape_xy = (*self._other_shape, _field.xlon.size, _nlat)
        # 3D data, function of longitude, latitude and height (again, xlon and
        # plev are currently not swapped back in the dirinv routines)
        out_dims_xyh = (*self._other_dims, "xlon", "ylat", "height")
        out_shape_xyh = (*self._other_shape, _field.xlon.size, _nlat, _field.height.size)
        # Combine all outputs into a dataset, reshape to restore the original
        # other dimensions that were flattened earlier
        return xr.Dataset(
            data_vars={
                "astarbaro": (out_dims_xy, out_fields["astarbaro"].reshape(out_shape_xy)),
                "ubaro": (out_dims_xy, out_fields["ubaro"].reshape(out_shape_xy)),
                "urefbaro": (out_dims_y, out_fields["urefbaro"].reshape(out_shape_y)),
                "ua1baro": (out_dims_xy, out_fields["ua1baro"].reshape(out_shape_xy)),
                "ua2baro": (out_dims_xy, out_fields["ua2baro"].reshape(out_shape_xy)),
                "ep1baro": (out_dims_xy, out_fields["ep1baro"].reshape(out_shape_xy)),
                "ep2baro": (out_dims_xy, out_fields["ep2baro"].reshape(out_shape_xy)),
                "ep3baro": (out_dims_xy, out_fields["ep3baro"].reshape(out_shape_xy)),
                "ep4": (out_dims_xy, out_fields["ep4"].reshape(out_shape_xy)),
                "astar1": (out_dims_xyh, out_fields["astar1"].reshape(out_shape_xyh)),
                "astar2": (out_dims_xyh, out_fields["astar2"].reshape(out_shape_xyh)),
            },
            coords={
                **self._other_coords,
                "height": _field.height,
                "ylat": _field.ylat[-_nlat:],
                "xlon": _field.xlon,
            },
            attrs=self.attrs
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

    Example
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
