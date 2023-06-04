import pytest

import numpy as np
try:
    import xarray as xr
except ImportError:
    pytest.skip("Optional package Xarray is not installed.", allow_module_level=True)

from hn2016_falwa.oopinterface import QGFieldNH18, QGFieldNHN22
from hn2016_falwa.xarrayinterface import QGDataset
from hn2016_falwa.xarrayinterface import _is_ascending, _is_descending, _is_equator
from hn2016_falwa.xarrayinterface import _get_name, _map_collect


def _generate_test_dataset(**additional_coords):
    from .test_oopinterface import xlon, ylat, plev, u_field, v_field, t_field
    dims = ("plev", "ylat", "xlon")
    return xr.Dataset(
        data_vars={ "u": (dims, u_field), "v": (dims, v_field), "t": (dims, t_field) },
        coords={ "plev": plev, "ylat": ylat, "xlon": xlon, **additional_coords }
    )


def test_qgdataset_with_dataset():
    data = _generate_test_dataset()
    qgds = QGDataset(data)
    # Make sure all computation functions run
    qgds.interpolate_fields()
    qgds.compute_reference_states()
    qgds.compute_lwa_and_barotropic_fluxes()

def test_qgdataset_with_dataarray():
    data = _generate_test_dataset()
    QGDataset(data["u"], data["v"], data["t"])

def test_qgdataset_with_mixed_args():
    data = _generate_test_dataset()
    QGDataset(data[["u", "t"]], da_v=data["v"])

def test_qgdataset_rejects_incomplete():
    data = _generate_test_dataset()
    for var in data:
        with pytest.raises(KeyError):
            QGDataset(data.drop_vars([var]))

def test_qgdataset_with_coordinate_mismatch():
    data = _generate_test_dataset()
    with pytest.raises(AssertionError):
        QGDataset(data["u"], data["v"], data["t"].rename({ "ylat": "latitude" }))

def test_qgdataset_rejects_transposed():
    data = _generate_test_dataset()
    with pytest.raises(AssertionError):
        transposed_data = data.transpose("xlon", "plev", "ylat")
        QGDataset(transposed_data)
    with pytest.raises(AssertionError):
        transposed_data = data.transpose("ylat", "xlon", "plev")
        QGDataset(transposed_data)


def test_qgdataset_flips_ylat():
    data = _generate_test_dataset()
    interp1 = QGDataset(data).interpolate_fields()
    interp2 = QGDataset(data.reindex({"ylat": data["ylat"][::-1]})).interpolate_fields()
    assert np.allclose(interp1["ylat"], interp2["ylat"])

def test_qgdataset_flips_plev():
    data = _generate_test_dataset()
    interp1 = QGDataset(data).interpolate_fields()
    interp2 = QGDataset(data.reindex({"plev": data["plev"][::-1]})).interpolate_fields()
    assert np.allclose(interp1["height"], interp2["height"])


def test_qgdataset_4d():
    data = xr.concat([_generate_test_dataset(time=t) for t in range(3)], dim="time")
    qgds = QGDataset(data)
    interp = qgds.interpolate_fields()
    # Verify that time dimension is preserved
    assert "time" in interp.coords
    assert interp["interpolated_u"].dims == ("time", "height", "ylat", "xlon")
    assert interp["interpolated_u"].shape == (
        data["time"].size,
        qgds.attrs["kmax"],
        data["ylat"].size,
        data["xlon"].size
    )

def test_qgdataset_5d():
    data = xr.concat([
        xr.concat([_generate_test_dataset(time=t, number=n) for t in range(3)], dim="time")
        for n in range(4)
    ], dim="number")
    qgds = QGDataset(data)
    interp = qgds.interpolate_fields()
    # Verify that additional dimensions are preserved
    assert "time" in interp.coords
    assert "number" in interp.coords
    assert interp["interpolated_u"].dims == ("number", "time", "height", "ylat", "xlon")
    assert interp["interpolated_u"].shape == (
        data["number"].size,
        data["time"].size,
        qgds.attrs["kmax"],
        data["ylat"].size,
        data["xlon"].size
    )


@pytest.mark.parametrize("nh_only", [False, True])
@pytest.mark.parametrize("QGField", [QGFieldNH18, QGFieldNHN22])
def test_basic_qgdataset_calls(QGField, nh_only):
    data = _generate_test_dataset()
    qgds = QGDataset(data, qgfield=QGField, qgfield_kwargs={
        "northern_hemisphere_results_only": nh_only
    })
    # Step 1: basic output verification
    out1 = qgds.interpolate_fields()
    np.testing.assert_allclose(out1["qgpv"], qgds.qgpv)
    np.testing.assert_allclose(out1["interpolated_u"], qgds.interpolated_u)
    np.testing.assert_allclose(out1["interpolated_v"], qgds.interpolated_v)
    np.testing.assert_allclose(out1["interpolated_theta"], qgds.interpolated_theta)
    assert "static_stability" in out1 or ("static_stability_n" in out1 and "static_stability_s" in out1)
    # Step 2: basic output verification
    out2 = qgds.compute_reference_states()
    np.testing.assert_allclose(out2["qref"], qgds.qref)
    np.testing.assert_allclose(out2["uref"], qgds.uref)
    np.testing.assert_allclose(out2["ptref"], qgds.ptref)
    # Step 3: basic output verification
    out3 = qgds.compute_lwa_and_barotropic_fluxes()
    np.testing.assert_allclose(out3["adv_flux_f1"], qgds.adv_flux_f1)
    np.testing.assert_allclose(out3["adv_flux_f2"], qgds.adv_flux_f2)
    np.testing.assert_allclose(out3["adv_flux_f3"], qgds.adv_flux_f3)
    np.testing.assert_allclose(out3["convergence_zonal_advective_flux"], qgds.convergence_zonal_advective_flux)
    np.testing.assert_allclose(out3["divergence_eddy_momentum_flux"], qgds.divergence_eddy_momentum_flux)
    np.testing.assert_allclose(out3["meridional_heat_flux"], qgds.meridional_heat_flux)
    np.testing.assert_allclose(out3["lwa_baro"], qgds.lwa_baro)
    np.testing.assert_allclose(out3["u_baro"], qgds.u_baro)
    np.testing.assert_allclose(out3["lwa"], qgds.lwa)


# Tests for internals

def test_is_ascending():
    assert _is_ascending([4])
    assert _is_ascending([0, 1, 2, 3, 4, 5])
    assert _is_ascending([-2.0, 0.3, 0.5, 1.2])
    assert not _is_ascending([0, -1])
    assert not _is_ascending([0, 2, 3, 3, 3, 2.9])
    assert not _is_ascending([-1, -2., -3.])

def test_is_descending():
    assert _is_descending([4])
    assert not _is_descending([0, 1, 2, 3, 4, 5])
    assert not _is_descending([-2.0, 0.3, 0.5, 1.2])
    assert _is_descending([0, -1])
    assert not _is_descending([0, 2, 3, 3, 3, 2.9])
    assert _is_descending([-1, -2., -3.])

def test_is_equator():
    assert _is_equator(0.)
    assert _is_equator(-0.)
    assert not _is_equator(180.)
    assert not _is_equator(90.)
    assert not _is_equator(-90.)


def test_get_name():
    ds = xr.Dataset(
        data_vars={ "foo": (("x"), np.ones(100)), "bar": (("x"), np.zeros(100)) },
        coords={ "x": np.arange(100), "y": np.arange(100) + 2 }
    )
    # Names from list are found
    assert _get_name(ds, ["foo"], None) == "foo"
    assert _get_name(ds, ["baz", "bar"], {}) == "bar"
    with pytest.raises(KeyError):
        _get_name(ds, ["baz"], None) == "foo"
    # Override baz with foo
    assert _get_name(ds, ["baz", "bar"], { "baz": "foo" }) == "foo"
    # Override trumps existence of name in dataset
    assert _get_name(ds, ["bar", "baz"], { "bar": "foo" }) == "foo"
    # Override bar with x, ignore baz since it is not first entry
    assert _get_name(ds, ["bar", "baz"], { "baz": "foo", "bar": "x" }) == "x"
    # Override fails because first name from list has to be specified
    with pytest.raises(KeyError):
        _get_name(ds, ["xyz", "baz"], { "baz": "foo", "bar": "x" })
    # Bad override fails
    with pytest.raises(KeyError):
        _get_name(ds, ["bar", "baz"], { "bar": "xyz", "baz": "foo" })


def test_map_collect():
    out = _map_collect(
        lambda x: (x, x*2, x**2), # function with 3 return values
        [i * np.ones(3) for i in range(4)], # applied to 4 numpy arrays
        ["foo", "bar", "baz"], # collect return values under these names
        np.asarray # and convert each collected output to a numpy array
    )
    assert out["foo"].shape == (4, 3)
    assert out["bar"].shape == (4, 3)
    assert out["baz"].shape == (4, 3)
    assert np.all(out["foo"] == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.all(out["bar"] == [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]])
    assert np.all(out["baz"] == [[0, 0, 0], [1, 1, 1], [4, 4, 4], [9, 9, 9]])

