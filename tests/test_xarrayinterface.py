import pytest

import numpy as np
import xarray as xr

from hn2016_falwa.xarrayinterface import QGDataset
from hn2016_falwa.xarrayinterface import _is_ascending, _is_descending, _is_equator
from hn2016_falwa.xarrayinterface import _get_name, _map_collect


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
        lambda x: (x, x*2, x**2),
        [i * np.ones(3) for i in range(4)],
        ["foo", "bar", "baz"],
        np.asarray
    )
    assert out["foo"].shape == (4, 3)
    assert out["bar"].shape == (4, 3)
    assert out["baz"].shape == (4, 3)
    assert np.all(out["foo"] == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.all(out["bar"] == [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]])
    assert np.all(out["baz"] == [[0, 0, 0], [1, 1, 1], [4, 4, 4], [9, 9, 9]])

