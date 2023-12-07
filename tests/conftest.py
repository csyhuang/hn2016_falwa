import pytest
import os
import xarray as xr


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """
    Return the path of directory of data storage for unit tests
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/data"


@pytest.fixture(scope="module")
def offgrid_intput_data(test_data_dir):
    """
    Return dataset with latitude grids not including equator.
    """
    return xr.open_dataset(f"{test_data_dir}/offgrid_input_uvt_data.nc")
