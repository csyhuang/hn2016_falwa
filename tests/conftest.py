import pytest
import os


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """
    Return the path of directory of data storage for unit tests
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/data"


