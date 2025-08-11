"""
------------------------------------------
File name: stat_utils.py
Author: Clare Huang
"""
import itertools
import numpy as np


def calculate_covariance(var_a: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """Calculate covariance of two variables in time.

    Parameters
    ----------
    var_a : np.ndarray
        A numpy array with dimensions (time, lat, lon).
    var_b : np.ndarray
        A numpy array with dimensions (time, lat, lon).

    Returns
    -------
    np.ndarray
        Covariance map with dimensions (lat, lon).
    """
    lat_dim = var_a.shape[1]
    lon_dim = var_a.shape[2]
    cov_map = np.zeros((lat_dim, lon_dim))
    for j in range(lat_dim):  # has to loop through a dimension to conserve memory
        cov_matrix = np.cov(m=var_a[:, j, :], y=var_b[:, j, :], rowvar=False)
        row_cov = np.diagonal(cov_matrix, offset=lon_dim)
        cov_map[j, :] = row_cov
    return cov_map
