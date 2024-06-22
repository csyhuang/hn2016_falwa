import itertools
import numpy as np


def calculate_covariance(var_a, var_b):
    """
    Calculate covariance of two variables in time.
    Args:
        var_a: a numpy array or handle that can access elements via [time, lat, lon]
        var_b: a numpy array or handle that can access elements via [time, lat, lon]
    Returns:
        cov_map in dimension of (lat, lon)
    """
    lat_dim = var_a.shape[1]
    lon_dim = var_a.shape[2]
    cov_map = np.zeros((lat_dim, lon_dim))
    for j in range(lat_dim):  # has to loop through a dimension to conserve memory
        cov_matrix = np.cov(m=var_a[:, j, :], y=var_b[:, j, :], rowvar=False)
        row_cov = np.diagonal(cov_matrix, offset=lon_dim)
        cov_map[j, :] = row_cov
    return cov_map
