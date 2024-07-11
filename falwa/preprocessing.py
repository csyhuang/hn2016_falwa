"""
------------------------------------------
File name: preprocessing.py
Author: Clare Huang
"""
import numpy as np
import gridfill


def gridfill_each_level(lat_lon_field, itermax=1000, verbose=False):
    """
    Fill missing values in lat-lon grids with values derived by solving Poisson's equation
    using a relaxation scheme.

    Args:
        lat_lon_field(np.ndarray): 2D array to apply gridfill on
        itermax(int): maximum iteration for poisson solver
        verbose(bool): verbose level of poisson solver

    Returns:
        A 2D array of the same dimension with all nan filled.
    """
    if np.isnan(lat_lon_field).sum() == 0:
        return lat_lon_field

    lat_lon_filled, converged = gridfill.fill(
        grids=np.ma.masked_invalid(lat_lon_field), xdim=1, ydim=0, eps=0.01,
        cyclic=True, itermax=itermax, verbose=verbose)

    return lat_lon_filled

