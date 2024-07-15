"""
------------------------------------------
File name: netcdf_utils.py
Author: Clare Huang
"""
from typing import List
import xarray as xr


def extract_one_timeslice(nc_file: str, output_fname: str, timeslice : List[int] = [0]) -> None:
    """
    Extract specific snapshots (in time) from large netCDF files and save it to another netCDF file.

    Parameters
    ----------
    nc_file : str
        netCDF file that contains the time slice to be extracted
    output_fname : str
        filename of the netCDF file that contains the time slice extracted
    timeslice : List[int], optional
        time index of slice to be extracted
    """
    dataset = xr.open_dataset(nc_file).isel(time=timeslice)
    print(f"Outputing {output_fname} with 1 timeslice.")
    dataset.to_netcdf(output_fname)
    print(f"Finished saving {output_fname} with 1 timeslice.")


if __name__ == "__main__":
    extract_one_timeslice(nc_file="cesm_mdtfv3_timeslice.U.1hr.nc", output_fname="cesm_1tslice.nc")

