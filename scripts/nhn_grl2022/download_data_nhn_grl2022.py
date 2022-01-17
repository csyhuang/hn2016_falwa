"""
Download data to reproduce plots in Neal et al. (submitted to GRL)
"""

import json
from datetime import date, datetime
from calendar import monthrange
import cdsapi


def download_era5_pressure_level_data(filename):
    cdsapi_client = cdsapi.Client()
    all_pressure_level = [
        '1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200',
        '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775',
        '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    variable_names = [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"]

    cdsapi_client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'pressure_level': all_pressure_level,
            'variable': variable_names,
            'time': ["00:00", "06:00", "12:00", "18:00"],
            'grid': "1.0/1.0",
            'product_type': 'reanalysis',
            'year': '2021',
            'day': [f'{i}' for i in range(20, 31)],
            'month': '6',
            'format': 'netcdf'
        },
        filename
    )


def download_single_level_data(self, filename):
    self._cdsapi_client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ["2m_temperature"],
            'time': ["00:00", "06:00", "12:00", "18:00"],
            'grid': "1.0/1.0",
            'product_type': 'reanalysis',
            'year': '2021',
            'day': [f'{i}' for i in range(20, 31)],
            'month': '6',
            'format': 'netcdf'
        },
        filename
    )


if __name__ == "__main__":
    download_era5_pressure_level_data("pressure_level_data_20210620to30.nc")
