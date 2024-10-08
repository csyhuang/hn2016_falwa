#!/usr/bin/env python
import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-pressure-levels"

params = {
    "_t": "temperature",
    "_u": "u_component_of_wind",
    "_v": "v_component_of_wind"
}

for param_string, param in params.items():
    request = {
        "product_type": ["reanalysis"],
        "variable": [param],
        "year": ["2005"],
        "month": ["01"],
        "day": [
            "23", "24", "25",
            "26", "27", "28",
            "29", "30"
        ],
        "time": [
            "00:00", "06:00", "12:00",
            "18:00"
        ],
        "pressure_level": [
            "1", "2", "3",
            "5", "7", "10",
            "20", "30", "50",
            "70", "100", "125",
            "150", "175", "200",
            "225", "250", "300",
            "350", "400", "450",
            "500", "550", "600",
            "650", "700", "750",
            "775", "800", "825",
            "850", "875", "900",
            "925", "950", "975",
            "1000"
        ],
        "grid": "1.5/1.5",
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    filename = "2005-01-23_to_2005-01-30" + param_string + ".nc"
    client.retrieve(dataset, request).download(filename)

