#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

param_u, param_v, param_t = "131.128", "132.128", "130.128"


for year in range(2000,2013):
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "{}-12-01/to/{}-03-31".format(year, year+1),
        "expver": "1",
        "grid": "1.5/1.5",
        "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "levtype": "pl",
        "param": '/'.join([param_u, param_v, param_t]),
        "step": "0",
        "stream": "oper",
        "format": "netcdf",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "target": "{}-12-01_to_{}-03-31_uvt.nc".format(year, year+1),
    })
