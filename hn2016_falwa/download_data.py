#!/usr/bin/env python
from enum import Enum
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()


class ERAICode(Enum):
    u = '131.128' # zonal wind
    v = '132.128' # meridional wind
    t = '130.128' # temperature

def retrieve_erai(start_date, end_date, file_suffix, var_list):
    """
    Args:
        start_date(datetime.date)
        start_date(datetime.date)
        file_suffix(str)
        var_list(list)

    Return:
        A boolean indicating whether the download is successful
    """

    start_date_str = start_date.strftime(format='%Y-%m-%d')
    end_date_str = end_date.strftime(format='%Y-%m-%d')
    fname = "{}_to_{}_{}.nc".format(start_date_str, end_date_str, file_suffix)
    param_str = "/".join([x.value for x in var_list])

    try:
        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": "{}/to/{}".format(start_date_str, end_date_str),
            "expver": "1",
            "grid": "1.5/1.5",
            "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
            "levtype": "pl",
            "param": param_str,
            "step": "0",
            "stream": "oper",
            "format": "netcdf",
            "time": "00:00:00/06:00:00/12:00:00/18:00:00",
            "type": "an",
            "target": fname,
        })
        print('Finished downloading {}'.format(fname))
        return True
    except:
        print('Failed downloading {}'.format(fname))
        return False


