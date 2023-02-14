import numpy as np
import xarray as xr
import datetime

from plot_utils import compare_two_fields

two_or_three = '2'

# noboru_data_dir_path = "/Users/claresyhuang/Dropbox/GitHub/hn2016_falwa/github_data_storage/20230201_check_values/"
# noboru_ep2a = xr.open_dataset(noboru_data_dir_path + f'2021_06_ep{two_or_three}a_N.nc')
#
# clare_data_dir_path = "/Users/claresyhuang/Dropbox/GitHub/hn2016_falwa/scripts/nhn_grl2022/"
# clare_file = Dataset(clare_data_dir_path + '2021-06-01_to_2021-06-30_output.nc')

xlon = np.arange(0, 360)
ylat = np.arange(0, 91)

tstamps = [datetime.datetime(2021,6,1,0,0) + datetime.timedelta(hours=6)*i for i in range(124)]

if __name__ == "__main__":
    # for tstep in range(0, 120, 10):
    noboru_data_dir_path = "/Users/claresyhuang/Dropbox/GitHub/hn2016_falwa/github_data_storage/"
    noboru_ep2a = xr.open_dataset(noboru_data_dir_path + f'2021_06_ep2a_N.nc')
    noboru_ep2 = xr.open_dataset(noboru_data_dir_path + f'2021_06_ep2_N.nc')
    noboru_ep3a = xr.open_dataset(noboru_data_dir_path + f'2021_06_ep3a_N.nc')
    noboru_ep3 = xr.open_dataset(noboru_data_dir_path + f'2021_06_ep3_N.nc')

    tstep = 20
    field_a = noboru_ep2a.isel({'time': tstep}).variables[f'ep2a'].values  # (91, 360)
    field_b = noboru_ep2.isel({'time': tstep}).variables[f'ep2'].values  # (91, 360)
    compare_two_fields(field_a=field_a, field_b=field_b, a_title='ep2a', b_title='ep2', x_coord=xlon, y_coord=ylat,
                       title='diff in ep2a and ep2', tstamp=tstamps[tstep])

    field_a = noboru_ep3a.isel({'time': tstep}).variables[f'ep3a'].values  # (91, 360)
    field_b = noboru_ep3.isel({'time': tstep}).variables[f'ep3'].values  # (91, 360)
    compare_two_fields(field_a=field_a, field_b=field_b, a_title='ep3a', b_title='ep3', x_coord=xlon, y_coord=ylat,
                       title='diff in ep3a and ep3', tstamp=tstamps[tstep])
