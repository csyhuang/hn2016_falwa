#This script reads in netCDF data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2022_data/"
filename1 = data_dir + "2021_06_tcw.nc"     # total column water (kg/m^2)
filename2 = data_dir + "2021_06_tcwv.nc"   # total column water vapor (kg/m^2)
filename3 = data_dir + "2021_06_sp.nc"   # sea level pressure (hPa)

ncin1 = Dataset(filename1, 'r', format='NETCDF4')
ncin2 = Dataset(filename2, 'r', format='NETCDF4')
ncin3 = Dataset(filename3, 'r', format='NETCDF4')
cw = ncin1.variables['tcw'] 
cw = (np.array(cw))
cwv = ncin2.variables['tcwv'] 
cwv = (np.array(cwv))
sp = ncin3.variables['sp'] 
sp = (np.array(sp))

tt = np.zeros((44,181,360))
pp = np.zeros((44,181,360))
t = np.zeros((181,360))
p = np.zeros((181,360))
m = 77
while(m < 120):
    tt[m-77,:,:] = cw[m,:,:]-cwv[m,:,:]
    pp[m-77,:,:] = sp[m,:,:]/100.
    m = m+1
day = ['06 UTC 20 June 2021','06 UTC 21 June 2021','06 UTC 22 June 2021','06 UTC 23 June 2021','06 UTC 24 June 2021','06 UTC 25 June 2021','06 UTC 26 June 2021','06 UTC 27 June 2021','06 UTC 28 June 2021','06 UTC 29 June 2021','06 UTC 30 June 2021']
b = ['CW_0620.png','CW_0621.png','CW_0622.png','CW_0623.png','CW_0624.png','CW_0625.png','CW_0626.png','CW_0627.png','CW_0628.png','CW_0629.png','CW_0630.png']
n = 0
while(n < 11):
    nn = n*4
    j = 0
    while(j < 181):
        t[j,:]=tt[nn,180-j,:]
        p[j,:]=pp[nn,180-j,:]
        j = j+1
    cl1 = np.arange(-0.1,3.6,0.1)
    c12 = np.arange(980,1032,4)
    x = np.arange(0,360)
    y = np.arange(0,181)-90.
    fig = plot.figure(figsize=(8,4))
    ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plot.xlim(140,280) 
    plot.ylim(10,80) 
    plot.title('Column water  '+day[n]) 
    plot.xlabel('Longitude')
    plot.ylabel('Latitude') 
    ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax5.coastlines(color='white',alpha = 0.7)
    ax5.set_aspect('auto', adjustable=None)
    ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    ott = ax5.contourf(x,y,t,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow') 
    fig.colorbar(ott,ax=ax5,label='kg/m$^2$')
    ott = ax5.contour(x,y,p,levels=c12,colors='white',alpha=0.5,transform=ccrs.PlateCarree(),linewidths=1) 
    ax5.clabel(ott, ott.levels,fmt='%5i')
#    ott = ax5.contour(x,y,z,levels=cl1,colors='black',transform=ccrs.PlateCarree(),linewidths=1) 
#    ax5.clabel(ott, ott.levels,fmt='%5i')
    plot.savefig(b[n],bbox_inches='tight',dpi =600)
    # plot.show()
    
    n = n+1