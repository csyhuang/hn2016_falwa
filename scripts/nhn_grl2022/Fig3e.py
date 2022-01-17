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
filename1 = data_dir + "2021_06_mtnlwrf.nc"     # net OLR
filename2 = data_dir + "2021_06_mtnlwrfcs.nc"   # OLR clear sky

ncin1 = Dataset(filename1, 'r', format='NETCDF4')
ncin2 = Dataset(filename2, 'r', format='NETCDF4')
olr = ncin1.variables['mtnlwrf'] 
olr = (np.array(olr))
olrcs = ncin2.variables['mtnlwrfcs'] 
olrcs = (np.array(olrcs))


print(olr.shape)

tt = np.zeros((44,181,360))
t = np.zeros((181,360))
m = 77
while(m < 120):
    tt[m-77,:,:] = olr[m,:,:]
    m = m+1
day = ['06 UTC 20 June 2021','06 UTC 21 June 2021','06 UTC 22 June 2021','06 UTC 23 June 2021','06 UTC 24 June 2021','06 UTC 25 June 2021','06 UTC 26 June 2021','06 UTC 27 June 2021','06 UTC 28 June 2021','06 UTC 29 June 2021','06 UTC 30 June 2021']
b = ['OLR_0620.png','OLR_0621.png','OLR_0622.png','OLR_0623.png','OLR_0624.png','OLR_0625.png','OLR_0626.png','OLR_0627.png','OLR_0628.png','OLR_0629.png','OLR_0630.png']
n = 0
while(n < 11):
    nn = n*4
    j = 0
    while(j < 181):
        t[j,:]=tt[nn,180-j,:]
        j = j+1
    cl1 = np.arange(80,390,10)
    x = np.arange(0,360)
    y = np.arange(0,181)-90.
    plot.rcParams.update({'font.size':14,'text.usetex': False})
    fig = plot.figure(figsize=(8,4))
    ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plot.xlim(140,280) 
    plot.ylim(10,80) 
    plot.title('OLR  '+day[n]) 
    plot.xlabel('Longitude')
    plot.ylabel('Latitude') 
    ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax5.coastlines(color='black',alpha = 0.7)
    ax5.set_aspect('auto', adjustable=None)
    ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    ott = ax5.contourf(x,y,-t,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow') 
    fig.colorbar(ott,ax=ax5,label='W/m$^2$')
#    ott = ax5.contour(x,y,z,levels=cl1,colors='black',transform=ccrs.PlateCarree(),linewidths=1) 
#    ax5.clabel(ott, ott.levels,fmt='%5i')
    plot.savefig(b[n],bbox_inches='tight',dpi =600)
    
    n = n+1