## This script reads in netCDF LWAb data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2022_data/"
# filename0 = data_dir + "2021_06_LWAb_N.nc"
filename0 = "2021-06-01_to_2021-06-30_output3.nc"

ncin1 = Dataset(filename0, 'r', format='NETCDF4')

lwa = ncin1.variables['lwa'] 
lwa = (np.array(lwa))

#print(lwa.shape)

z = np.zeros((91,360))
z[:,:] = lwa[100,:,:]-lwa[76,:,:]   # m = 100 is 00 UTC 26 June 2021, m = 76 is 00 UTC 20 June 2021

zs = np.zeros((91,360))  # smoothed z #

#### smoothing in longitude ####
n = 5     # smoothing width #
j = 0
while(j < 91):
    zx = np.zeros(360)
    zx[:] = z[j,:]
    nn = -n
    while(nn < n+1):
        zy = np.roll(zx,nn)
        zs[j,:] = zs[j,:] + zy[:]/(2*n+1)
        nn = nn+1   
    j = j+1
    

cl2 = np.arange(-80,90,10)
x = np.arange(0,360)
y = np.arange(0,91)
plot.rcParams.update({'font.size':14})
fig = plot.figure(figsize=(8,4))
ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
plot.xlim(140,280) 
plot.ylim(10,80) 
plot.title('Column LWA Change  June 20 - 26') 
plot.xlabel('Longitude')
plot.ylabel('Latitude') 
ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
ax5.coastlines(alpha = 0.3)
ax5.set_aspect('auto', adjustable=None)
ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax5.xaxis.set_major_formatter(lon_formatter)
ax5.yaxis.set_major_formatter(lat_formatter)
ott = ax5.contourf(x,y,zs,levels=cl2,transform=ccrs.PlateCarree(),cmap='rainbow')
fig.colorbar(ott,ax=ax5,label='LWA (m/s)')
plot.savefig('dLWA.png',bbox_inches='tight',dpi =600)
#plot.show()

filename1 = "2021-06-01_to_2021-06-30_output3.nc"
filename2 = "2021-06-01_to_2021-06-30_output3.nc"
filename3 = "2021-06-01_to_2021-06-30_output3.nc"
filename4 = "2021-06-01_to_2021-06-30_output3.nc"
filename5 = "2021-06-01_to_2021-06-30_output3.nc"
filename6 = "2021-06-01_to_2021-06-30_output3.nc"

# filename1 = data_dir + "2021_06_ua1_N.nc"
# filename2 = data_dir + "2021_06_ua2_N.nc"
# filename3 = data_dir + "2021_06_ep1_N.nc"
# filename4 = data_dir + "2021_06_ep2_N.nc"
# filename5 = data_dir + "2021_06_ep3_N.nc"
# filename6 = data_dir + "2021_06_ep4_N.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')
ua1 = ncin1.variables['ua1'] 
ua1 = (np.array(ua1))
ncin2 = Dataset(filename2, 'r', format='NETCDF4')
ua2 = ncin2.variables['ua2'] 
ua2 = (np.array(ua2))
ncin3 = Dataset(filename3, 'r', format='NETCDF4')
ep1 = ncin3.variables['ep1'] 
ep1 = (np.array(ep1))
ncin4 = Dataset(filename4, 'r', format='NETCDF4')
ep2 = ncin4.variables['ep2'] 
ep2 = (np.array(ep2))
ncin5 = Dataset(filename5, 'r', format='NETCDF4')
ep3 = ncin5.variables['ep3'] 
ep3 = (np.array(ep3))
ncin6 = Dataset(filename6, 'r', format='NETCDF4')
ep4 = ncin6.variables['ep4'] 
ep4 = (np.array(ep4))

f1 = np.zeros((91,360))
f2 = np.zeros((91,360))
f11 = np.zeros((91,360))
f22 = np.zeros((91,360))

z1 = np.zeros((91,360))
z2 = np.zeros((91,360))
z3 = np.zeros((91,360))
dt = 3600.*6.
a = 6378000.
dl = 2.*np.pi/360.
dp = 2.*np.pi/360.
m = 76              # m = 76 is 20 June 2021 00 UTC
while(m < 100):     # m = 100 is 26 June 2021 00 UTC
    z3[:,:] = z3[:,:]+0.5*dt*ep4[m,:,:]
    z3[:,:] = z3[:,:]+0.5*dt*ep4[m+1,:,:]
    f1[:,:] = f1[:,:]+(0.5/24.)*(ua1[m,:,:]+ua2[m,:,:]+ep1[m,:,:])
    f1[:,:] = f1[:,:]+(0.5/24.)*(ua1[m+1,:,:]+ua2[m+1,:,:]+ep1[m+1,:,:])
    f11[:,:] = f11[:,:]+(0.5/24.)*(ua1[m-24,:,:]+ua2[m-24,:,:]+ep1[m-24,:,:])
    f11[:,:] = f11[:,:]+(0.5/24.)*(ua1[m-23,:,:]+ua2[m-23,:,:]+ep1[m-23,:,:])
    j = 0
    while(j < 90):
        phi = dp*j
        const = 0.5*dt/(2.*a*np.cos(phi)*dl)
        z2[j,:]=z2[j,:]+const*(ep2[m,j,:]-ep3[m,j,:])
        z2[j,:]=z2[j,:]+const*(ep2[m+1,j,:]-ep3[m+1,j,:])
        f2[j,:] = f2[j,:]+(0.25/24.)*(ep2[m,j,:]+ep3[m,j,:])/np.cos(phi)
        f2[j,:] = f2[j,:]+(0.25/24.)*(ep2[m+1,j,:]+ep3[m+1,j,:])/np.cos(phi)
        f22[j,:] = f22[j,:]+(0.25/24.)*(ep2[m-24,j,:]+ep3[m-24,j,:])/np.cos(phi)
        f22[j,:] = f22[j,:]+(0.25/24.)*(ep2[m-23,j,:]+ep3[m-23,j,:])/np.cos(phi)
        i = 1
        while(i < 359): 
            z1[j,i] = z1[j,i]-const*(ua1[m,j,i+1]+ua2[m,j,i+1]+ep1[m,j,i+1]-ua1[m,j,i-1]-ua2[m,j,i-1]-ep1[m,j,i-1])
            z1[j,i] = z1[j,i]-const*(ua1[m+1,j,i+1]+ua2[m+1,j,i+1]+ep1[m+1,j,i+1]-ua1[m+1,j,i-1]-ua2[m+1,j,i-1]-ep1[m+1,j,i-1])
            i = i+1
        z1[j,0] = z1[j,0]-const*(ua1[m,j,1]+ua2[m,j,1]+ep1[m,j,1]-ua1[m,j,359]-ua2[m,j,359]-ep1[m,j,359])
        z1[j,0] = z1[j,0]-const*(ua1[m+1,j,1]+ua2[m+1,j,1]+ep1[m+1,j,1]-ua1[m+1,j,359]-ua2[m+1,j,359]-ep1[m+1,j,359])
        z1[j,359] = z1[j,359]-const*(ua1[m,j,0]+ua2[m,j,0]+ep1[m,j,0]-ua1[m,j,358]-ua2[m,j,358]-ep1[m,j,358])
        z1[j,359] = z1[j,359]-const*(ua1[m+1,j,0]+ua2[m+1,j,0]+ep1[m+1,j,0]-ua1[m+1,j,358]-ua2[m+1,j,358]-ep1[m+1,j,358])
        j = j+1
    m = m+1
    
z1s = np.zeros((91,360))  # smoothed z1 #
z2s = np.zeros((91,360))  # smoothed z2 #
z3s = np.zeros((91,360))  # smoothed z3 #

#### smoothing in longitude ####
j = 0
while(j < 91):
    z1x = np.zeros(360)
    z1x[:] = z1[j,:]
    z2x = np.zeros(360)
    z2x[:] = z2[j,:]
    z3x = np.zeros(360)
    z3x[:] = z3[j,:]
    nn = -n
    while(nn < n+1):
        z1y = np.roll(z1x,nn)
        z1s[j,:] = z1s[j,:] + z1y[:]/(2*n+1)
        z2y = np.roll(z2x,nn)
        z2s[j,:] = z2s[j,:] + z2y[:]/(2*n+1)
        z3y = np.roll(z3x,nn)
        z3s[j,:] = z3s[j,:] + z3y[:]/(2*n+1)
        nn = nn+1   
    j = j+1

##### Wind vectors ######

x1 = np.arange(0,24)*15.+5.
y1 = np.arange(0,30)*3.
xx,yy = np.meshgrid(x1,y1)
uu = np.zeros((30,24))
vv = np.zeros((30,24))

j = 0
while(j < 30):
    i = 0
    while(i < 24):
        uu[j,i] = f1[j*3,i*15+5]-f11[j*3,i*15+5]
        vv[j,i] = f2[j*3,i*15+5]-f22[j*3,i*15+5]
        i = i+1
    j = j+1

print(zs[49,242],z1s[49,242],z2s[49,242],z3s[49,242],zs[49,242]-z1s[49,242]-z2s[49,242]-z3s[49,242])
print(z1s[60,248]+z2s[60,248])

cl1 = np.arange(-200,220,20)
x = np.arange(0,360)
y = np.arange(0,91)
plot.rcParams.update({'font.size':14})
fig = plot.figure(figsize=(8,4))
ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
#ax6 = fig.add_subplot(1,1,1)
plot.xlim(0,360) 
plot.ylim(10,80) 
plot.title('Integrated Terms (I)+(II)   June 20 - 26') 
plot.xlabel('Longitude')
plot.ylabel('Latitude') 
ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
ax6.coastlines(alpha = 0.3)
ax6.set_aspect('auto', adjustable=None)
ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax6.xaxis.set_major_formatter(lon_formatter)
ax6.yaxis.set_major_formatter(lat_formatter)
ott = ax6.contourf(x,y,z1s+z2s-0.1,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
fig.colorbar(ott,ax=ax6,label='(m/s)')
ax6.quiver(xx,yy,uu,vv,transform=ccrs.PlateCarree())
plot.savefig('divFx+Fy.png',bbox_inches='tight',dpi =600)
#plot.show()
    
cl1 = np.arange(-200,220,20)
x = np.arange(0,360)
y = np.arange(0,91)
plot.rcParams.update({'font.size':14})
fig = plot.figure(figsize=(8,4))
ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
plot.xlim(0,360) 
plot.ylim(10,80) 
plot.title('Integrated Term (III)   June 20 - 26') 
plot.xlabel('Longitude')
plot.ylabel('Latitude') 
ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
ax6.coastlines(alpha = 0.3)
ax6.set_aspect('auto', adjustable=None)
ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax6.xaxis.set_major_formatter(lon_formatter)
ax6.yaxis.set_major_formatter(lat_formatter)
ott = ax6.contourf(x,y,z3s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
fig.colorbar(ott,ax=ax6,label='(m/s)')
#ax6.quiver(xx,yy,uu,vv,transform=ccrs.PlateCarree())
plot.savefig('EP4.png',bbox_inches='tight',dpi =600)
#plot.show()

cl1 = np.arange(-200,220,20)
x = np.arange(0,360)
y = np.arange(0,91)
plot.rcParams.update({'font.size':14})
fig = plot.figure(figsize=(8,4))
ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
plot.xlim(0,360) 
plot.ylim(10,80) 
plot.title('Integrated Term (IV)   June 20 - 26') 
plot.xlabel('Longitude')
plot.ylabel('Latitude') 
ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
ax6.coastlines(alpha = 0.3)
ax6.set_aspect('auto', adjustable=None)
ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax6.xaxis.set_major_formatter(lon_formatter)
ax6.yaxis.set_major_formatter(lat_formatter)
ott = ax6.contourf(x,y,zs-z1s-z2s-z3s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
fig.colorbar(ott,ax=ax6,label='(m/s)')
ax6.quiver(xx,yy,uu,vv,transform=ccrs.PlateCarree())
plot.savefig('Residual.png',bbox_inches='tight',dpi =600)
#plot.show()



