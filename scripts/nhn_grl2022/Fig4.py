## This script reads in netCDF LWAb data for ERA5
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plot


dt = 3600.*6.
a = 6378000.
dl = 2.*np.pi/360.
dp = 2.*np.pi/360.

jm = 49
jp = 49

#-----------------
# read netCDF files
#-----------------

data_dir = "grl2022_data/"
# filename1 = data_dir + "2021_06_LWAb_N.nc"
filename1 = "2021-06-01_to_2021-06-30_output3.nc"

ncin1 = Dataset(filename1, 'r', format='NETCDF4')

lwa = ncin1.variables['lwa'] 
lwa = (np.array(lwa))

print(lwa.shape)

z = np.zeros((124,360))  # wave activity tendency
w = np.zeros((124,360))  # wave activity
m = 1
while(m < 119):
    j = jm
    cos0 = 0
    while(j < jp+1):
        phi = dp*j
        z[m,:] = z[m,:]+np.cos(phi)*(lwa[m+1,j,:]-lwa[m-1,j,:])/(2.*dt)   # wave activity tendency at 49 N
        w[m,:] = w[m,:]+np.cos(phi)*lwa[m,j,:]
        cos0 = cos0 + np.cos(phi)
        j = j+1
    z[m,:] = z[m,:]/cos0
    w[m,:] = w[m,:]/cos0
    m = m+1
    
filename1 = data_dir + "2021_06_ua1_N.nc"
filename2 = data_dir + "2021_06_ua2_N.nc"
filename3 = data_dir + "2021_06_ep1_N.nc"
filename4 = data_dir + "2021_06_ep2_N.nc"
filename5 = data_dir + "2021_06_ep3_N.nc"
filename6 = data_dir + "2021_06_ep4_N.nc"
# filename1 = "2021-06-01_to_2021-06-30_output2.nc"
# filename2 = "2021-06-01_to_2021-06-30_output2.nc"
# filename3 = "2021-06-01_to_2021-06-30_output2.nc"
# filename4 = "2021-06-01_to_2021-06-30_output2.nc"
# filename5 = "2021-06-01_to_2021-06-30_output2.nc"
# filename6 = "2021-06-01_to_2021-06-30_output2.nc"

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

f1 = np.zeros((124,360))

z1 = np.zeros((124,360))
z2 = np.zeros((124,360))
z3 = np.zeros((124,360))
z4 = np.zeros((124,360))

#phi = dp*49.
#const = 1./(2.*a*np.cos(phi)*dl)
i = 1
while(i < 359):
    j = jm
    while(j < jp+1):
        phi = dp*j
        const = 1./(2.*a*dl)
        z1[:,i] = z1[:,i]-const*(ua1[:,j,i+1]+ua2[:,j,i+1]+ep1[:,j,i+1]-ua1[:,j,i-1]-ua2[:,j,i-1]-ep1[:,j,i-1])
        j = j+1
    z1[:,i] = z1[:,i]/cos0
    i = i+1
i = 0
while(i < 360):
    j = jm
    while(j < jp+1):
        phi = dp*j
        const = 1./(2.*a*dl)
        f1[:,i]=f1[:,i]+(ua1[:,j,i]+ua2[:,j,i]+ep1[:,j,i])*np.cos(phi)   # zonal wave activity flux
        z2[:,i]=z2[:,i]+const*(ep2[:,j,i]-ep3[:,j,i])         # metisional convergence of wave activity flux
        z3[:,i]=z3[:,i]+np.cos(phi)*ep4[:,j,i]                          # bottom source
        j = j+1
    f1[:,i] = f1[:,i]/cos0
    z2[:,i] = z2[:,i]/cos0
    z3[:,i] = z3[:,i]/cos0    
    i = i+1
    
zs = np.zeros((124,360))  # smoothed z #
ws = np.zeros((124,360))  # smoothed w #
f1s = np.zeros((124,360))  # smoothed f1 #
z1s = np.zeros((124,360))  # smoothed z1 #
z2s = np.zeros((124,360))  # smoothed z2 #
z3s = np.zeros((124,360))  # smoothed z3 #

#### smoothing in longitude ####
n = 4   # smoothing width = 2n+1
j = 0
while(j < 124):
    zx = np.zeros(360)
    zx[:] = z[j,:]
    wx = np.zeros(360)
    wx[:] = w[j,:]
    f1x = np.zeros(360)
    f1x[:] = f1[j,:]
    z1x = np.zeros(360)
    z1x[:] = z1[j,:]
    z2x = np.zeros(360)
    z2x[:] = z2[j,:]
    z3x = np.zeros(360)
    z3x[:] = z3[j,:]
    nn = -n
    while(nn < n+1):
        wy = np.roll(wx,nn)
        ws[j,:] = ws[j,:] + wy[:]/(2*n+1)
        f1y = np.roll(f1x,nn)
        f1s[j,:] = f1s[j,:] + f1y[:]/(2*n+1)
        zy = np.roll(zx,nn)
        zs[j,:] = zs[j,:] + zy[:]/(2*n+1)
        z1y = np.roll(z1x,nn)
        z1s[j,:] = z1s[j,:] + z1y[:]/(2*n+1)
        z2y = np.roll(z2x,nn)
        z2s[j,:] = z2s[j,:] + z2y[:]/(2*n+1)
        z3y = np.roll(z3x,nn)
        z3s[j,:] = z3s[j,:] + z3y[:]/(2*n+1)
        nn = nn+1   
    j = j+1
z4[:,:]=zs[:,:]-z1s[:,:]-z2s[:,:]-z3s[:,:]  # residual



##############################################

cl2 = np.arange(0,135,5)
x = np.arange(0,360)
y = np.arange(0,124)*0.25
plot.rcParams.update({'font.size': 20})
fig,ax5 = plot.subplots(1,figsize=(8,8)) 
plot.xlim(140,280) 
plot.ylim(20,29.75) 
plot.title('Column LWA 49$^\circ$N') 
plot.xlabel('Longitude')
plot.ylabel('Day') 
ott = ax5.contourf(x,y,w,levels=cl2,cmap='rainbow')
fig.colorbar(ott,ax=ax5,label='(ms$^{-1}$)')
plot.savefig('HovmollerLWA.png',bbox_inches='tight',dpi =600)
#plot.show()


##############################################

cl2 = np.arange(-100,1500,100)
x = np.arange(0,360)
y = np.arange(0,124)*0.25
plot.rcParams.update({'font.size': 20})
fig,ax5 = plot.subplots(1,figsize=(8,8))    
plot.xlim(140,280) 
plot.ylim(20,29.75) 
plot.title('<F$_\lambda$> 49$^\circ$N') 
plot.xlabel('Longitude')
plot.ylabel('Day') 
ott = ax5.contourf(x,y,f1,levels=cl2,cmap='rainbow')
fig.colorbar(ott,ax=ax5,label='(m$^2$s$^{-2}$)')
#ott = ax5.contourf(x,y,z4,levels=[0.0005,0.001])
plot.savefig('HovmollerFx.png',bbox_inches='tight',dpi =600)
#plot.show()


##############################################

cl2 = np.arange(-10.,9,1)
x = np.arange(0,360)
y = np.arange(0,124)*0.25
plot.rcParams.update({'font.size': 20})
fig,ax5 = plot.subplots(1,figsize=(8,8))    
plot.xlim(140,280) 
plot.ylim(20,29.5) 
plot.title('Term (IV) 49$^\circ$N') 
plot.xlabel('Longitude')
plot.ylabel('Day') 
ott = ax5.contourf(x,y,z4*10000,levels=cl2,cmap='rainbow')
fig.colorbar(ott,ax=ax5,label='(10$^{-4}$ ms$^{-2}$)')
plot.savefig('HovmollerRes.png',bbox_inches='tight',dpi =600)
#plot.show()


##############################################

cl2 = np.arange(-10.,9,1)
x = np.arange(0,360)
y = np.arange(0,124)*0.25
plot.rcParams.update({'font.size': 20})
fig,ax5 = plot.subplots(1,figsize=(8,8))    
plot.xlim(140,280) 
plot.ylim(20,29.75) 
plot.title('Terms (I)+(II) 49$^\circ$N') 
plot.xlabel('Longitude')
plot.ylabel('Day') 
ott = ax5.contourf(x,y,(z1s+z2s+z3s)*10000,levels=cl2,cmap='rainbow')
fig.colorbar(ott,ax=ax5,label='(10$^{-4}$ ms$^{-2}$)')
plot.savefig('HovmollerFxy.png',bbox_inches='tight',dpi =600)
#plot.show()