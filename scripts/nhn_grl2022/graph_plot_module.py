"""
This module contains plot functions to reproduce the graphs in NHN GRL2021
"""
import numpy as np
from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import pyplot as plt
from netCDF4._netCDF4 import Dataset


# *** Shared variables ***
date_stamp = [f'00 UTC {i} June 2021' for i in range(20, 31)]


def plot_figure1a(z_filename, u_filename, v_filename):
    ncin1 = Dataset(z_filename, 'r', format='NETCDF4')
    ncin2 = Dataset(u_filename, 'r', format='NETCDF4')
    ncin3 = Dataset(v_filename, 'r', format='NETCDF4')

    zmean = ncin1.variables['z']
    zmean = (np.array(zmean))
    umean = ncin2.variables['u']
    umean = (np.array(umean))
    vmean = ncin3.variables['v']
    vmean = (np.array(vmean))

    print(zmean.shape)

    zz = np.zeros((44,181,360))
    z = np.zeros((181,360))
    uu = np.zeros((44,181,360))
    u = np.zeros((181,360))
    vv = np.zeros((44,181,360))
    v = np.zeros((181,360))
    m = 76
    while(m < 120):
        zz[m-76,:,:] = zmean[m,16,:,:]
        uu[m-76,:,:] = umean[m,16,:,:]
        vv[m-76,:,:] = vmean[m,16,:,:]
        m = m+1
    b = [f'06{i}.png' for i in range(20, 31)]
    for n in range(0, 11):
        nn = n*4
        for j in range(0, 181):
            z[j,:]=zz[nn,180-j,:]/9.81
            u[j,:]=uu[nn,180-j,:]
            v[j,:]=vv[nn,180-j,:]
            j = j+1
        cl1 = np.arange(9600,11300,100)
        cl2 = np.arange(0,95,5)
        plt.rcParams.update({'font.size': 15})
        x = np.arange(0,360)
        y = np.arange(0,181)-90.
        plt.rcParams.update({'font.size':15, 'text.usetex': False})
        fig = plt.figure(figsize=(8, 4))
        ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(10, 80)
    #    plot.title(''+day[n])
        if(n > 9):
            plt.xlabel('Longitude', fontsize = 22)
        plt.ylabel('Latitude', fontsize = 22)
        ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
        ax5.coastlines(color='white',alpha = 0.7)
        ax5.set_aspect('auto', adjustable=None)
        if(n > 9):
            ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
        ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax5.xaxis.set_major_formatter(lon_formatter)
        ax5.yaxis.set_major_formatter(lat_formatter)
        ott = ax5.contourf(x,y,np.sqrt(u*u+v*v),levels=cl2,transform=ccrs.PlateCarree(),cmap='rainbow')
        fig.colorbar(ott,ax=ax5,label='wind speed (m/s)')
        ott = ax5.contour(x,y,z,levels=cl1,colors='black',transform=ccrs.PlateCarree(),linewidths=1)
        ax5.clabel(ott, ott.levels,fmt='%5i')
        plt.savefig(b[n], bbox_inches='tight', dpi =600)
        plt.close()


def plot_figure1b(t_filename):
    ncin1 = Dataset(t_filename, 'r', format='NETCDF4')

    tmean = ncin1.variables['t']
    tmean = (np.array(tmean))

    print(tmean.shape)

    tt = np.zeros((44,181,360))
    t = np.zeros((181,360))

    r = 287.
    cp = 1004.
    kappa = r/cp

    for m in range(76, 120):
        tt[m-76,:,:] = tmean[m,20,:,:]*np.power((1000./450.),kappa)

    b = [f'T_06{i}.png' for i in range(20, 31)]

    for n in range(0, 11):
        nn = n*4
        for j in range(0, 181):
            t[j,:]=tt[nn,180-j,:]
        cl1 = np.arange(290,342,2) # 450 hPa
        cl2 = np.arange(0,95,5)
        x = np.arange(0,360)
        y = np.arange(0,181)-90.
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure(figsize=(8, 4))
        ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(10, 80)
        if(n > 9):
            plt.xlabel('Longitude', fontsize = 22)
        ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
        ax5.coastlines(color='black',alpha = 0.7)
        ax5.set_aspect('auto', adjustable=None)
        if(n > 9):
            ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax5.xaxis.set_major_formatter(lon_formatter)
        ax5.yaxis.set_major_formatter(lat_formatter)
        ott = ax5.contourf(x,y,t,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
        fig.colorbar(ott,ax=ax5,label='Kelvin')
        plt.savefig(b[n], bbox_inches='tight', dpi =600)
        plt.close()


def plot_figure1c(t2m_filename):
    ncin1 = Dataset(t2m_filename, 'r', format='NETCDF4')
    tm = ncin1.variables['t2m']
    tm = (np.array(tm))

    print(tm.shape)

    tt = np.zeros((44,181,360))
    t = np.zeros((181,360))

    r = 287.
    cp = 1004.
    kappa = r/cp

    for m in range(76, 120):
        tt[m-76,:,:] = tm[m,:,:]

    b = [f'2T_06{i}.png' for i in range(20, 31)]
    n = 0
    for n in range(0, 11):
        nn = n*4

        for j in range(0, 181):
            t[j,:]=tt[nn,180-j,:]

        cl1 = np.arange(250,325,5)
        cl2 = np.arange(0,95,5)
        plt.rcParams.update({'font.size': 15})
        x = np.arange(0,360)
        y = np.arange(0,181)-90.
        fig = plt.figure(figsize=(8, 4))
        ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(10, 80)
        if(n > 9):
            plt.xlabel('Longitude', fontsize=22)
        ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
        ax5.coastlines(color='black',alpha = 0.7)
        ax5.set_aspect('auto', adjustable=None)
        if(n > 9):
            ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax5.xaxis.set_major_formatter(lon_formatter)
        ax5.yaxis.set_major_formatter(lat_formatter)
        ott = ax5.contourf(x,y,t,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
        fig.colorbar(ott,ax=ax5,label='Kelvin')
        plt.savefig(b[n], bbox_inches='tight', dpi =600)
        plt.close()


def plot_figure1d_2a(t_filename):
    ncin1 = Dataset(t_filename, 'r', format='NETCDF4')

    tmean = ncin1.variables['t']
    tmean = (np.array(tmean))

    ttheta = np.zeros((44, 37, 360))
    theta = np.zeros((37, 360))
    z = np.zeros(37)
    p = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50.,
                  70., 100., 125., 150., 175., 200., 225., 250., 300.,
                  350., 400., 450., 500., 550., 600., 650., 700., 750.,
                  775., 800., 825., 850., 875., 900., 925., 950., 975., 1000.])
    r = 287.
    cp = 1004.
    kappa = r / cp
    for m in range(76, 120):
        for k in range(37):
            z[36 - k] = -8000. * np.log(p[k] / 1000.)  # pseudoheight
            ttheta[m - 76, 36 - k, :] = tmean[m, k, 41, :]
            ttheta[m - 76, 36 - k, :] = ttheta[m - 76, 36 - k, :] * np.power((1000. / p[k]), kappa)  # potential temp

    b = [f'THETA_06{i}.png' for i in range(20, 31)]
    for n in range(11):
        nn = n * 4
        theta[:, :] = ttheta[nn, :, :]
        cl1 = np.arange(250, 365, 5)
        x = np.arange(0, 360)
        plt.rcParams.update({'font.size': 15, 'text.usetex': False})
        fig = plt.figure(figsize=(8, 4))
        ax5 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(0, 10)
        #    plot.title('49$^\circ$N '+day[n])
        if (n > 9):
            plt.xlabel('Longitude', fontsize=22)
        plt.ylabel('pseudoheight (km)', fontsize=22)
        ax5.set_extent([-220, -80, 0, 10], ccrs.PlateCarree())
        ax5.set_aspect('auto', adjustable=None)
        if (n > 9):
            ax5.set_xticks([140, 160, 180, 200, 220, 240, 260, 280], crs=ccrs.PlateCarree())
        ax5.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax5.xaxis.set_major_formatter(lon_formatter)
        ott = ax5.contourf(x, z / 1000., theta, levels=cl1, transform=ccrs.PlateCarree(), cmap='rainbow')
        fig.colorbar(ott, ax=ax5, label='Kelvin')
        ott = ax5.contour(x, z / 1000., theta, levels=cl1, transform=ccrs.PlateCarree(), colors='black', linewidths=0.5)
        ott = ax5.contour(x, z / 1000., theta, levels=np.arange(320, 325, 5), transform=ccrs.PlateCarree(),
                          colors='black', linewidths=1)
        ax5.clabel(ott, ott.levels, fmt='%5i')
        plt.savefig(b[n], bbox_inches='tight', dpi=600)
        plt.close()

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(6, 4))
    ax5.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.title('49$^\circ$N 119$^\circ$W  00 UTC')
    plt.xlabel('Kelvin')
    plt.ylabel('pseudoheight (km)')
    plt.xlim(290, 360)
    plt.ylim(0, 10)

    lcolor = np.array(['blue', 'red', 'green', 'black'])
    lstyle = np.array(['dotted', 'dashed', 'dashdot', 'solid'])
    lalpha = np.array([1, 1, 1, 1])

    for n in range(2, 6):
        thetaz = np.zeros(37)
        nn = n * 8
        thetaz[:] = ttheta[nn, :, 241]

        for i in range(37):
            if (z[i] < 1000.):
                thetaz[i] = np.nan

        fig = plt.plot(thetaz, z / 1000., color=lcolor[n - 2], linestyle=lstyle[n - 2], alpha=lalpha[n - 2])

    plt.savefig('t_profile.png', bbox_inches='tight', dpi=600)
    plt.close()


def plot_figure3_and_S1(lwa_flux_filename):
    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')

    lwa = ncin1.variables['lwa']
    lwa = (np.array(lwa))

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
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 5))
    ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plt.xlim(140, 280)
    plt.ylim(10, 80)
    plt.title('Column LWA Change  00 UTC 20 - 26 June 2021')
    #plot.xlabel('Longitude')
    plt.ylabel('Latitude', fontsize=22)
    ax5.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax5.coastlines(alpha = 0.7)
    ax5.set_aspect('auto', adjustable=None)
    ax5.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax5.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    ott = ax5.contourf(x,y,zs,levels=cl2,transform=ccrs.PlateCarree(),cmap='rainbow')
    fig.colorbar(ott,ax=ax5,label='LWA (m/s)')
    plt.savefig('dLWA_0.png', bbox_inches='tight', dpi =600)

    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ua1 = ncin1.variables['ua1']
    ua1 = (np.array(ua1))
    ncin2 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ua2 = ncin2.variables['ua2']
    ua2 = (np.array(ua2))
    ncin3 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep1 = ncin3.variables['ep1']
    ep1 = (np.array(ep1))
    ncin4 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep2 = ncin4.variables['ep2']
    ep2 = (np.array(ep2))
    ncin5 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep3 = ncin5.variables['ep3']
    ep3 = (np.array(ep3))
    ncin6 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
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
    #m = 52              # m = 52 is 14 June 2021 00 UTC
    #while(m < 76):     # m = 76 is 20 June 2021 00 UTC

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


    cl1 = np.arange(-200,220,20)
    x = np.arange(0,360)
    y = np.arange(0,91)
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 5))
    ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plt.xlim(0, 360)
    plt.ylim(10, 80)
    plt.title('Integrated column -div (Fx + Fy)   20 - 26 June 2021')
    plt.ylabel('Latitude', fontsize=22)
    ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax6.coastlines(alpha = 0.7)
    ax6.set_aspect('auto', adjustable=None)
    ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    ott = ax6.contourf(x,y,z1s+z2s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
    fig.colorbar(ott,ax=ax6,label='(m/s)')
    ax6.quiver(xx[2:-2,:],yy[2:-2,:],uu[2:-2, :],vv[2:-2, :],transform=ccrs.PlateCarree())
    plt.savefig('divFx+Fy_0.png', bbox_inches='tight', dpi =600)

    cl1 = np.arange(-200,220,20)
    x = np.arange(0,360)
    y = np.arange(0,91)
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 5))
    ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plt.xlim(0, 360)
    plt.ylim(10, 80)
    plt.title('Integrated column -div Fy   20 - 26 June 2021')
    plt.xlabel('Longitude', fontsize=22)
    plt.ylabel('Latitude', fontsize=22)
    ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax6.coastlines(alpha = 0.7)
    ax6.set_aspect('auto', adjustable=None)
    ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    ott = ax6.contourf(x,y,z2s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
    fig.colorbar(ott,ax=ax6,label='(m/s)')
    ax6.quiver(xx[2:-2,:],yy[2:-2,:],uu[2:-2, :],vv[2:-2, :],transform=ccrs.PlateCarree())
    plt.savefig('divFy_0.png', bbox_inches='tight', dpi =600)

    cl1 = np.arange(-200,220,20)
    x = np.arange(0,360)
    y = np.arange(0,91)
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 5))
    ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plt.xlim(0, 360)
    plt.ylim(10, 80)
    plt.title('Integrated low-level source   20 - 26 June 2021')
    ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax6.coastlines(alpha = 0.7)
    ax6.set_aspect('auto', adjustable=None)
    ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    ott = ax6.contourf(x,y,z3s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
    fig.colorbar(ott,ax=ax6,label='(m/s)')
    plt.savefig('EP4_0.png', bbox_inches='tight', dpi =600)

    cl1 = np.arange(-200,220,20)
    x = np.arange(0,360)
    y = np.arange(0,91)
    fig = plt.figure(figsize=(10, 5))
    ax6 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
    plt.xlim(0, 360)
    plt.ylim(10, 80)
    plt.title('Integrated residual   20 - 26 June 2021')
    ax6.set_extent([-220, -80, 10, 80], ccrs.PlateCarree())
    ax6.coastlines(alpha = 0.7)
    ax6.set_aspect('auto', adjustable=None)
    ax6.set_xticks([140,160,180,200,220,240,260,280], crs=ccrs.PlateCarree())
    ax6.set_yticks([10, 20, 30, 40, 50, 60, 70, 80], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    ott = ax6.contourf(x,y,zs-z1s-z2s-z3s,levels=cl1,transform=ccrs.PlateCarree(),cmap='rainbow')
    fig.colorbar(ott,ax=ax6,label='(m/s)')
    ax6.quiver(xx[2:-2,:],yy[2:-2,:],uu[2:-2, :],vv[2:-2, :],transform=ccrs.PlateCarree())
    plt.savefig('Residual_0.png', bbox_inches='tight', dpi =600)
    plt.close("all")


def plot_figure3e(mtnlwrf_filename, mtnlwrfcs_filename):
    """
    :param mtnlwrf_filename: netCDF fileof net OLR data
    :param mtnlwrfcs_filename: netCDF fileof clear sky OLR data
    :return:
    """
    ncin1 = Dataset(mtnlwrf_filename, 'r', format='NETCDF4')
    ncin2 = Dataset(mtnlwrfcs_filename, 'r', format='NETCDF4')
    olr = ncin1.variables['mtnlwrf']
    olr = (np.array(olr))
    olrcs = ncin2.variables['mtnlwrfcs']
    olrcs = (np.array(olrcs))

    tt = np.zeros((44,181,360))
    t = np.zeros((181,360))
    m = 77
    for m in range(77,120):
        tt[m-77,:,:] = olr[m,:,:]

    b = [f'OLR_06{i}.png' for i in range(20, 31)]

    for n in range(0, 11):
        nn = n*4

        for j in range(0, 181):
            t[j,:]=tt[nn,180-j,:]

        cl1 = np.arange(80,390,10)
        x = np.arange(0,360)
        y = np.arange(0,181)-90.
        plt.rcParams.update({'font.size':16, 'text.usetex': False})
        fig = plt.figure(figsize=(10, 5))
        ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(10, 80)
        plt.title('OLR  ' + date_stamp[n])
        plt.xlabel('Longitude', fontsize=22)
        plt.ylabel('Latitude', fontsize=22)
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
        plt.savefig(b[n], bbox_inches='tight', dpi =600)
        plt.close()


def plot_figure3f(tcw_filename, tcwv_filename, sp_filename):
    """
    :param tcw_filename: filename of netCDF file with total column water (kg/m^2)
    :param tcwv_filename: filename of netCDF file with total column water vapor (kg/m^2)
    :param sp_filename: filename of netCDF file with sea level pressure (hPa)
    :return:
    """
    ncin1 = Dataset(tcw_filename, 'r', format='NETCDF4')
    ncin2 = Dataset(tcwv_filename, 'r', format='NETCDF4')
    ncin3 = Dataset(sp_filename, 'r', format='NETCDF4')
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

    for m in range(77, 120):
        tt[m-77,:,:] = cw[m,:,:]-cwv[m,:,:]
        pp[m-77,:,:] = sp[m,:,:]/100.

    b = [f'CW_06{i}.png' for i in range(20, 31)]

    for n in range(0, 11):
        nn = n*4

        for j in range(0, 181):
            t[j,:]=tt[nn,180-j,:]
            p[j,:]=pp[nn,180-j,:]

        cl1 = np.arange(-0.1,3.6,0.1)
        c12 = np.arange(980,1032,4)
        x = np.arange(0,360)
        y = np.arange(0,181)-90.
        plt.rcParams.update({'font.size':16, 'text.usetex': False})
        fig = plt.figure(figsize=(10, 5))
        ax5 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(180))
        plt.xlim(140, 280)
        plt.ylim(10, 80)
        plt.title('Column water  ' + date_stamp[n])
        plt.xlabel('Longitude', fontsize=22)
        #plot.ylabel('Latitude')
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
        plt.savefig(b[n], bbox_inches='tight', dpi =600)
        plt.close()


def plot_figure4(lwa_flux_filename):
    dt = 3600. * 6.
    a = 6378000.
    dl = 2. * np.pi / 360.
    dp = 2. * np.pi / 360.

    jm = 49
    jp = 49

    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')

    lwa = ncin1.variables['lwa']
    lwa = (np.array(lwa))

    z = np.zeros((124,360))  # wave activity tendency
    w = np.zeros((124,360))  # wave activity

    for m in range(1, 120):

        cos0 = 0
        for j in range(jm, jp+1):

            phi = dp*j
            z[m,:] = z[m,:]+np.cos(phi)*(lwa[m+1,j,:]-lwa[m-1,j,:])/(2.*dt)   # wave activity tendency at 49 N
            w[m,:] = w[m,:]+np.cos(phi)*lwa[m,j,:]
            cos0 = cos0 + np.cos(phi)

        z[m,:] = z[m,:]/cos0
        w[m,:] = w[m,:]/cos0

    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ua1 = ncin1.variables['ua1']
    ua2 = ncin1.variables['ua2']
    ep1 = ncin1.variables['ep1']
    ep2 = ncin1.variables['ep2']
    ep3 = ncin1.variables['ep3']
    ep4 = (np.array(ncin1.variables['ep4']))

    f1 = np.zeros((124,360))

    z1 = np.zeros((124,360))
    z2 = np.zeros((124,360))
    z3 = np.zeros((124,360))
    z4 = np.zeros((124,360))


    for i in range(1, 359):

        for j in range(jm, jp+1):
            phi = dp*j
            const = 1./(2.*a*dl)
            z1[:,i] = z1[:,i]-const*(ua1[:,j,i+1]+ua2[:,j,i+1]+ep1[:,j,i+1]-ua1[:,j,i-1]-ua2[:,j,i-1]-ep1[:,j,i-1])

        z1[:,i] = z1[:,i]/cos0


    for i in range(0, 360):

        for j in range(jm, jp+1):
            phi = dp*j
            const = 1./(2.*a*dl)
            f1[:,i]=f1[:,i]+(ua1[:,j,i]+ua2[:,j,i]+ep1[:,j,i])*np.cos(phi)   # zonal wave activity flux
            z2[:,i]=z2[:,i]+const*(ep2[:,j,i]-ep3[:,j,i])         # metisional convergence of wave activity flux
            z3[:,i]=z3[:,i]+np.cos(phi)*ep4[:,j,i]                          # bottom source

        f1[:,i] = f1[:,i]/cos0
        z2[:,i] = z2[:,i]/cos0
        z3[:,i] = z3[:,i]/cos0


    zs = np.zeros((124,360))  # smoothed z #
    ws = np.zeros((124,360))  # smoothed w #
    f1s = np.zeros((124,360))  # smoothed f1 #
    z1s = np.zeros((124,360))  # smoothed z1 #
    z2s = np.zeros((124,360))  # smoothed z2 #
    z3s = np.zeros((124,360))  # smoothed z3 #

    #### smoothing in longitude ####
    n = 4   # smoothing width = 2n+1

    for j in range(0, 124):
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

        for nn in range(-n, n+1):

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

    z4[:,:]=zs[:,:]-z1s[:,:]-z2s[:,:]-z3s[:,:]  # residual

    ##############################################

    cl2 = np.arange(0,135,5)
    x = np.arange(0,360)
    y = np.arange(0,124)*0.25
    plt.rcParams.update({'font.size': 28})
    fig,ax5 = plt.subplots(1, figsize=(8, 8))
    plt.xlim(140, 280)
    plt.ylim(20, 29.75)
    plt.title('Column LWA 49$^\circ$N')
    plt.xlabel('Longitude')
    plt.ylabel('Day')
    ott = ax5.contourf(x,y,w,levels=cl2,cmap='rainbow')
    fig.colorbar(ott,ax=ax5,label='(ms$^{-1}$)')
    plt.savefig('HovmollerLWA.png', bbox_inches='tight', dpi =600)

    ##############################################

    cl2 = np.arange(-100,1500,100)
    x = np.arange(0,360)
    y = np.arange(0,124)*0.25
    plt.rcParams.update({'font.size': 28})
    fig,ax5 = plt.subplots(1, figsize=(8, 8))
    plt.xlim(140, 280)
    plt.ylim(20, 29.75)
    plt.title('<F$_\lambda$> 49$^\circ$N')
    plt.xlabel('Longitude')
    plt.ylabel('Day')
    ott = ax5.contourf(x,y,f1,levels=cl2,cmap='rainbow')
    fig.colorbar(ott,ax=ax5,label='(m$^2$s$^{-2}$)')
    plt.savefig('HovmollerFx.png', bbox_inches='tight', dpi =600)


    ##############################################

    cl2 = np.arange(-10.,9,1)
    x = np.arange(0,360)
    y = np.arange(0,124)*0.25
    plt.rcParams.update({'font.size': 28})
    fig,ax5 = plt.subplots(1, figsize=(8, 8))
    plt.xlim(140, 280)
    plt.ylim(20, 29.5)
    plt.title('Term (IV) 49$^\circ$N')
    plt.xlabel('Longitude')
    plt.ylabel('Day')
    ott = ax5.contourf(x,y,z4*10000,levels=cl2,cmap='rainbow')
    fig.colorbar(ott,ax=ax5,label='(10$^{-4}$ ms$^{-2}$)')
    plt.savefig('HovmollerRes.png', bbox_inches='tight', dpi =600)
    #plot.show()


    ##############################################

    cl2 = np.arange(-10.,9,1)
    x = np.arange(0,360)
    y = np.arange(0,124)*0.25
    plt.rcParams.update({'font.size': 28})
    fig,ax5 = plt.subplots(1, figsize=(8, 8))
    plt.xlim(140, 280)
    plt.ylim(20, 29.75)
    plt.title('Terms (I)+(II) 49$^\circ$N')
    plt.xlabel('Longitude')
    plt.ylabel('Day')
    ott = ax5.contourf(x,y,(z1s+z2s+z3s)*10000,levels=cl2,cmap='rainbow')
    fig.colorbar(ott,ax=ax5,label='(10$^{-4}$ ms$^{-2}$)')
    plt.savefig('HovmollerFxy.png', bbox_inches='tight', dpi =600)


def plot_figure5(lwa_flux_filename):

    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')

    lwa = ncin1.variables['lwa']
    lwa = (np.array(lwa))

    print(lwa.shape)

    z = np.zeros((120,360))
    for m in range(0, 120):
        z[m,:] = lwa[m,49,:]   # 49N LWA for June 1-30

    zs = np.zeros((120,360))  # smoothed z #

    #### smoothing in longitude ####
    n = 5     # smoothing width #
    for m in range(0, 120):
        zx = np.zeros(360)
        zx[:] = z[m,:]

        for nn in range(-n, n+1):
            zy = np.roll(zx,nn)
            zs[m,:] = zs[m,:] + zy[:]/(2*n+1)

    zx[:] = zs[100,:]-zs[76,:]
    zy[:] = 0

    ncin1 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ua1 = ncin1.variables['ua1']
    ua1 = (np.array(ua1))
    ncin2 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ua2 = ncin2.variables['ua2']
    ua2 = (np.array(ua2))
    ncin3 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep1 = ncin3.variables['ep1']
    ep1 = (np.array(ep1))
    ncin4 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep2 = ncin4.variables['ep2']
    ep2 = (np.array(ep2))
    ncin5 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep3 = ncin5.variables['ep3']
    ep3 = (np.array(ep3))
    ncin6 = Dataset(lwa_flux_filename, 'r', format='NETCDF4')
    ep4 = ncin6.variables['ep4']
    ep4 = (np.array(ep4))

    f1 = np.zeros((120,360))
    f2 = np.zeros((120,360))
    f11 = np.zeros((120,360))
    f22 = np.zeros((120,360))

    z1 = np.zeros((120,360))
    z2 = np.zeros((120,360))
    z3 = np.zeros((120,360))
    dt = 3600.*6.
    a = 6378000.
    dl = 2.*np.pi/360.
    dp = 2.*np.pi/360.
                       # m = 76 is 20 June 2021 00 UTC
    for m in range(0, 120):     # m = 100 is 26 June 2021 00 UTC
        z3[m,:] = ep4[m,49,:]
        f1[m,:] = ua1[m,49,:]+ua2[m,49,:]+ep1[m,49,:]
        j = 49
        phi = dp*j
        const = 1./(2.*a*np.cos(phi)*dl)
        z2[m,:]=const*(ep2[m,49,:]-ep3[m,49,:])
        f2[m,:] = 0.5*(ep2[m,49,:]+ep3[m,49,:])/np.cos(phi)

        for i in range(1, 359):
            z1[m,i] = -const*(f1[m,i+1]-f1[m,i-1])

        z1[m,0] = -const*(f1[m,1]-f1[m,359])
        z1[m,359] = -const*(f1[m,0]-f1[m,358])


    z1s = np.zeros((120,360))  # smoothed z1 #
    z2s = np.zeros((120,360))  # smoothed z2 #
    z3s = np.zeros((120,360))  # smoothed z3 #
    f1s = np.zeros((120,360))  # smoothed f1 #

    #### smoothing in longitude ####

    for m in range(0, 120):
        z1x = np.zeros(360)
        z1x[:] = z1[m,:]
        z2x = np.zeros(360)
        z2x[:] = z2[m,:]
        z3x = np.zeros(360)
        z3x[:] = z3[m,:]
        f1x = np.zeros(360)
        f1x[:] = f1[m,:]

        nn = -n
        for nn in range(-n, n+1):
            z1y = np.roll(z1x,nn)
            z1s[m,:] = z1s[m,:] + z1y[:]/(2*n+1)
            z2y = np.roll(z2x,nn)
            z2s[m,:] = z2s[m,:] + z2y[:]/(2*n+1)
            z3y = np.roll(z3x,nn)
            z3s[m,:] = z3s[m,:] + z3y[:]/(2*n+1)
            f1y = np.roll(f1x,nn)
            f1s[m,:] = f1s[m,:] + f1y[:]/(2*n+1)



    ##### Time integration test ######

    lwa = np.zeros(360)
    res = np.zeros((120,360))
    dlwa = np.zeros((120,360))
    gama = np.zeros((120,360))
    gama1 = np.zeros((120,360))
    gama2 = np.zeros((120,360))
    cgx = np.zeros((120,360))

    m = 0
    for m in range(0, 119):
        dlwa[m,:] = zs[m+1,:]-zs[m,:]
        cgx[m,:] = (f1s[m,:]+f1s[m+1,:])/(zs[m,:]+zs[m+1,:])
        res[m,:] = (dlwa[m,:] - 0.5*dt*(z1s[m,:]+z1s[m+1,:]+z2s[m,:]+z2s[m+1,:]+z3s[m,:]+z3s[m+1,:]))/dt
        gama[m,:] = 2.*res[m,:]/(zs[m,:]+zs[m+1,:])


    for m in range(76, 100):
        lwa[:] = lwa[:] + 0.5*dt*(z1s[m,:]+z1s[m+1,:]+z2s[m,:]+z2s[m+1,:]+z3s[m,:]+z3s[m+1,:]) + dt*res[m,:]
        m = m+1

    j = 49
    phi = dp*j
    const = 1./(2.*a*np.cos(phi)*dl)
    lwa0 = np.zeros((120,360))
    lwa1 = np.zeros((120,360))

    lwa0[:,:] = zs[:,:]
    lwa1[:,:] = zs[:,:]


    for m in range(76, 100):

        for i in range(1, 359):
            d0 = -const*0.5*((lwa0[m,i+1]+lwa0[m+1,i+1])*cgx[m,i+1]-(lwa0[m,i-1]+lwa0[m+1,i-1])*cgx[m,i-1])
            d2 = 0.5*(z2s[m+1,i]+z2s[m,i])
            d3 = 0.5*(z3s[m+1,i]+z3s[m,i])
            d4 = gama[m,i]*0.5*(zs[m,i]+zs[m+1,i])
            lwa1[m+1,i] = lwa1[m,i]+dt*(d0+d2+d3+d4)

        d0 = -const*0.5*((lwa0[m,0]+lwa0[m+1,0])*cgx[m,0]-(lwa0[m,358]+lwa0[m+1,358])*cgx[m,358])
        d2 = 0.5*(z2s[m+1,359]+z2s[m,359])
        d3 = 0.5*(z3s[m+1,359]+z3s[m,359])
        d4 = gama[m,359]*0.5*(zs[m,359]+zs[m+1,359])
        lwa1[m+1,359] = lwa1[m,359]+dt*(d0+d2+d3+d4)

        d0 = -const*0.5*((lwa0[m,1]+lwa0[m+1,1])*cgx[m,1]-(lwa0[m,359]+lwa0[m+1,359])*cgx[m,359])
        d2 = 0.5*(z2s[m+1,0]+z2s[m,0])
        d3 = 0.5*(z3s[m+1,0]+z3s[m,0])
        d4 = gama[m,0]*0.5*(zs[m,0]+zs[m+1,0])
        lwa1[m+1,0] = lwa1[m,0]+dt*(d0+d2+d3+d4)


    ####  Modify gamma  ####

    gama1[:,:] = gama[:,:]
    gama2[:,:] = gama[:,:]

    for m in range(76, 100):
        if((m >= 84) and (m <= 92)):
            i = 200
            for i in range(200, 221):
                if(gama[m,i] > 0):
                    gama1[m,i] = gama[m,i]*0.7
                    gama2[m,i] = gama[m,i]*0.


    gama1[:,:] = gama1[:,:]-gama[:,:]
    gama2[:,:] = gama2[:,:]-gama[:,:]

    ##### Interpolate gama, gama1, cgx onto finer tiem mesh dt = 30 min #####

    gamap = np.zeros((8640,360))
    gama1p = np.zeros((8640,360))
    gama2p = np.zeros((8640,360))
    cgxp = np.zeros((8640,360))
    forcep = np.zeros((8640,360))
    lwap = np.zeros((8640,360))

    for m in range(76, 101):
        for i in range(0, 360):
            mm = (m-76)*360+i-180
            if(mm >= 0):
                if(mm < 8640):
                    gamap[mm,:] = gama[m,:]*(i/360.)+gama[m-1,:]*(1.-i/360.)
                    gama1p[mm,:] = gama1[m,:]*(i/360.)+gama1[m-1,:]*(1.-i/360.)
                    gama2p[mm,:] = gama2[m,:]*(i/360.)+gama2[m-1,:]*(1.-i/360.)
                    cgxp[mm,:] = cgx[m,:]*(i/360.)+cgx[m-1,:]*(1.-i/360.)

    for m in range(76, 100):
        for i in range(0, 360):
            mm = (m-76)*360+i
            forcep[mm,:] = (z2s[m+1,:]+z3s[m+1,:])*(i/360.)+(z2s[m,:]+z3s[m,:])*(1.-i/360.)
            lwap[mm,:] = zs[m+1,:]*(i/360.)+zs[m,:]*(1.-i/360.)


    ##### Time integration ####

    dt = 60.
    j = 49
    phi = dp*j
    const = 1./(2.*a*np.cos(phi)*dl)
    al = 0.
    diff = 200000.
    dlwap = np.zeros((8641,360))
    dlwap2 = np.zeros((8641,360))


    for m in range(1, 8640):
        for i in range(1, 359):
            d1 = -const*(dlwap[m,i+1]*(cgxp[m,i+1]-al*(dlwap[m,i+1]+lwap[m,i+1]))-dlwap[m,i-1]*(cgxp[m,i-1]-al*(dlwap[m,i-1]+lwap[m,i-1])))
            d2 = gama1p[m,i]*dlwap[m,i]
            d3 = gamap[m,i]*dlwap[m,i]
            d4 = gama1p[m,i]*lwap[m,i]
            d5 = diff*(dlwap[m-1,i+1]+dlwap[m-1,i-1]-2.*dlwap[m-1,i])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
            dlwap[m+1,i] = dlwap[m-1,i]+2.*dt*(d1+d2+d3+d4+d5)

            d12 = -const*(dlwap2[m,i+1]*(cgxp[m,i+1]-al*(dlwap2[m,i+1]+lwap[m,i+1]))-dlwap2[m,i-1]*(cgxp[m,i-1]-al*(dlwap2[m,i-1]+lwap[m,i-1])))
            d22 = gama2p[m,i]*dlwap2[m,i]
            d32 = gamap[m,i]*dlwap2[m,i]
            d42 = gama2p[m,i]*lwap[m,i]
            d52 = diff*(dlwap2[m-1,i+1]+dlwap2[m-1,i-1]-2.*dlwap2[m-1,i])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
            dlwap2[m+1,i] = dlwap2[m-1,i]+2.*dt*(d12+d22+d32+d42+d52)

        d1 = -const*(dlwap[m,0]*(cgxp[m,0]-al*(dlwap[m,0]+lwap[m,0]))-dlwap[m,358]*(cgxp[m,358]-al*(dlwap[m,358]+lwap[m,358])))
        d2 = gama1p[m,359]*dlwap[m,359]
        d3 = gamap[m,359]*dlwap[m,359]
        d4 = gama1p[m,359]*lwap[m,359]
        d5 = diff*(dlwap[m-1,0]+dlwap[m-1,358]-2.*dlwap[m-1,359])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap[m+1,359] = dlwap[m-1,359]+2.*dt*(d1+d2+d3+d4+d5)

        d12 = -const*(dlwap2[m,0]*(cgxp[m,0]-al*(dlwap2[m,0]+lwap[m,0]))-dlwap2[m,358]*(cgxp[m,358]-al*(dlwap2[m,358]+lwap[m,358])))
        d22 = gama2p[m,359]*dlwap2[m,359]
        d32 = gamap[m,359]*dlwap2[m,359]
        d42 = gama2p[m,359]*lwap[m,359]
        d52 = diff*(dlwap2[m-1,0]+dlwap2[m-1,358]-2.*dlwap2[m-1,359])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap2[m+1,359] = dlwap[m-1,359]+2.*dt*(d12+d22+d32+d42+d52)

        d1 = -const*(dlwap[m,1]*(cgxp[m,1]-al*(dlwap[m,1]+lwap[m,1]))-dlwap[m,359]*(cgxp[m,359]-al*(dlwap[m,359]+lwap[m,359])))
        d2 = gama1p[m,0]*dlwap[m,0]
        d3 = gamap[m,0]*dlwap[m,0]
        d4 = gama1p[m,0]*lwap[m,0]
        d5 = diff*(dlwap[m-1,1]+dlwap[m-1,359]-2.*dlwap[m-1,0])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap[m+1,0] = dlwap[m-1,0]+2.*dt*(d1+d2+d3+d4+d5)

        d12 = -const*(dlwap2[m,1]*(cgxp[m,1]-al*(dlwap2[m,1]+lwap[m,1]))-dlwap2[m,359]*(cgxp[m,359]-al*(dlwap2[m,359]+lwap[m,359])))
        d22 = gama2p[m,0]*dlwap2[m,0]
        d32 = gamap[m,0]*dlwap2[m,0]
        d42 = gama2p[m,0]*lwap[m,0]
        d52 = diff*(dlwap2[m-1,1]+dlwap2[m-1,359]-2.*dlwap2[m-1,0])/(a*a*dl*dl*np.cos(phi)*np.cos(phi))
        dlwap2[m+1,0] = dlwap2[m-1,0]+2.*dt*(d12+d22+d32+d42+d52)

    gm = np.zeros(360)
    gm2 = np.zeros(360)
    gm[:] = lwa[:]+dlwap[8640,:]
    gm2[:] = lwa[:]+dlwap2[8640,:]
    x = np.arange(0,360)
    plt.rcParams.update({'font.size':14})
    fig = plt.figure(figsize=(8, 4))
    plt.xlim(140, 280)
    plt.ylim(-80, 80)
    plt.title('Column LWA Change  49$^\circ$N  June 20 - 26  00 UTC')
    plt.xlabel('Longitude')
    plt.ylabel('$\Delta$LWA (m/s)')
    ott = plt.plot(x, zx)
    ott = plt.plot(x, zy, color='black', alpha = 0.3)
    ott = plt.plot(x, gm, color='red', alpha = 0.5)
    ott = plt.plot(x, gm2, 'r--', alpha = 0.5)
    plt.savefig('dLWAp.png', bbox_inches='tight', dpi =600)


