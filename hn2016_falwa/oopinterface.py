# This is the api for object oriented interface
import numpy as np
from math import pi
from scipy import interpolate


# The function assumes uniform field

def curl_2D(ufield, vfield, clat, dlambda, dphi, planet_radius=6.378e+6):
    """
    Assuming regular latitude and longitude [in degree] grid, compute the curl
    of velocity on a pressure level in spherical coordinates.
    """

    ans = np.zeros_like((ufield))
    ans[1:-1, 1:-1] = (vfield[1:-1, 2:] - vfield[1:-1, :-2])/(2.*dlambda) - \
                      (ufield[2:, 1:-1] * clat[2:, np.newaxis] -
                       ufield[:-2, 1:-1] * clat[:-2, np.newaxis])/(2.*dphi)
    ans[0, :] = 0.0
    ans[-1, :] = 0.0
    ans[1:-1, 0] = ((vfield[:, 1] - vfield[:, -1])/(2.*dlambda) -
                    (ufield[2:, 0] * clat[2:, np.newaxis] -
                     ufield[:-2, 0] * clat[:-2, np.newaxis])/(2.*dphi))
    ans[1:-1, -1] = ((vfield[:, 0] - vfield[:, -2])/(2.*dlambda) -
                     (ufield[2:, -1] * clat[2:, np.newaxis] -
                      ufield[:-2, -1] * clat[:-2, np.newaxis])/(2.*dphi))
    ans[1:-1, :] = ans[1:-1, :] / planet_radius / clat[1:-1, np.newaxis]
    return ans


class BarotropicField(object):

    """
    An object that deals with barotropic (2D) wind and/or PV fields

    :param  xlon: Longitude array in degree with dimension *nlon*.
    :type xlon: sequence of array_like

    :param  ylat: Latitutde array in degree, monotonically increasing with dimension *nlat*
    :type ylat: sequence of array_like

    :param  area: Differential area at each lon-lat grid points with dimension (nlat,nlon). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface.
    :type area: sequence of array_like

    :param  dphi: Differential length element along the lat grid with dimension nlat.
    :type dphi: sequence of array_like

    :param  pv_field: Absolute vorticity field with dimension [nlat x nlon]. If 'pv_field=None': pv_field is expected to be computed with u,v,t field.
    :type pv_field: sequence of array_like

    :returns: an instance of the object BarotropicField

    :example:
    >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)

    """

    def __init__(self, xlon, ylat, pv_field, area=None, dphi=None,
                 n_partitions=None, planet_radius=6.378e+6):

        """Create a windtempfield object.

        **Arguments:**

        *xlon*
            Longitude array in degree with dimension [nlon].

        *ylat*
            Latitutde array in degree, monotonically increasing with dimension
            [nlat].

        *area*
            Differential area at each lon-lat grid points with dimension
            [nlat x nlon].
            If None, it will be initiated as:
            2.*pi*Earth_radius**2 *(np.cos(ylat[:,np.newaxis]*pi/180.)*dphi)/float(nlon) * np.ones((nlat,nlon)).
            This would be problematic if the grids are not uniformly distributed in degree.

        *dphi*
            Differential length element along the lat grid with dimension nlat.

        *pv_field*
            Absolute vorticity field with dimension [nlat x nlon].
            If none, pv_field is expected to be computed with u,v,t field.

        """

        self.xlon = xlon
        self.ylat = ylat
        self.clat = np.abs(np.cos(np.deg2rad(ylat)))
        self.nlon = xlon.size
        self.nlat = ylat.size
        self.planet_radius = planet_radius
        if dphi is None:
            self.dphi = pi/(self.nlat-1) * np.ones((self.nlat))
        else:
            self.dphi = dphi

        if area is None:
            self.area = 2.*pi*planet_radius**2*(np.cos(ylat[:, np.newaxis]*pi/180.)*self.dphi[:, np.newaxis])/float(self.nlon)*np.ones((self.nlat, self.nlon))
        else:
            self.area = area

        self.pv_field = pv_field

        if n_partitions is None:
            self.n_partitions = self.nlat
        else:
            self.n_partitions = n_partitions

    def equivalent_latitudes(self):

        """
        Compute equivalent latitude with the *pv_field* stored in the object.

        :returns: an numpy array with dimension (nlat) of equivalent latitude array.

        :example:
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> eqv_lat = barofield1.equivalent_latitudes()

        """

        from hn2016_falwa import basis

        pv_field = self.pv_field
        area = self.area
        ylat = self.ylat
        planet_radius = self.planet_radius

        self.eqvlat, dummy = basis.eqvlat(ylat, pv_field, area, self.n_partitions,
                                    planet_radius=planet_radius)
        return self.eqvlat


    def lwa(self):

        """
        Compute the finite-amplitude local wave activity based on the *equivalent_latitudes* and the *pv_field* stored in the object.

        :returns: an 2-D numpy array with dimension (nlat,nlon) of local wave activity values.

        :example:
        >>> barofield1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)
        >>> eqv_lat = barofield1.equivalent_latitudes() # This line is optional
        >>> lwa = barofield1.lwa()

        """
        from hn2016_falwa import basis

        if self.eqvlat is None:
            self.eqvlat = self.equivalent_latitudes(self)

        lwa_ans, dummy = basis.lwa(self.nlon, self.nlat, self.pv_field, self.eqvlat,
                                   self.planet_radius * self.clat * self.dphi)
        return lwa_ans


# === Next is a class of 3D objects ===
class QGField(object):

    """
    An object that deals with barotropic (2D) wind and/or PV fields

    :param  xlon: Longitude array in degree with dimension (*nlon*).
    :type xlon: sequence of array_like

    :param  ylat: Latitutde array in degree, monotonically increasing with dimension (*nlat*)
    :type ylat: sequence of array_like

    :param  zlev: Pseudoheight array in meters, monotonically increasing with dimension (*nlev*)
    :type zlev: sequence of array_like

    :param  u_field: Zonal wind field in meters, with dimension (*nlev*,*nlat*,*nlon*).
    :type u_field: sequence of array_like

    :param  v_field: Meridional wind field in meters, with dimension (*nlev*,*nlat*,*nlon*).
    :type v_field: sequence of array_like

    :param  t_field: Temperature field in Kelvin, with dimension (*nlev*,*nlat*,*nlon*).
    :type t_field: sequence of array_like

    :param  qgpv_field: Quasi-geostrophic potential vorticity field in 1/second, with dimension (*nlev*,*nlat*,*nlon*). If u_field, v_field and t_field are input, qgpv_field can be using the method compute_qgpv.
    :type qgpv_field: sequence of array_like

    :param  area: Differential area at each lon-lat grid points with dimension (*nlat*,*nlon*). If 'area=None': it will be initiated as area of uniform grid (in degree) on a spherical surface.
    :type area: sequence of array_like

    :param  dphi: Differential length element along the lat grid with dimension (*nlat*).
    :type dphi: sequence of array_like

    :param  pv_field: Absolute vorticity field with dimension [nlat x nlon]. If 'pv_field=None': pv_field is expected to be computed with u,v,t field.
    :type pv_field: sequence of array_like

    :returns: an instance of the object BarotropicField

    :example:
    >>> qgfield1 = QGField(xlon, ylat, np.array([240.]), u, qgpv_field=QGPV)

    """


    def __init__(self, xlon, ylat, zlev, u_field, v_field=None, t_field=None,
                 qgpv_field=None, area=None, dphi=None,
                 n_partitions=None, rkappa=287./1004., planet_radius=6.378e+6,
                 scale_height=7000.):

        """Create a windtempfield object.

        **Arguments:**

        *xlon*
            Longitude array in degree with dimension [nlon].

        *ylat*
            Latitutde array in degree, monotonically increasing with dimension
            [nlat].

        *zlev*
            Pseudoheight array in meters, monotonically increasing with dimension
            [nlev].

        *u_field*
            Zonal wind field in meters, with dimension [nlev x nlat x nlon].

        *v_field*
            Meridional wind field in meters, with dimension [nlev x nlat x nlon].

        *t_field*
            Temperature field in Kelvin, with dimension [nlev x nlat x nlon].

        *qgpv_field*
            Quasi-geostrophic potential vorticity field in 1/second, with dimension
            [nlev x nlat x nlon]. If u_field, v_field and t_field are input,
            qgpv_field can be using the method compute_qgpv.

        *area*
            Differential area at each lon-lat grid points with dimension
            [nlat x nlon].
            If None, it will be initiated as:
            2.*pi*Earth_radius**2 *(np.cos(ylat[:,np.newaxis]*pi/180.)*dphi)/float(nlon) * np.ones((nlat,nlon)).
            This would be problematic if the grids are not uniformly distributed in degree.

        *dphi*
            Differential length element along the lat grid with dimension nlat.

        *n_partitions*
            Number of partitions used to compute equivalent latitude. If not
            given, it will be assigned nlat.

        """

        self.xlon = xlon
        self.ylat = ylat
        self.zlev = zlev
        self.clat = np.abs(np.cos(np.deg2rad(ylat)))
        self.nlon = xlon.size
        self.nlat = ylat.size
        self.nlev = zlev.size
        self.planet_radius = planet_radius
        if dphi is None:
            self.dphi = pi/(self.nlat-1) * np.ones((self.nlat))
        else:
            self.dphi = dphi

        if area is None:
            self.area = 2.*pi*planet_radius**2*(np.cos(ylat[:, np.newaxis]*pi/180.)*self.dphi[:, np.newaxis])/float(self.nlon)*np.ones((self.nlat, self.nlon))
        else:
            self.area = area

        self.qgpv_field = qgpv_field

        if n_partitions is None:
            self.n_partitions = self.nlat
        else:
            self.n_partitions = n_partitions

        # First, check if the qgpv_field is present
        print('check self.qgpv_field')
        # print self.qgpv_field
        if (qgpv_field is None) & (v_field is None):
            raise ValueError('qgpv_field is missing.')
        elif (qgpv_field is None):
            print('Compute QGPV field from u and v field.')

        # === Obtain potential temperature field ===
        if t_field:
            self.pt_field = t_field[:, :, :] * \
             np.exp(rkappa * zlev[:, np.newaxis, np.newaxis]/scale_height)
            # Interpolation
            f_Thalf = interpolate.interp1d(zlev, self.pt_field.mean(axis=-1),
                                           axis=0)
            zlev_half = np.array([zlev[0] + 0.5*(zlev[1]-zlev[0])]*i \
                                 for i in range(zlev.size * 2 + 1))
            self.pt_field_half = f_Thalf(zlev_half) # dim = [2*nlev+1,nlat]
            print('self.pt_field_half.shape')
            print(self.pt_field_half.shape)


    def equivalent_latitudes(self, domain_size='half_globe'): # Has to be changed since it is qgpv.
                                    # Use half-globe?
        """
        Compute equivalent latitude with the *pv_field* stored in the object.

        :param  domain_size: domain of grids to be used to compute equivalent latitude. It can he 'half_globe' or 'full_globe'.
        :type domain_size: string

        :returns: an numpy array with dimension (*nlev*,*nlat*) of equivalent latitude array.

        :example:
        >>> qgfield1 = QGField(xlon, ylat, np.array([240.]), u, qgpv_field=QGPV)
        >>> qgfield_eqvlat = qgfield1.equivalent_latitudes(domain_size='half_globe')

        """

        def eqv_lat_core(ylat, vort, area, n_points):
            vort_min = np.min([vort.min(), vort.min()])
            vort_max = np.max([vort.max(), vort.max()])
            q_part_u = np.linspace(vort_min, vort_max, n_points,
                                   endpoint=True)
            aa = np.zeros(q_part_u.size)  # to sum up area
            vort_flat = vort.flatten()  # Flatten the 2D arrays to 1D
            area_flat = area.flatten()
            # Find equivalent latitude:
            inds = np.digitize(vort_flat, q_part_u)
            for i in np.arange(0, aa.size):  # Sum up area in each bin
                aa[i] = np.sum(area_flat[np.where(inds == i)])
            aq = np.cumsum(aa)
            y_part = aq/(2*pi*planet_radius**2) - 1.0
            lat_part = np.arcsin(y_part)*180/pi
            q_part = np.interp(ylat, lat_part, q_part_u)
            return q_part

        area = self.area
        ylat = self.ylat
        planet_radius = self.planet_radius
        self.eqvlat = np.zeros((self.nlev, self.nlat))

        for k in range(self.nlev):
            pv_field = self.qgpv_field[k, ...]

            if domain_size == 'half_globe':
                nlat_s = int(self.nlat/2)
                qref = np.zeros(self.nlat)
                # --- Southern Hemisphere ---
                # qref1 = eqv_lat_core(ylat[:nlat_s],vort[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=planet_radius)
                qref[:nlat_s] = eqv_lat_core(ylat[:nlat_s], pv_field[:nlat_s,:],
                                             area[:nlat_s, :], nlat_s)
                # --- Northern Hemisphere ---
                pv_field_inverted = -pv_field[::-1, :]  # Added the minus sign, but gotta see if NL_North is affected
                qref2 = eqv_lat_core(ylat[:nlat_s], pv_field_inverted[:nlat_s,:],
                                     area[:nlat_s, :], nlat_s)
                #qref2 = eqvlat(ylat[:nlat_s],vort2[:nlat_s,:],area[:nlat_s,:],nlat_s,planet_radius=planet_radius)
                qref[-nlat_s:] = -qref2[::-1]
            elif domain_size == 'full_globe':
                qref = eqv_lat_core(ylat, pv_field, area, self.nlat,
                                    planet_radius=planet_radius)
            else:
                raise ValueError('Domain size is not properly specified.')

            self.eqvlat[k, :] = qref
        return self.eqvlat

    def lwa(self):
        """
        Compute the finite-amplitude local wave activity on each pseudoheight layer based on the *equivalent_latitudes* and the *qgpv_field* stored in the object.

        :returns: an 3-D numpy array with dimension (*nlev*,*nlat*,*nlon*) of local wave activity values.

        :example:
        >>> qgfield = QGField(xlon, ylat, np.array([240.]), u, qgpv_field=QGPV)
        >>> qgfield_lwa = qgfield.lwa()

        """

        try:
            self.eqvlat
        except:
            self.eqvlat = self.equivalent_latitudes(domain_size='half_globe')

        lwact = np.zeros((self.nlev, self.nlat, self.nlon))

        for k in range(self.nlev):
            pv_field = self.qgpv_field[k, :, :]
            for j in np.arange(0, self.nlat-1):
                vort_e = pv_field[:, :]-self.eqvlat[k, j]
                vort_boo = np.zeros((self.nlat, self.nlon))
                vort_boo[np.where(vort_e[:, :] < 0)] = -1
                vort_boo[:j+1, :] = 0
                vort_boo[np.where(vort_e[:j+1, :] > 0)] = 1
                lwact[k, j, :] = np.sum(vort_e*vort_boo * self.planet_radius *
                                        self.clat[:, np.newaxis] *
                                        self.dphi[:, np.newaxis], axis=0)
        return lwact


def main():
    from netCDF4 import Dataset
    import numpy as np
    import matplotlib.pyplot as plt

    # === List of tests ===
    test_2D = False
    test_3D = True

    # === Testing the 2D object ===
    if test_2D:
        data_path = '../examples/barotropic_vorticity.nc'
        readFile = Dataset(data_path, mode='r')
        abs_vorticity = readFile.variables['absolute_vorticity'][:]

        xlon = np.linspace(0, 360., 512, endpoint=False)
        ylat = np.linspace(-90, 90., 256, endpoint=True)
        nlon = xlon.size
        nlat = ylat.size
        Earth_radius = 6.378e+6
        dphi = (ylat[2]-ylat[1])*pi/180.
        area = 2.*pi*Earth_radius**2 * (np.cos(ylat[:, np.newaxis]*pi/180.)
                                        * dphi)/float(nlon) * np.ones((nlat, nlon))

        cc1 = BarotropicField(xlon, ylat, pv_field=abs_vorticity)  # area computed in the class assumed uniform grid

        # Compute equivalent latitudes
        cc1_eqvlat = cc1.equivalent_latitudes()

        # Compute equivalent latitudes
        cc1_lwa = cc1.lwa()

        # --- Color axis for plotting LWA --- #
        LWA_caxis = np.linspace(0, cc1_lwa.max(), 31, endpoint=True)

        # --- Plot the abs. vorticity field, LWA and equivalent-latitude relationship and LWA --- #
        fig = plt.subplots(figsize=(14, 4))

        plt.subplot(1, 3, 1)  # Absolute vorticity map
        c = plt.contourf(xlon, ylat, cc1.pv_field, 31)
        cb = plt.colorbar(c)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('right')
        cb.update_ticks()
        plt.title('Absolute vorticity [1/s]')
        plt.xlabel('Longitude (degree)')
        plt.ylabel('Latitude (degree)')

        plt.subplot(1, 3, 2)  # LWA (full domain)
        plt.contourf(xlon, ylat, cc1_lwa, LWA_caxis)
        plt.colorbar()
        plt.title('Local Wave Activity [m/s]')
        plt.xlabel('Longitude (degree)')
        plt.ylabel('Latitude (degree)')

        plt.subplot(1, 3, 3)  # Equivalent-latitude relationship Q(y)
        plt.plot(cc1_eqvlat, ylat, 'b', label='Equivalent-latitude relationship')
        plt.plot(np.mean(cc1.pv_field, axis=1), ylat, 'g', label='zonal mean abs. vorticity')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ylim(-90, 90)
        plt.legend(loc=4, fontsize=10)
        plt.title('Equivalent-latitude profile')
        plt.ylabel('Latitude (degree)')
        plt.xlabel('Q(y) [1/s] | y = latitude')
        plt.tight_layout()
        plt.show()

    # === Testing the 2D object ===
    if test_3D:
        print('Test QGField')
        u_QGPV_File = Dataset('../examples/u_QGPV_240hPa_2012Oct28to31.nc', mode='r')
        # --- Read in longitude and latitude arrays --- #
        xlon = u_QGPV_File.variables['longitude'][:]
        ylat = u_QGPV_File.variables['latitude'][:]
        clat = np.abs(np.cos(ylat*pi/180.)) # cosine latitude
        nlon = xlon.size
        nlat = ylat.size

        u = u_QGPV_File.variables['U'][0, ...]
        QGPV = u_QGPV_File.variables['QGPV'][0, ...]
        u_QGPV_File.close()

        print(u.shape)
        print(QGPV.shape)

        cc2 = QGField(xlon, ylat, np.array([240.]), u, qgpv_field=QGPV)  # area computed in the class assumed uniform grid
        cc3 = cc2.lwa()
        print('cc3 shape')
        print(cc3.shape)

        # print 'test empty qgpv fields'
        # cc4 = QGField(xlon, ylat, np.array([240.]), u)

        plt.figure(figsize=(8, 3))
        c = plt.contourf(xlon, ylat[80:], cc3[0, 80:, :], 31)
        cb = plt.colorbar(c)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('right')
        cb.update_ticks()
        plt.title('Local Wave Activity at 240hPa [m/s]')
        plt.xlabel('Longitude (degree)')
        plt.ylabel('Latitude (degree)')
        plt.show()





if __name__ == "__main__":
    main()
