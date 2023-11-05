SUBROUTINE interpolate_fields(nlon, nlat, nlev, kmax, uu, vv, temp, plev, height, &
                              aa, omega, dz, hh, rr, cp, pv, ut, vt, avort, theta, stat)

   
    INTEGER, INTENT(IN) :: nlon, nlat, nlev, kmax
    REAL, INTENT(IN) :: uu(nlon,nlat,nlev), vv(nlon,nlat,nlev), temp(nlon,nlat,nlev), &
                        plev(nlev), height(kmax)
    REAL, INTENT(in) :: aa, omega, hh, rr, cp
    REAL, INTENT(out) :: pv(nlon,nlat,kmax), ut(nlon,nlat,kmax), vt(nlon,nlat,kmax), avort(nlon,nlat,kmax)
    REAL, INTENT(out) :: theta(nlon,nlat,kmax), stat(kmax)


    REAL ::  tt(nlon,nlat,nlev),tz(nlat,kmax),tzd(nlat,kmax)
    REAL ::  uz(nlat,kmax),uzd(nlat,kmax)
    REAL ::  vz(nlat,kmax),vzd(nlat,kmax)
    REAL ::  t0(kmax),zlev(nlev)
    REAL ::  st(nlon,nlat),zmst(nlat)
    REAL ::  zmav(nlat,kmax)
    REAL ::  zmpv(nlat,kmax)
    REAL :: rkappa, pi, dphi


    rkappa = rr/cp
    pi = acos(-1.)
    dphi = pi/float(nlat-1)


    ! ====  vertical interpolation ====
    do k = 1,nlev
        zlev(k) = -hh*alog(plev(k)/1000.)
    enddo

    do i = 1,nlon
        do j = 1,nlat

            do k = 1,nlev
                tt(i,j,k) = temp(i,j,k)*((1000./plev(k))**rkappa)
            enddo



            st(i,j) = tt(i,j,1)      ! surface pot. temp

            do kk = 1,kmax   ! vertical interpolation 
                ttt = height(kk)
                do k = 1,nlev-1
                    tt2 = zlev(k+1)
                    tt1 = zlev(k)
                    if((ttt.ge.tt1).and.(ttt.lt.tt2)) then
                        dd1 = (ttt-tt1)/(tt2-tt1)
                        dd2 = 1.-dd1
                        theta(i,j,kk) = tt(i,j,k)*dd2 + tt(i,j,k+1)*dd1
                        ut(i,j,kk) = uu(i,j,k)*dd2 + uu(i,j,k+1)*dd1
                        vt(i,j,kk) = vv(i,j,k)*dd2 + vv(i,j,k+1)*dd1
                        goto 348
                    endif
                enddo
                348 continue
            enddo
        enddo
    enddo

    !  **** compute zonal mean ****
    tz = 0.
    uz = 0.
    vz = 0.

    tzd = 0.
    uzd = 0.
    vzd = 0.


    do j = 1,nlat
        do k = 1,kmax
            do i = 1,nlon
                tzd(j,k) = tzd(j,k) + theta(i,j,k)/float(nlon)
                uzd(j,k) = uzd(j,k) + ut(i,j,k)/float(nlon)
                vzd(j,k) = vzd(j,k) + vt(i,j,k)/float(nlon)
                tz(j,k) = tz(j,k)+theta(i,j,k)/(float(nlon))
                uz(j,k) = uz(j,k)+ut(i,j,k)/(float(nlon))
                vz(j,k) = vz(j,k)+vt(i,j,k)/(float(nlon))
            enddo
        enddo
    enddo


    ! reference theta
    do kk = 1,kmax
        t0(kk) = 0.
        csm = 0.
        do j = 1,nlat
            phi0 = -90.+float(j-1)*180./float(nlat-1)
            phi0 = phi0*pi/180.
            t0(kk) = t0(kk) + tzd(j,kk)*cos(phi0)
            csm = csm + cos(phi0)
        enddo
        t0(kk) = t0(kk)/csm
    enddo

    ! static stability
    do kk = 2,kmax-1
        stat(kk) = (t0(kk+1)-t0(kk-1))/(height(kk+1)-height(kk-1))
    enddo
    stat(kmax) = (t0(kmax)-t0(kmax-1))/(height(kmax)-height(kmax-1))
    stat(1) = (t0(2)-t0(1))/(height(2)-height(1))


    ! surface temp

    do j = 1,nlat
        zmst(j) = 0.
        do i = 1,nlon
            zmst(j) = zmst(j) + st(i,j)/float(nlon)
        enddo
    enddo

    ! interior abs. vort

    do kk = 1,kmax
        do j = 2,nlat-1
            phi0 = -90.+float(j-1)*180./float(nlat-1)
            phi0 = phi0*pi/180.
            phim = -90.+float(j-2)*180./float(nlat-1)
            phim = phim*pi/180.
            phip = -90.+float(j)*180./float(nlat-1)
            phip = phip*pi/180.

            do i = 2,nlon-1
                av1 = 2.*omega*sin(phi0)
                av2 = (vt(i+1,j,kk)-vt(i-1,j,kk))/(2.*aa*cos(phi0)*dphi)
                av3 = -(ut(i,j+1,kk)*cos(phip)-ut(i,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)  
                avort(i,j,kk) = av1+av2+av3
            enddo

            av1 = 2.*omega*sin(phi0)
            av2 = (vt(2,j,kk)-vt(nlon,j,kk))/(2.*aa*cos(phi0)*dphi)
            av3 = -(ut(1,j+1,kk)*cos(phip)-ut(1,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)  
            avort(1,j,kk) = av1+av2+av3
            av4 = 2.*omega*sin(phi0)
            av5 = (vt(1,j,kk)-vt(nlon-1,j,kk))/(2.*aa*cos(phi0)*dphi)
            av6 =   & 
            -(ut(nlon,j+1,kk)*cos(phip)-ut(nlon,j-1,kk)*cos(phim))/(2.*aa*cos(phi0)*dphi)
            avort(nlon,j,kk) = av4+av5+av6
        enddo

        avs = 0.
        avn = 0.
        do i = 1,nlon
            avs = avs + avort(i,2,kk)/float(nlon)
            avn = avn + avort(i,nlat-1,kk)/float(nlon)
        enddo
        avort(:,1,kk) = avs
        avort(:,nlat,kk) = avn
    enddo

    ! zonal mean vort

    do kk = 1,kmax
        do j = 1,nlat
            zmav(j,kk) = 0.
            do i = 1,nlon
                zmav(j,kk) = zmav(j,kk)+avort(i,j,kk)/float(nlon)
            enddo
        enddo
    enddo

    ! interior pv

    do kk = 2,kmax-1
        do j = 1,nlat
            phi0 = -90.+float(j-1)*180./float(nlat-1)
            phi0 = phi0*pi/180.
            f = 2.*omega*sin(phi0)
            do i = 1,nlon
                altp = exp(-height(kk+1)/hh)*(theta(i,j,kk+1)-t0(kk+1))/stat(kk+1)
                altm = exp(-height(kk-1)/hh)*(theta(i,j,kk-1)-t0(kk-1))/stat(kk-1)
                altp = altp
                altm = altm
                strc = (altp-altm)*zmav(j,kk)/(height(kk+1)-height(kk-1))
                !     strc = (altp-altm)*f/(height(kk+1)-height(kk-1))
                pv(i,j,kk) = avort(i,j,kk) + exp(height(kk)/hh)*strc
            enddo
        enddo
    enddo

    ! zonal mean pv

    do kk = 1,kmax
        do j = 1,nlat
            zmpv(j,kk) = 0.
            do i = 1,nlon
                zmpv(j,kk) = zmpv(j,kk)+pv(i,j,kk)/float(nlon)
            enddo
        enddo
    enddo


!     write(6,*) 'U ',maxval(uz),minval(uz)
!     write(6,*) 'V ',maxval(vz),minval(vz)
!     write(6,*) 'T ',maxval(tz),minval(tz)    
!     write(6,*) 'AVORT',maxval(avort),minval(avort)    
!     write(6,*) 'PV ',maxval(zmpv),minval(zmpv)
!     write(6,*) 'zmpv', sum(zmpv, dim=1)

END SUBROUTINE