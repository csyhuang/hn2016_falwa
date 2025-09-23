SUBROUTINE compute_qgpv(nlon, nlat, kmax, ut, vt, theta, height, t0, stat, &
                        aa, omega, dz, hh, rr, cp, &
                        pv, avort)

   
    INTEGER, INTENT(IN) :: nlon, nlat, kmax
    REAL, INTENT(IN) :: ut(kmax,nlat,nlon), vt(kmax,nlat,nlon), theta(kmax,nlat,nlon), &
                        height(kmax), t0(kmax), stat(kmax)
    REAL, INTENT(in) :: aa, omega, hh, rr, cp
    REAL, INTENT(out) :: pv(kmax,nlat,nlon), avort(kmax,nlat,nlon)


    REAL ::  tz(kmax,nlat),tzd(kmax,nlat)
    REAL ::  uz(kmax,nlat),uzd(kmax,nlat)
    REAL ::  vz(kmax,nlat),vzd(kmax,nlat)
    REAL ::  st(nlat,nlon),zmst(nlat)
    REAL ::  zmav(kmax,nlat)
    REAL ::  zmpv(kmax,nlat)
    REAL :: rkappa, pi, dphi


    rkappa = rr/cp
    pi = acos(-1.)
    dphi = pi/float(nlat-1)

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
                tzd(k,j) = tzd(k,j) + theta(k,j,i)/float(nlon)
                uzd(k,j) = uzd(k,j) + ut(k,j,i)/float(nlon)
                vzd(k,j) = vzd(k,j) + vt(k,j,i)/float(nlon)
                tz(k,j) = tz(k,j)+theta(k,j,i)/(float(nlon))
                uz(k,j) = uz(k,j)+ut(k,j,i)/(float(nlon))
                vz(k,j) = vz(k,j)+vt(k,j,i)/(float(nlon))
            enddo
        enddo
    enddo

    ! surface temp
    do j = 1,nlat
        zmst(j) = 0.
        do i = 1,nlon
            zmst(j) = zmst(j) + st(j,i)/float(nlon)
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
                av2 = (vt(kk,j,i+1)-vt(kk,j,i-1))/(2.*aa*cos(phi0)*dphi)
                av3 = -(ut(kk,j+1,i)*cos(phip)-ut(kk,j-1,i)*cos(phim))/(2.*aa*cos(phi0)*dphi)
                avort(kk,j,i) = av1+av2+av3
            enddo

            av1 = 2.*omega*sin(phi0)
            av2 = (vt(kk,j,2)-vt(kk,j,nlon))/(2.*aa*cos(phi0)*dphi)
            av3 = -(ut(kk,j+1,1)*cos(phip)-ut(kk,j-1,1)*cos(phim))/(2.*aa*cos(phi0)*dphi)
            avort(kk,j,1) = av1+av2+av3
            av4 = 2.*omega*sin(phi0)
            av5 = (vt(kk,j,1)-vt(kk,j,nlon-1))/(2.*aa*cos(phi0)*dphi)
            av6 =   &
            -(ut(kk,j+1,nlon)*cos(phip)-ut(kk,j-1,nlon)*cos(phim))/(2.*aa*cos(phi0)*dphi)
            avort(kk,j,nlon) = av4+av5+av6
        enddo

        avs = 0.
        avn = 0.
        do i = 1,nlon
            avs = avs + avort(kk,2,i)/float(nlon)
            avn = avn + avort(kk,nlat-1,i)/float(nlon)
        enddo
        avort(kk,1,:) = avs
        avort(kk,nlat,:) = avn
    enddo

    ! zonal mean vort

    do kk = 1,kmax
        do j = 1,nlat
            zmav(kk,j) = 0.
            do i = 1,nlon
                zmav(kk,j) = zmav(kk,j)+avort(kk,j,i)/float(nlon)
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
                altp = exp(-height(kk+1)/hh)*(theta(kk+1,j,i)-t0(kk+1))/stat(kk+1)
                altm = exp(-height(kk-1)/hh)*(theta(kk-1,j,i)-t0(kk-1))/stat(kk-1)
                altp = altp
                altm = altm
                strc = (altp-altm)*zmav(kk,j)/(height(kk+1)-height(kk-1))
                !     strc = (altp-altm)*f/(height(kk+1)-height(kk-1))
                pv(kk,j,i) = avort(kk,j,i) + exp(height(kk)/hh)*strc
            enddo
        enddo
    enddo

    ! zonal mean pv

    do kk = 1,kmax
        do j = 1,nlat
            zmpv(kk,j) = 0.
            do i = 1,nlon
                zmpv(kk,j) = zmpv(kk,j)+pv(kk,j,i)/float(nlon)
            enddo
        enddo
    enddo

END SUBROUTINE