SUBROUTINE compute_lwa_and_barotropic_fluxes(nlon, nlat, kmax, jd, &
                          pv,uu,vv,pt,qref,uref,tref,&
                         a, om, dz, h, r, cp, prefactor, &
                         astar,astarbaro,ua1baro,ubaro,ua2baro,ep1baro,ep2baro,ep3baro,ep4)

    integer, intent(in) :: nlon, nlat, kmax, jd
    real, intent(in) :: pv(nlon,nlat,kmax),uu(nlon,nlat,kmax),vv(nlon,nlat,kmax), &
                        pt(nlon,nlat,kmax),qref(jd,kmax),uref(jd,kmax),tref(jd,kmax)
    real, intent(in) ::  a, om, dz, h, r
    real, intent(out) :: astar(nlon,jd,kmax),astarbaro(nlon,jd),ua1baro(nlon,jd),ubaro(nlon,jd),&
                         ua2baro(nlon,jd),ep1baro(nlon,jd),ep2baro(nlon,jd),ep3baro(nlon,jd),ep4(nlon,jd)
    ! === dummy variables ===
    real :: qe(nlon,nlat),ue(nlon,nlat)
    real :: ua1(nlon,jd,kmax)
    real :: ua2(nlon,jd,kmax),ep1(nlon,jd,kmax)
    real :: ep2(nlon,jd,kmax),ep3(nlon,jd,kmax)
    real :: tg(kmax)
    real :: rkappa, zero, half, qtr, one

    pi = acos(-1.)
    dp = pi/float(nlat-1)
    zero = 0.
    half = 0.5
    qtr = 0.25
    one = 1.
    rkappa = r/cp


    ! **** hemispheric-mean potential temperature ****

    tg(:) = 0.              ! mean PT
    wt = 0.                 ! area weight
    do j = 1,jd             ! NH
        phi0 = dp*float(j-1)               ! latitude
        tg(:) = tg(:)+cos(phi0)*tref(j,:)
        wt = wt + cos(phi0)
    enddo
    tg(:) = tg(:)/wt       ! averaging

    ! **** wave activity and nonlinear zonal flux F2 ****

    do k = 2,kmax-1
        do i = 1,nlon
            do j = 1,jd-1            ! 13.5N and higher latitude
                astar(i,j,k) = 0.       ! LWA*cos(phi)
                ua2(i,j,k) = 0.         ! F2
                phi0 = dp*float(j-1)           !latitude
                cor = 2.*om*sin(phi0)          !Coriolis parameter
                do jj = 1,jd
                    phi1 = dp*float(jj-1)
                    qe(i,jj) = pv(i,jj+jd-1,k)-qref(j,k)*cor   !qe; Q = qref*cor
                    ue(i,jj) = uu(i,jj+jd-1,k)-uref(j,k)       !ue
                    aa = a*dp*cos(phi1)                      !length element
                    if((qe(i,jj).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
                        astar(i,j,k)=astar(i,j,k)-qe(i,jj)*aa
                        ua2(i,j,k) = ua2(i,j,k)-qe(i,jj)*ue(i,jj)*aa
                    endif  
                    if((qe(i,jj).gt.0.).and.(jj.lt.j)) then
                        astar(i,j,k)=astar(i,j,k)+qe(i,jj)*aa
                        ua2(i,j,k) = ua2(i,j,k)+qe(i,jj)*ue(i,jj)*aa
                    endif  
                enddo

                !    *********  Other fluxes *********

                ua1(i,j,k) = uref(j,k)*astar(i,j,k)            !F1     
                ep1(i,j,k) = -0.5*(uu(i,j+jd-1,k)-uref(j,k))**2  !F3a
                ep1(i,j,k) = ep1(i,j,k)+0.5*vv(i,j+jd-1,k)**2    !F3a+b
                ep11 = 0.5*(pt(i,j+jd-1,k)-tref(j,k))**2         !F3c
                zz = dz*float(k-1)
                ep11 = ep11*(r/h)*exp(-rkappa*zz/h)
                ep11 = ep11*2.*dz/(tg(k+1)-tg(k-1))
                ep1(i,j,k) = ep1(i,j,k)-ep11                   !F3
                phip = dp*float(j)                       
                cosp = cos(phip)          ! cosine for one grid north
                phi0 = dp*float(j-1)
                cos0 = cos(phi0)          ! cosine for latitude grid
                sin0 = sin(phi0)          ! sine for latitude grid
                phim = dp*float(j-2)
                cosm = cos(phim)          ! cosine for one grid south
                ep1(i,j,k) = ep1(i,j,k)*cos0 ! correct for cosine factor


                ! meridional eddy momentum flux one grid north and south
                ep2(i,j,k)=(uu(i,j+jd,k)-uref(j,k))*vv(i,j+jd,k)*cosp*cosp
                ep3(i,j,k)=(uu(i,j+jd-2,k)-uref(j,k))*vv(i,j+jd-2,k)*cosm*cosm

                ! low-level meridional eddy heat flux
                if(k.eq.2) then     ! (26) of SI-HN17
                    ep41 = 2.*om*sin0*cos0*dz/prefactor       ! prefactor
                    ep42 = exp(-dz/h)*vv(i,j+jd-1,2)*(pt(i,j+jd-1,2)-tref(j,2))
                    ep42 = ep42/(tg(3)-tg(1))
                    ep43 = vv(i,j+jd-1,1)*(pt(i,j+jd-1,1)-tref(j,1))
                    ep43 = 0.5*ep43/(tg(2)-tg(1))
                    ep4(i,j) = ep41*(ep42+ep43)   ! low-level heat flux
                endif
            enddo
        enddo
    enddo

    ! ******** Column average: (25) of SI-HN17 ********

    astarbaro(:,:) = 0.
    ubaro(:,:) = 0.
    ua1baro(:,:) = 0.
    ua2baro(:,:) = 0.
    ep1baro(:,:) = 0.
    ep2baro(:,:) = 0.
    ep3baro(:,:) = 0.
    dc = dz/prefactor

    do k = 2,kmax-1
        zk = dz*float(k-1)
        astarbaro(:,:) = astarbaro(:,:)+astar(:,:,k)*exp(-zk/h)*dc
        ua1baro(:,:) = ua1baro(:,:)+ua1(:,:,k)*exp(-zk/h)*dc
        ua2baro(:,:) = ua2baro(:,:)+ua2(:,:,k)*exp(-zk/h)*dc
        ep1baro(:,:) = ep1baro(:,:)+ep1(:,:,k)*exp(-zk/h)*dc
        ep2baro(:,:) = ep2baro(:,:)+ep2(:,:,k)*exp(-zk/h)*dc
        ep3baro(:,:) = ep3baro(:,:)+ep3(:,:,k)*exp(-zk/h)*dc
        do j = 1,jd  ! ### yet to be multiplied by cosine
            ubaro(:,j) = ubaro(:,j)+uu(:,j+jd-1,k)*exp(-zk/h)*dc
        enddo
    enddo


end subroutine
