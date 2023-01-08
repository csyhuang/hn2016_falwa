SUBROUTINE compute_lwa_and_barotropic_fluxes(nlon, nlat, kmax, jd, &
                          pv,uu,vv,pt,qref,uref,tref,&
                         a, om, dz, h, r, cp, prefactor, &
                         astar,astarbaro,ua1baro,ubaro,ua2baro,ep1baro,ep2baro,ep3baro,ep4)

    implicit none

    integer, intent(in) :: nlon, nlat, kmax, jd
    real, intent(in) :: pv(nlon,nlat,kmax),uu(nlon,nlat,kmax),vv(nlon,nlat,kmax), &
                        pt(nlon,nlat,kmax),qref(jd,kmax),uref(jd,kmax),tref(jd,kmax)
    real, intent(in) ::  a, om, dz, h, r, cp, prefactor
    real, intent(out) :: astar(nlon,jd,kmax),astarbaro(nlon,jd),ua1baro(nlon,jd),ubaro(nlon,jd),&
                         ua2baro(nlon,jd),ep1baro(nlon,jd),ep2baro(nlon,jd),ep3baro(nlon,jd),ep4(nlon,jd)
    ! === dummy variables ===
    integer :: i, j, jj, k
    real :: qe(nlon,nlat),ue(nlon,nlat)
    real :: ua1(nlon,jd,kmax)
    real :: ua2(nlon,jd,kmax),ep1(nlon,jd,kmax)
    real :: ep2(nlon,jd,kmax),ep3(nlon,jd,kmax)
    real :: tg(kmax)
    real :: aa, cor, dc, ep11, ep41, ep42, ep43, wt, zk, zz, dp, rkappa
    real :: cosphi(0:jd+1), sinphi(0:jd+1)
    real, parameter :: pi = acos(-1.)

    dp = pi/float(nlat-1)
    rkappa = r/cp

    ! cos and sin of latitude are often used in the following computations.
    ! Because the evaluation of trigonometric functions is computationally
    ! expensive, values are computed once here and stored for later.
    cosphi(0) = cos(0.) ! fill values for meridional
    sinphi(0) = sin(0.) !  heat flux computation
    do j=1,jd+1
        cosphi(j) = cos(dp*float(j-1))
        sinphi(j) = sin(dp*float(j-1))
    enddo

    ! **** hemispheric-mean potential temperature ****

    tg(:) = 0.              ! mean PT
    wt = 0.                 ! area weight
    do j = 1,jd             ! NH
        tg(:) = tg(:)+cosphi(j)*tref(j,:)
        wt = wt + cosphi(j)
    enddo
    tg(:) = tg(:)/wt       ! averaging

    ! **** wave activity and nonlinear zonal flux F2 ****

    do k = 2,kmax-1
        do i = 1,nlon
            do j = 1,jd-1            ! 13.5N and higher latitude
                astar(i,j,k) = 0.       ! LWA*cos(phi)
                ua2(i,j,k) = 0.         ! F2
                cor = 2.*om*sinphi(j)          !Coriolis parameter
                ! South of the current latitude
                do jj = 1,j-1
                    qe(i,jj) = pv(i,jj+jd-1,k)-qref(j,k)*cor   !qe; Q = qref*cor
                    ue(i,jj) = uu(i,jj+jd-1,k)-uref(j,k)       !ue
                    aa = a*dp*cosphi(jj)                       !length element
                    if(qe(i,jj).gt.0.) then                    !LWA*cos(phi) and F2
                        astar(i,j,k)=astar(i,j,k)+qe(i,jj)*aa
                        ua2(i,j,k) = ua2(i,j,k)+qe(i,jj)*ue(i,jj)*aa
                    endif  
                enddo
                ! North of the current latitude
                do jj = j,jd
                    qe(i,jj) = pv(i,jj+jd-1,k)-qref(j,k)*cor   !qe; Q = qref*cor
                    ue(i,jj) = uu(i,jj+jd-1,k)-uref(j,k)       !ue
                    aa = a*dp*cosphi(jj)                       !length element
                    if(qe(i,jj).le.0.) then                    !LWA*cos(phi) and F2
                        astar(i,j,k)=astar(i,j,k)-qe(i,jj)*aa
                        ua2(i,j,k) = ua2(i,j,k)-qe(i,jj)*ue(i,jj)*aa
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
                ep1(i,j,k) = ep1(i,j,k)*cosphi(j) ! correct for cosine factor

                ! meridional eddy momentum flux one grid north and south
                if (j.gt.1) then  ! only compute ep2 and ep3 for interior points (on latitude grid)
                    ep2(i,j,k) = (uu(i,j+jd,k)-uref(j+1,k))*cosphi(j+1)*cosphi(j+1) * vv(i,j+jd,k)
                    ep3(i,j,k) = (uu(i,j+jd-2,k)-uref(j-1,k))*cosphi(j-1)*cosphi(j-1) * vv(i,j+jd-2,k)
                end if

                ! low-level meridional eddy heat flux
                if(k.eq.2) then     ! (26) of SI-HN17
                    ep41 = 2.*om*sinphi(j)*cosphi(j)*dz/prefactor       ! prefactor
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
