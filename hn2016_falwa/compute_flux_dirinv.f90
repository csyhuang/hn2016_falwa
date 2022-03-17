SUBROUTINE compute_flux_dirinv(pv,uu,vv,pt,tn0,ts0,statn,stats,qref,uref,tref,fawa,ubar,tbar,&
        imax, JMAX, kmax, nd, nnd, jb, jd, &
        a, om, dz, h, rr, cp, prefac,&
        astarbaro,ubaro,urefbaro,ua1baro,ua2baro,ep1baro,ep2baro,ep3baro,ep4,astar1,astar2)

  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, nnd, jb, jd
  REAL, INTENT(IN) :: pv(imax,jmax,kmax),uu(imax,jmax,kmax),vv(imax,jmax,kmax),pt(imax,jmax,kmax),&
          tn0(kmax),ts0(kmax),statn(kmax),stats(kmax),qref(nd,kmax),uref(jd,kmax),tref(jd,kmax),&
          fawa(nd,kmax),ubar(nd,kmax),tbar(nd,kmax)
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astarbaro(imax,nd),ubaro(imax,nd),urefbaro(nd),ua1baro(imax,nd),ua2baro(imax,nd),&
          ep1baro(imax,nd),ep2baro(imax,nd),ep3baro(imax,nd),ep4(imax,nd),astar1(imax,nd,kmax),astar2(imax,nd,kmax)

  REAL :: tb(kmax),tg(kmax)
  REAL :: ua1(imax,nd),ua2(imax,nd),ep1(imax,nd)
  REAL :: ep2(imax,nd),ep3(imax,nd)
  REAL :: qe(imax,nd),ue(imax,nd)
  REAL :: z(kmax)
  REAL :: u(nd,kmax)

  !a = 6378000.
  pi = acos(-1.)
  !om = 7.29e-5
  dp = pi/float(jmax-1)
  !dz = 500.
  !h = 7000.
  !r = 287.
  rkappa = rr/cp
  !prefac = 6745.348

  ! *** Default values for boundary ***
  !jb = 5
  !jd = 86 ! nd - lower bounding latitude

  do k = 1,kmax
   z(k) = dz*float(k-1)
  enddo


! **** hemispheric-mean potential temperature ****
  tg(:) = tn0(:)

! **** wave activity and nonlinear zonal flux F2 ****

  astarbaro(:,:) = 0.
  ubaro(:,:) = 0.
  urefbaro(:) = 0.
  ua1baro(:,:) = 0.
  ua2baro(:,:) = 0.
  ep1baro(:,:) = 0.
  ep2baro(:,:) = 0.
  ep3baro(:,:) = 0.
  ep4(:,:) = 0.
  dc = dz/prefac

  do k = 2,kmax-1
    zk = dz*float(k-1)
    do i = 1,imax
      do j = jb+1,nd-1            ! 5N and higher latitude
        astar1(i,j,k) = 0.       ! LWA*cos(phi)
        astar2(i,j,k) = 0.       ! LWA*cos(phi)
        ua2(i,j) = 0.          !F2
        phi0 = dp*float(j-1)           !latitude
        cor = 2.*om*sin(phi0)          !Coriolis parameter
        do jj = 1,nd
          phi1 = dp*float(jj-1)
          qe(i,jj) = pv(i,jj+nd-1,k)-qref(j,k)   !qe; Q = qref
          ue(i,jj) = uu(i,jj+nd-1,k)-uref(j-jb,k)   !ue; shift uref 5N
          aa = a*dp*cos(phi1)                      !length element
          if((qe(i,jj).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            astar2(i,j,k)=astar2(i,j,k)-qe(i,jj)*aa  !anticyclonic
            ua2(i,j) = ua2(i,j)-qe(i,jj)*ue(i,jj)*aa
          endif
          if((qe(i,jj).gt.0.).and.(jj.lt.j)) then
            astar1(i,j,k)=astar1(i,j,k)+qe(i,jj)*aa  !cyclonic
            ua2(i,j) = ua2(i,j)+qe(i,jj)*ue(i,jj)*aa
          endif
        enddo

        !  ******** Other fluxes ******

        ua1(i,j) = uref(j-jb,k)*(astar1(i,j,k) +  &
        astar2(i,j,k))            !F1
        ep1(i,j) = -0.5*(uu(i,j+nd-1,k)-uref(j-jb,k))**2  !F3a
        ep1(i,j) = ep1(i,j)+0.5*vv(i,j+nd-1,k)**2    !F3a+b
        ep11 = 0.5*(pt(i,j+nd-1,k)-tref(j-jb,k))**2         !F3c
        zz = dz*float(k-1)
        ep11 = ep11*(rr/h)*exp(-rkappa*zz/h)
        ep11 = ep11*2.*dz/(tg(k+1)-tg(k-1))
        ep1(i,j) = ep1(i,j)-ep11                   !F3
        phip = dp*float(j)
        cosp = cos(phip)          ! cosine for one grid north
        phi0 = dp*float(j-1)
        cos0 = cos(phi0)          ! cosine for latitude grid
        sin0 = sin(phi0)          ! sine for latitude grid
        phim = dp*float(j-2)
        cosm = cos(phim)          ! cosine for one grid south
        ep1(i,j) = ep1(i,j)*cos0 ! correct for cosine factor


        ! meridional eddy momentum flux one grid north and south
        ep2(i,j)=(uu(i,j+91,k)-uref(j-jb,k))*vv(i,j+91,k)*cosp*cosp
        ep3(i,j)=(uu(i,j+89,k)-uref(j-jb,k))*vv(i,j+89,k)*cosm*cosm

        ! low-level meridional eddy heat flux
        if(k.eq.2) then     ! (26) of SI-HN17
          ep41 = 2.*om*sin0*cos0*dz/prefac       ! prefactor
          ep42 = exp(-dz/h)*vv(i,j+nd-1,2)*(pt(i,j+nd-1,2)-tref(j-jb,2))
          ep42 = ep42/(tg(3)-tg(1))
          ep43 = vv(i,j+nd-1,1)*(pt(i,j+nd-1,1)-tref(j-jb,1))
          ep43 = 0.5*ep43/(tg(2)-tg(1))
          ep4(i,j) = ep41*(ep42+ep43)   ! low-level heat flux
        endif
      enddo
    enddo

    ! ******** Column average: (25) of SI-HN17 ********

    astarbaro(:,:) = astarbaro(:,:)+(astar1(:,:,k)    &
    + astar2(:,:,k))*exp(-zk/h)*dc
    ua1baro(:,:) = ua1baro(:,:)+ua1(:,:)*exp(-zk/h)*dc
    ua2baro(:,:) = ua2baro(:,:)+ua2(:,:)*exp(-zk/h)*dc
    ep1baro(:,:) = ep1baro(:,:)+ep1(:,:)*exp(-zk/h)*dc
    ep2baro(:,:) = ep2baro(:,:)+ep2(:,:)*exp(-zk/h)*dc
    ep3baro(:,:) = ep3baro(:,:)+ep3(:,:)*exp(-zk/h)*dc
    do j = jb+1,nd  ! ### yet to be multiplied by cosine
      ubaro(:,j) = ubaro(:,j)+uu(:,j+nd-1,k)*exp(-zk/h)*dc
      urefbaro(j) = urefbaro(j)+uref(j-jb,k)*exp(-zk/h)*dc
    enddo
  enddo

END SUBROUTINE
