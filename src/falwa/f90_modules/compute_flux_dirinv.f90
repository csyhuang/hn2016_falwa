SUBROUTINE compute_flux_dirinv_nshem( &
    pv, uu, vv, pt, ncforce, tn0, qref, uref, tref, &
    imax, JMAX, kmax, nd, jb, jd, is_nhem, &
    a, om, dz, h, rr, cp, prefac, &
    astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4)

  REAL, INTENT(IN) :: pv(imax,jmax,kmax),uu(imax,jmax,kmax),vv(imax,jmax,kmax),pt(imax,jmax,kmax), &
          ncforce(imax, jmax, kmax), &
          tn0(kmax),qref(nd,kmax),uref(jd,kmax),tref(jd,kmax)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, jb, jd
  LOGICAL, INTENT(IN) :: is_nhem
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astar1(imax,nd,kmax), astar2(imax,nd,kmax), ncforce3d(imax,nd,kmax), &
          ua1(imax,nd,kmax),  ua2(imax,nd,kmax), ep1(imax,nd,kmax), ep2(imax,nd,kmax), ep3(imax,nd,kmax), ep4(imax,nd)

  REAL :: tg(kmax)
  REAL :: qe(imax,nd),ue(imax,nd)
  REAL :: ncforce2d(imax,nd)
  REAL :: z(kmax)
  REAL :: aa, ab
  INTEGER :: jstart, jend

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

! **** Computing ncforce (testing) ***
  ncforce2d(:,:) = 0.0
  ncforce3d(:,:,:) = 0.0

! **** wave activity and nonlinear zonal flux F2 ****
  astar1(:,:,:) = 0.
  astar2(:,:,:) = 0.
  ncforce3d(:,:,:) = 0.
  ua1(:,:,:) = 0.
  ua2(:,:,:) = 0.
  ep1(:,:,:) = 0.
  ep2(:,:,:) = 0.
  ep3(:,:,:) = 0.
  ep4(:,:) = 0.
  dc = dz/prefac

  ! Bounds of y-indices in N/SHem
  if (is_nhem) then  ! 5N and higher latitude
    jstart = jb+1  ! 6
    jend = nd-1
  else
    jstart = 2
    jend = nd-jb   ! nd - 5
  endif

  do k = 2,kmax-1
    zk = dz*float(k-1)
    do i = 1,imax
      do j = jstart, jend
        astar1(i,j,k) = 0.       ! LWA*cos(phi)
        astar2(i,j,k) = 0.       ! LWA*cos(phi)
        ncforce3d(i,j,k) = 0.
        ua2(i,j,k) = 0.          !F2
        if (is_nhem) then        !latitude
          phi0 = dp*float(j-1)
        else
          phi0 = dp*float(j-1)-0.5*pi
        endif
        cor = 2.*om*sin(phi0)          !Coriolis parameter
        ab = a*dp  !constant length element
        do jj = 1,nd
          if (is_nhem) then  ! Northern Hemisphere
            phi1 = dp*float(jj-1)
            qe(i,jj) = pv(i,jj+nd-1,k)-qref(j,k)    !qe; Q = qref
            ncforce2d(i,jj) = ncforce(i,jj+nd-1,k)
            ue(i,jj) = uu(i,jj+nd-1,k)*cos(phi0)-uref(j-jb,k)*cos(phi1) !ue; shift uref 5N
          else  ! Southern Hemisphere
            phi1 = dp*float(jj-1)-0.5*pi
            qe(i,jj) = pv(i,jj,k)-qref(j,k)     !qe; Q = qref
            ncforce2d(i,jj) = ncforce(i,jj,k)
            ue(i,jj) = uu(i,jj,k)*cos(phi0)-uref(j,k)*cos(phi1)  !ue;
          endif
          aa = a*dp*cos(phi1)   !cosine factor in the meridional integral
          if((qe(i,jj).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            if (is_nhem) then  ! Northern Hemisphere
              astar2(i,j,k)=astar2(i,j,k)-qe(i,jj)*aa  !anticyclonic
            else  ! Southern Hemisphere
              astar1(i,j,k)=astar1(i,j,k)-qe(i,jj)*aa  !cyclonic
            endif
            ncforce3d(i,j,k)=ncforce3d(i,j,k)-ncforce2d(i,jj)*aa
            ua2(i,j,k) = ua2(i,j,k)-qe(i,jj)*ue(i,jj)*ab
          endif
          if((qe(i,jj).gt.0.).and.(jj.lt.j)) then
            if (is_nhem) then  ! Northern Hemisphere
              astar1(i,j,k)=astar1(i,j,k)+qe(i,jj)*aa  !cyclonic
            else  ! Southern Hemisphere
              astar2(i,j,k)=astar2(i,j,k)+qe(i,jj)*aa  !anticyclonic
            endif
            ncforce3d(i,j,k)=ncforce3d(i,j,k)+ncforce2d(i,jj)*aa
            ua2(i,j,k) = ua2(i,j,k)+qe(i,jj)*ue(i,jj)*ab
          endif
        enddo

        !  ******** Other fluxes ******
        if (is_nhem) then
          ua1(i,j,k) = uref(j-jb,k)*(astar1(i,j,k) + astar2(i,j,k))  !F1
          ep1(i,j,k) = -0.5*(uu(i,j+nd-1,k)-uref(j-jb,k))**2         !F3a
          ep1(i,j,k) = ep1(i,j,k)+0.5*vv(i,j+nd-1,k)**2                !F3a+b
          ep11 = 0.5*(pt(i,j+nd-1,k)-tref(j-jb,k))**2              !F3c
        else
          ua1(i,j,k) = uref(j,k)*(astar1(i,j,k) + astar2(i,j,k))     !F1
          ep1(i,j,k) = -0.5*(uu(i,j,k)-uref(j,k))**2                 !F3a
          ep1(i,j,k) = ep1(i,j,k)+0.5*vv(i,j,k)**2                     !F3a+b
          ep11 = 0.5*(pt(i,j,k)-tref(j,k))**2               !F3c
        endif
        zz = dz*float(k-1)
        ep11 = ep11*(rr/h)*exp(-rkappa*zz/h)
        ep11 = ep11*2.*dz/(tg(k+1)-tg(k-1))
        ep1(i,j,k) = ep1(i,j,k)-ep11                   !F3
        if (is_nhem) then
          phip = dp*float(j)
          phi0 = dp*float(j-1)
          phim = dp*float(j-2)
        else
          phip = dp*float(j) - 0.5*pi
          phi0 = dp*float(j-1) - 0.5*pi
          phim = dp*float(j-2) - 0.5*pi
        endif
        cosp = cos(phip)          ! cosine for one grid north
        cos0 = cos(phi0)          ! cosine for latitude grid
        cosm = cos(phim)          ! cosine for one grid south
        sin0 = sin(phi0)          ! sine for latitude grid
        ep1(i,j,k) = ep1(i,j,k)*cos0 ! correct for cosine factor

        ! meridional eddy momentum flux one grid north and south
        if (is_nhem) then
          ep2(i,j,k) = (uu(i,j+nd,k)-uref(j-jb+1,k))*cosp*cosp * vv(i,j+nd,k)
          ep3(i,j,k) = (uu(i,j+nd-2,k)-uref(j-jb-1,k))*cosm*cosm * vv(i,j+nd-2,k)
        else
          ! TODO: double check.
          ep2(i,j,k) = (uu(i,j+1,k)-uref(j+1,k))*cosp*cosp * vv(i,j+1,k)
          ep3(i,j,k) = (uu(i,j-1,k)-uref(j-1,k))*cosm*cosm * vv(i,j-1,k)
        endif

        ! low-level meridional eddy heat flux
        if(k.eq.2) then     ! (26) of SI-HN17
          ep41 = 2.*om*sin0*cos0*dz/prefac       ! prefactor
          if (is_nhem) then
            ep42 = exp(-dz/h)*vv(i,j+nd-1,2)*(pt(i,j+nd-1,2)-tref(j-jb,2))/(tg(3)-tg(1))
            ep43 = vv(i,j+nd-1,1)*(pt(i,j+nd-1,1)-tref(j-jb,1))
          else
            ep42 = exp(-dz/h)*vv(i,j,2)*(pt(i,j,2)-tref(j,2))/(tg(3)-tg(1))
            ep43 = vv(i,j,1)*(pt(i,j,1)-tref(j,1))
          endif
          ep43 = 0.5*ep43/(tg(2)-tg(1))
          ep4(i,j) = ep41*(ep42+ep43)   ! low-level heat flux
        endif
      enddo
      phip = dp*jb
      phi0 = dp*(jb-1)
      cosp = cos(phip)          ! cosine for one grid north
      cos0 = cos(phi0)          ! cosine for latitude grid
      ep2(i,jb,k) = (uu(i,nd+jb+1,k)-uref(2,k))*cosp*cosp*vv(i,nd+jb+1,k)
      ep3(i,jb,k) = (uu(i,nd+jb,k)-uref(1,k))*cos0*cos0*vv(i,nd+jb,k)
    enddo
  enddo

END SUBROUTINE compute_flux_dirinv_nshem
