SUBROUTINE compute_flux_dirinv_nshem( &
    pv, uu, vv, pt, ncforce, tn0, qref, uref, tref, &
    imax, JMAX, kmax, nd, jb, jd, is_nhem, &
    a, om, dz, h, rr, cp, prefac, &
    astar1, astar2, ncforce3d, ua1, ua2, ep1, ep2, ep3, ep4)

  REAL, INTENT(IN) :: pv(kmax,jmax,imax),uu(kmax,jmax,imax),vv(kmax,jmax,imax),pt(kmax,jmax,imax), &
          ncforce(kmax,jmax,imax), &
          tn0(kmax),qref(kmax,nd),uref(kmax,jd),tref(kmax,jd)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, jb, jd
  LOGICAL, INTENT(IN) :: is_nhem
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astar1(kmax,nd,imax), astar2(kmax,nd,imax), ncforce3d(kmax,nd,imax), &
          ua1(kmax,nd,imax),  ua2(kmax,nd,imax), ep1(kmax,nd,imax), ep2(kmax,nd,imax), ep3(kmax,nd,imax), ep4(nd,imax)

  REAL :: tg(kmax)
  REAL :: qe(nd,imax),ue(nd,imax)
  REAL :: ncforce2d(nd,imax)
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
        astar1(k,j,i) = 0.       ! LWA*cos(phi)
        astar2(k,j,i) = 0.       ! LWA*cos(phi)
        ncforce3d(k,j,i) = 0.
        ua2(k,j,i) = 0.          !F2
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
            qe(jj,i) = pv(k,jj+nd-1,i)-qref(k,j)    !qe; Q = qref
            ncforce2d(jj,i) = ncforce(k,jj+nd-1,i)
            ue(jj,i) = uu(k,jj+nd-1,i)*cos(phi0)-uref(k,j-jb)*cos(phi1) !ue; shift uref 5N
          else  ! Southern Hemisphere
            phi1 = dp*float(jj-1)-0.5*pi
            qe(jj,i) = pv(k,jj,i)-qref(k,j)     !qe; Q = qref
            ncforce2d(jj,i) = ncforce(k,jj,i)
            ue(jj,i) = uu(k,jj,i)*cos(phi0)-uref(k,j)*cos(phi1)  !ue;
          endif
          aa = a*dp*cos(phi1)   !cosine factor in the meridional integral
          if((qe(jj,i).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            if (is_nhem) then  ! Northern Hemisphere
              astar2(k,j,i)=astar2(k,j,i)-qe(jj,i)*aa  !anticyclonic
            else  ! Southern Hemisphere
              astar1(k,j,i)=astar1(k,j,i)-qe(jj,i)*aa  !cyclonic
            endif
            ncforce3d(k,j,i)=ncforce3d(k,j,i)-ncforce2d(jj,i)*aa
            ua2(k,j,i) = ua2(k,j,i)-qe(jj,i)*ue(jj,i)*ab
          endif
          if((qe(jj,i).gt.0.).and.(jj.lt.j)) then
            if (is_nhem) then  ! Northern Hemisphere
              astar1(k,j,i)=astar1(k,j,i)+qe(jj,i)*aa  !cyclonic
            else  ! Southern Hemisphere
              astar2(k,j,i)=astar2(k,j,i)+qe(jj,i)*aa  !anticyclonic
            endif
            ncforce3d(k,j,i)=ncforce3d(k,j,i)+ncforce2d(jj,i)*aa
            ua2(k,j,i) = ua2(k,j,i)+qe(jj,i)*ue(jj,i)*ab
          endif
        enddo

        !  ******** Other fluxes ******
        if (is_nhem) then
          ua1(k,j,i) = uref(k,j-jb)*(astar1(k,j,i) + astar2(k,j,i))  !F1
          ep1(k,j,i) = -0.5*(uu(k,j+nd-1,i)-uref(k,j-jb))**2         !F3a
          ep1(k,j,i) = ep1(k,j,i)+0.5*vv(k,j+nd-1,i)**2                !F3a+b
          ep11 = 0.5*(pt(k,j+nd-1,i)-tref(k,j-jb))**2              !F3c
        else
          ua1(k,j,i) = uref(k,j)*(astar1(k,j,i) + astar2(k,j,i))     !F1
          ep1(k,j,i) = -0.5*(uu(k,j,i)-uref(k,j))**2                 !F3a
          ep1(k,j,i) = ep1(k,j,i)+0.5*vv(k,j,i)**2                     !F3a+b
          ep11 = 0.5*(pt(k,j,i)-tref(k,j))**2               !F3c
        endif
        zz = dz*float(k-1)
        ep11 = ep11*(rr/h)*exp(-rkappa*zz/h)
        ep11 = ep11*2.*dz/(tg(k+1)-tg(k-1))
        ep1(k,j,i) = ep1(k,j,i)-ep11                   !F3
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
        ep1(k,j,i) = ep1(k,j,i)*cos0 ! correct for cosine factor

        ! meridional eddy momentum flux one grid north and south
        if (is_nhem) then
          ep2(k,j,i) = (uu(k,j+nd,i)-uref(k,j-jb+1))*cosp*cosp * vv(k,j+nd,i)
          ep3(k,j,i) = (uu(k,j+nd-2,i)-uref(k,j-jb-1))*cosm*cosm * vv(k,j+nd-2,i)
        else
          ! TODO: double check.
          ep2(k,j,i) = (uu(k,j+1,i)-uref(k,j+1))*cosp*cosp * vv(k,j+1,i)
          ep3(k,j,i) = (uu(k,j-1,i)-uref(k,j-1))*cosm*cosm * vv(k,j-1,i)
        endif

        ! low-level meridional eddy heat flux
        if(k.eq.2) then     ! (26) of SI-HN17
          ep41 = 2.*om*sin0*cos0*dz/prefac       ! prefactor
          if (is_nhem) then
            ep42 = exp(-dz/h)*vv(2,j+nd-1,i)*(pt(2,j+nd-1,i)-tref(2,j-jb))/(tg(3)-tg(1))
            ep43 = vv(1,j+nd-1,i)*(pt(1,j+nd-1,i)-tref(1,j-jb))
          else
            ep42 = exp(-dz/h)*vv(2,j,i)*(pt(2,j,i)-tref(2,j))/(tg(3)-tg(1))
            ep43 = vv(1,j,i)*(pt(1,j,i)-tref(1,j))
          endif
          ep43 = 0.5*ep43/(tg(2)-tg(1))
          ep4(j,i) = ep41*(ep42+ep43)   ! low-level heat flux
        endif
      enddo
      phip = dp*jb
      phi0 = dp*(jb-1)
      cosp = cos(phip)          ! cosine for one grid north
      cos0 = cos(phi0)          ! cosine for latitude grid
      ep2(k,jb,i) = (uu(k,nd+jb+1,i)-uref(k,2))*cosp*cosp*vv(k,nd+jb+1,i)
      ep3(k,jb,i) = (uu(k,nd+jb,i)-uref(k,1))*cos0*cos0*vv(k,nd+jb,i)
    enddo
  enddo

END SUBROUTINE compute_flux_dirinv_nshem
