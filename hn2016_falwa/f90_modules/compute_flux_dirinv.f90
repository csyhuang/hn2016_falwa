SUBROUTINE compute_flux_dirinv_nshem(pv,uu,vv,pt,tn0,qref,uref,tref,&
        imax, JMAX, kmax, nd, jb, jd, is_nhem, &
        a, om, dz, h, rr, cp, prefac,&
        astarbaro,ubaro,urefbaro,ua1baro,ua2baro,ep1baro,ep2baro,ep3baro,ep4,astar1,astar2)

  REAL, INTENT(IN) :: pv(imax,jmax,kmax),uu(imax,jmax,kmax),vv(imax,jmax,kmax),pt(imax,jmax,kmax),&
          tn0(kmax),qref(nd,kmax),uref(jd,kmax),tref(jd,kmax)
  INTEGER, INTENT(IN) :: imax, JMAX, kmax, nd, jb, jd
  LOGICAL, INTENT(IN) :: is_nhem
  REAL, INTENT(IN) :: a, om, dz, h, rr, cp, prefac
  REAL, INTENT(OUT) :: astarbaro(imax,nd),ubaro(imax,nd),urefbaro(nd),ua1baro(imax,nd),ua2baro(imax,nd),&
          ep1baro(imax,nd),ep2baro(imax,nd),ep3baro(imax,nd),ep4(imax,nd),astar1(imax,nd,kmax),astar2(imax,nd,kmax)

  REAL :: tg(kmax),ua1(imax,nd),ua2(imax,nd),ep1(imax,nd),ep2(imax,nd),ep3(imax,nd)
  REAL :: qe(imax,nd),ue(imax,nd)
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
        ua2(i,j) = 0.          !F2
        if (is_nhem) then        !latitude
          phi0 = dp*float(j-1)
        else
          phi0 = dp*float(j-1)-0.5*pi
        endif
        cor = 2.*om*sin(phi0)          !Coriolis parameter
        ab = a*dp*cos(phi0)
        do jj = 1,nd
          if (is_nhem) then
            phi1 = dp*float(jj-1)
            qe(i,jj) = pv(i,jj+nd-1,k)-qref(j,k)    !qe; Q = qref
            ue(i,jj) = uu(i,jj+nd-1,k)-uref(j-jb,k) !ue; shift uref 5N
          else
            phi1 = dp*float(jj-1)-0.5*pi
            qe(i,jj) = pv(i,jj,k)-qref(j,k)     !qe; Q = qref
            ue(i,jj) = uu(i,jj,k)-uref(j-jb,k)  !ue;
          endif
          aa = a*dp*cos(phi1)   !cosine factor in the meridional integral
          if((qe(i,jj).le.0.).and.(jj.ge.j)) then  !LWA*cos and F2
            if (is_nhem) then
              astar2(i,j,k)=astar2(i,j,k)-qe(i,jj)*aa  !anticyclonic
            else
              astar1(i,j,k)=astar1(i,j,k)-qe(i,jj)*aa  !cyclonic
            endif
            ua2(i,j) = ua2(i,j)-qe(i,jj)*ue(i,jj)*ab
          endif
          if((qe(i,jj).gt.0.).and.(jj.lt.j)) then
            if (is_nhem) then
              astar1(i,j,k)=astar1(i,j,k)+qe(i,jj)*aa  !cyclonic
            else
              astar2(i,j,k)=astar2(i,j,k)+qe(i,jj)*aa  !anticyclonic
            endif
            ua2(i,j) = ua2(i,j)+qe(i,jj)*ue(i,jj)*ab
          endif
        enddo

        !  ******** Other fluxes ******
        if (is_nhem) then
          ua1(i,j) = uref(j-jb,k)*(astar1(i,j,k) + astar2(i,j,k))  !F1
          ep1(i,j) = -0.5*(uu(i,j+nd-1,k)-uref(j-jb,k))**2         !F3a
          ep1(i,j) = ep1(i,j)+0.5*vv(i,j+nd-1,k)**2                !F3a+b
          ep11 = 0.5*(pt(i,j+nd-1,k)-tref(j-jb,k))**2              !F3c
        else
          ua1(i,j) = uref(j,k)*(astar1(i,j,k) + astar2(i,j,k))     !F1
          ep1(i,j) = -0.5*(uu(i,j,k)-uref(j,k))**2                 !F3a
          ep1(i,j) = ep1(i,j)+0.5*vv(i,j,k)**2                     !F3a+b
          ep11 = 0.5*(pt(i,j,k)-tref(j,k))**2               !F3c
        endif
        zz = dz*float(k-1)
        ep11 = ep11*(rr/h)*exp(-rkappa*zz/h)
        ep11 = ep11*2.*dz/(tg(k+1)-tg(k-1))
        ep1(i,j) = ep1(i,j)-ep11                   !F3
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
        ep1(i,j) = ep1(i,j)*cos0 ! correct for cosine factor


        ! meridional eddy momentum flux one grid north and south
        if (is_nhem) then
          ep2(i,j) = (uu(i,j+nd,k)-uref(j-jb+1,k))*cosp*cosp * vv(i,j+nd,k)
          ep3(i,j) = (uu(i,j+nd-2,k)-uref(j-jb-1,k))*cosm*cosm * vv(i,j+nd-2,k)
        else
          ! TODO: double check.
          ep2(i,j) = (uu(i,j+1,k)-uref(j+1,k))*cosp*cosp * vv(i,j+1,k)
          ep3(i,j) = (uu(i,j-1,k)-uref(j-1,k))*cosm*cosm * vv(i,j-1,k)
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
      ep2(i,jb) = (uu(i,nd+jb+1,k)-uref(2,k))*cosp*cosp*vv(i,nd+jb+1,k)
      ep3(i,jb) = (uu(i,nd+jb,k)-uref(1,k))*cos0*cos0*vv(i,nd+jb,k)
    enddo

    ! ******** Column average: (25) of SI-HN17 ********

    astarbaro(:,:) = astarbaro(:,:)+(astar1(:,:,k)    &
    + astar2(:,:,k))*exp(-zk/h)*dc
    ua1baro(:,:) = ua1baro(:,:)+ua1(:,:)*exp(-zk/h)*dc
    ua2baro(:,:) = ua2baro(:,:)+ua2(:,:)*exp(-zk/h)*dc
    ep1baro(:,:) = ep1baro(:,:)+ep1(:,:)*exp(-zk/h)*dc
    ep2baro(:,:) = ep2baro(:,:)+ep2(:,:)*exp(-zk/h)*dc
    ep3baro(:,:) = ep3baro(:,:)+ep3(:,:)*exp(-zk/h)*dc

    ! Bounds of y-indices in N/SHem
    if (is_nhem) then  ! 5N and higher latitude
      jstart = jb+1  ! 6
      jend = nd
    else
      jstart = 1
      jend = nd-jb   ! nd - 5
    endif

    do j = jstart,jend  ! ### yet to be multiplied by cosine
      ubaro(:,j) = ubaro(:,j)+uu(:,j+nd-1,k)*exp(-zk/h)*dc
      urefbaro(j) = urefbaro(j)+uref(j-jb,k)*exp(-zk/h)*dc
    enddo
  enddo

END SUBROUTINE
