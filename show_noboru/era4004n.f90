        program main

!     USE mkl95_BLAS, ONLY: GEMM,GEMV
      USE mkl95_LAPACK, ONLY: GETRF,GETRI


!   **** take QG analysis and compute Q_ref and invert for 
!   U_ref & Theta_ref for NH (Direct solver) ***

        integer,parameter :: imax = 360, JMAX = 181, KMAX = 97
        integer,parameter :: nd = 91,nnd=181
        integer,parameter :: jb = 5   ! lower bounding latitude
        integer,parameter :: jd = 86  ! nd - lower bounding latitude  
        common /array/ pv(imax,jmax,kmax),pv2(imax,jmax)
        common /brray/ uu(imax,jmax,kmax)
        common /bbray/ vort(imax,jmax,kmax),vort2(imax,jmax)
        common /bcray/ pt(imax,jmax,kmax)
        common /bdray/ stats(kmax),statn(kmax),ts0(kmax),tn0(kmax)
        common /crray/ qn(nnd),an(nnd),aan(nnd),tb(kmax),tg(kmax)
        common /drray/ cn(nnd),ccn(nnd),tref(jd,kmax)
        common /frray/ alat(nd),phi(nd),z(kmax),cbar(nd,kmax)
        common /krray/ tjk(jd-2,kmax-1),tj(jd-2),rj(jd-2)
        common /lrray/ qjj(jd-2,jd-2),cjj(jd-2,jd-2)
        common /orray/ xjj(jd-2,jd-2),yj(jd-2)
        common /mrray/ djj(jd-2,jd-2),sjj(jd-2,jd-2)
        common /nrray/ sjk(jd-2,jd-2,kmax-1)
        common /prray/ pjk(jd-2,kmax),pj(jd-2)
        common /irray/ qref(nd,kmax),u(jd,kmax),cref(nd,kmax)
        common /krray/ fawa(nd,kmax),ckref(nd,kmax)
        common /jrray/ qbar(nd,kmax),ubar(nd,kmax),tbar(nd,kmax)
        integer :: md(12),ipiv(jd-2)

        character*35 fn,fn0,fn1
        character*34 fu
        character*34 ft
        character*4  fn2(12),fy,fy1,fy2
        character*18 f1,f2
        character*19 f3
        character*36 fv
        character*38 fr

        a = 6378000.
        pi = acos(-1.)
        om = 7.29e-5
        dp = pi/180.
        dz = 500.
        h = 7000.
        r = 287.
        rkappa = r/1004.

        do nn = 1,nd
         phi(nn) = dp*float(nn-1)   
         alat(nn) = 2.*pi*a*a*(1.-sin(phi(nn)))
        enddo

        do k = 1,kmax
         z(k) = dz*float(k-1)
        enddo

        do m = 1979,2020

        md(1) = 31
        md(2) = 28
         if(mod(m,4).eq.0) md(2) = 29
        md(3) = 31
        md(4) = 30
        md(5) = 31
        md(6) = 30
        md(7) = 31
        md(8) = 31
        md(9) = 30
        md(10) = 31
        md(11) = 30
        md(12) = 31

        fn2(1) = '_01_'
        fn2(2) = '_02_'
        fn2(3) = '_03_'
        fn2(4) = '_04_'
        fn2(5) = '_05_'
        fn2(6) = '_06_'
        fn2(7) = '_07_'
        fn2(8) = '_08_'
        fn2(9) = '_09_'
        fn2(10) = '_10_'
        fn2(11) = '_11_'
        fn2(12) = '_12_'

        write(fy,266) m
 266    format(i4)

        do n = 1,12
         fn = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGPV'
         fu = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGU'
         ft = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGT'
         fr = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QGREF_N' 
         fv = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'QVORT' 
         write(6,*) fn,md(n)
        open(35,file =fn,  &
                form='unformatted',status = 'old')
        open(36,file =fu,  &
                form='unformatted',status = 'old')
        open(37,file =ft,  &
                form='unformatted',status = 'old')
        open(38,file =fr,  &
                form='unformatted',status = 'new')
        open(39,file =fv,  &
                form='unformatted',status = 'old')

          do mm = 1,md(n)*4

            read(35) pv 
            read(36) uu 
            read(39) vort 
            read(37) pt,tn0,ts0,statn,stats

! **** Zonal-mean field ****
            do j = 91,jmax 
              qbar(j-90,:) = 0.
              tbar(j-90,:) = 0.
              ubar(j-90,:) = 0.
            do i = 1,imax
              qbar(j-90,:) = qbar(j-90,:)+pv(i,j,:)/float(imax)
              tbar(j-90,:) = tbar(j-90,:)+pt(i,j,:)/float(imax)
              ubar(j-90,:) = ubar(j-90,:)+uu(i,j,:)/float(imax)
            enddo
            enddo
         
! **** hemispheric-mean potential temperature ****
            tb(:) = tn0(:)
            
          do k = 2,96
            pv2(:,:) = pv(:,:,k)    
            vort2(:,:) = vort(:,:,k)    

!  **** compute qref via area analysis ****
            qmax = maxval(pv2)
            qmin = minval(pv2)
            dq = (qmax-qmin)/float(nnd-1)
            qn(:) = 0.
            an(:) = 0.
            cn(:) = 0.
            do nn = 1,nnd
              qn(nn) = qmax - dq*float(nn-1)
            enddo            
            do j = 1,jmax
              phi0 = -0.5*pi+dp*float(j-1)
            do i = 1,imax
              ind = 1+int((qmax-pv2(i,j))/dq)
              da = a*a*dp*dp*cos(phi0)
              an(ind) = an(ind) + da
              cn(ind) = cn(ind) + da*pv2(i,j)
            enddo
            enddo
              aan(1) = 0.
              ccn(1) = 0.
            do nn = 2,nnd
              aan(nn) = aan(nn-1)+an(nn) 
              ccn(nn) = ccn(nn-1)+cn(nn) 
            enddo
             do j = 1,nd-1
             do nn = 1,nnd-1
              if(aan(nn).le.alat(j).and.aan(nn+1).gt.alat(j)) then
               dd = (alat(j)-aan(nn))/(aan(nn+1)-aan(nn))
               qref(j,k) = qn(nn)*(1.-dd)+qn(nn+1)*dd
               cref(j,k) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
              endif
             enddo 
             enddo

               qref(nd,k) = qmax

               cbar(nd,k) = 0.
             do j=nd-1,1,-1
               phi0 = dp*(float(j)-0.5)
               cbar(j,k) = cbar(j+1,k)+0.5*(qbar(j+1,k)+qbar(j,k)) &
                 *a*dp*2.*pi*a*cos(phi0)
             enddo 

! **** compute Kelvin's circulation based on absolute vorticity (for
! b.c.) ****

            qmax = maxval(vort2)
            qmin = minval(vort2)
            dq = (qmax-qmin)/float(nnd-1)
            qn(:) = 0.
            an(:) = 0.
            cn(:) = 0.
            do nn = 1,nnd
              qn(nn) = qmax - dq*float(nn-1)
            enddo            
            do j = 1,jmax
              phi0 = -0.5*pi+dp*float(j-1)
            do i = 1,imax
              ind = 1+int((qmax-vort2(i,j))/dq)
              da = a*a*dp*dp*cos(phi0)
              an(ind) = an(ind) + da
              cn(ind) = cn(ind) + da*vort2(i,j)
            enddo
            enddo
              aan(1) = 0.
              ccn(1) = 0.
            do nn = 2,nnd
              aan(nn) = aan(nn-1)+an(nn) 
              ccn(nn) = ccn(nn-1)+cn(nn) 
            enddo
             do j = 1,nd-1
             do nn = 1,nnd-1
              if(aan(nn).le.alat(j).and.aan(nn+1).gt.alat(j)) then
               dd = (alat(j)-aan(nn))/(aan(nn+1)-aan(nn))
               ckref(j,k) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
              endif
             enddo 
             enddo

          enddo

! ***** normalize QGPV by sine (latitude)  ****

            do j = 2,nd
              phi0 = dp*float(j-1)
              cor = sin(phi0)
              qref(j,:) = qref(j,:)/cor
            enddo

            do k = 2,kmax-1
              qref(1,k) = 2.*qref(2,k)-qref(3,k)
            enddo

! ***** FAWA *****
         fawa(:,:) = (cref(:,:)-cbar(:,:))/(2.*pi*a)

!  ***** Direct solver to invert Q_ref *****

!  *** downward sweep ***

! **** top boundary condition (Eqs. 24-25) *****
        tjk(:,:) = 0.
        sjk(:,:,:) = 0.
       do jj = jb+2,90
        j = jj-jb
       phi0 = float(jj-1)*dp 
       cos0 = cos(phi0)
       sin0 = sin(phi0)
       tjk(j-1,kmax-1) = -dz*r*cos0*exp(-z(kmax-1)*rkappa/h) 
       tjk(j-1,kmax-1) = tjk(j-1,kmax-1)*(tbar(j+1,kmax)-tbar(j-1,kmax))
       tjk(j-1,kmax-1) = tjk(j-1,kmax-1)/(4.*om*sin0*dp*h*a) 
       sjk(j-1,j-1,kmax-1) = 1.
       enddo

! **** Evaluate Eqs. 22-23 downward ***

        do k = kmax-1,2,-1
          zp = 0.5*(z(k+1)+z(k))
          zm = 0.5*(z(k-1)+z(k))
          statp = 0.5*(statn(k+1)+statn(k))
          statm = 0.5*(statn(k-1)+statn(k))
          cjj(:,:) = 0.
          djj(:,:) = 0.
          qjj(:,:) = 0.
          sjj(:,:) = sjk(:,:,k)
          tj(:) = tjk(:,k)
        do jj = jb+2,90
         j = jj - jb
         phi0 = float(jj-1)*dp 
         phip = (float(jj)-0.5)*dp
         phim = (float(jj)-1.5)*dp
         cos0 = cos(phi0)
         cosp = cos(phip)
         cosm = cos(phim)
         sin0 = sin(phi0)
         sinp = sin(phip)
         sinm = sin(phim)
         
          fact = 4.*om*om*h*a*a*sin0*dp*dp/(dz*dz*r*cos0)
          amp = exp(-zp/h)*exp(rkappa*zp/h)/statp
          amp = amp*fact*exp(z(k)/h)
          amm = exp(-zm/h)*exp(rkappa*zm/h)/statm
          amm = amm*fact*exp(z(k)/h)

! ***** Specify A, B, C, D, E, F (Eqs. 4-9) *****
          ajk = 1./(sinp*cosp)
          bjk = 1./(sinm*cosm)
          cjk = amp
          djk = amm
          ejk = ajk+bjk+cjk+djk
          fjk = -0.5*a*dp*(qref(jj+1,k)-qref(jj-1,k))

! ***** Specify rk (Eq. 15) ****

      ! **** North-south boundary conditions ****
         u(jd,k) = 0.
         phi0 = dp*float(jb)
!        u(1,k) = ubar(jb+1,k)*cos(phi0)   
         u(1,k) = ckref(jb+1,k)/(2.*pi*a)-om*a*cos(phi0)

          rj(j-1) = fjk
         if(j.eq.2) rj(j-1) = fjk - bjk*u(1,k)
         if(j.eq.jd-1) rj(j-1) = fjk - ajk*u(jd,k)

! ***** Specify Ck & Dk (Eqs. 18-19) *****
          cjj(j-1,j-1) = cjk
          djj(j-1,j-1) = djk

! **** Specify Qk (Eq. 17) *****
          qjj(j-1,j-1) = -ejk
        if(j-1.ge.1.and.j-1.lt.jd-2) then
          qjj(j-1,j) = ajk
        endif
        if(j-1.gt.1.and.j-1.le.jd-2) then
          qjj(j-1,j-2) = bjk
        endif
        enddo
      
! **** Compute Qk + Ck Sk *******
        do i = 1,jd-2
        do j = 1,jd-2
          xjj(i,j) = 0.
        do kk = 1,jd-2
          xjj(i,j) = xjj(i,j)+cjj(i,kk)*sjj(kk,j)
        enddo
          qjj(i,j) = qjj(i,j)+xjj(i,j)
        enddo
        enddo
!         call gemm(cjj,sjj,xjj)
!         qjj(:,:) = qjj(:,:)+xjj(:,:)

! **** Invert (Qk + Ck Sk) ********
          call getrf(qjj,ipiv)
          call getri(qjj,ipiv)

! **** Evaluate Eq. 22 ****
        do i = 1,jd-2
        do j = 1,jd-2
          xjj(i,j) = 0.
        do kk = 1,jd-2
          xjj(i,j) = xjj(i,j)+qjj(i,kk)*djj(kk,j)
        enddo
          sjk(i,j,k-1) = -xjj(i,j)
        enddo
        enddo

!         call gemm(qjj,djj,xjj)
!         sjk(:,:,k-1) = -xjj(:,:)

!  **** Evaluate rk - Ck Tk ****
        do i = 1,jd-2
          yj(i) = 0.
        do kk = 1,jd-2
          yj(i) = yj(i)+cjj(i,kk)*tj(kk)
        enddo
          yj(i) =  rj(i)-yj(i)
        enddo

!         call gemv(cjj,tj,yj) 
!         yj(:) = rj(:)-yj(:)
!         call gemv(qjj,yj,tj)
!         tjk(:,k-1) = tj(:)


! ***** Evaluate Eq. 23 *******
        do i = 1,jd-2
          tj(i) = 0.
        do kk = 1,jd-2
          tj(i) = tj(i)+qjj(i,kk)*yj(kk)
        enddo
          tjk(i,k-1) = tj(i)
        enddo
  
        enddo

! ***** upward sweep (Eq. 20) ****

         pjk(:,1) = 0.
       do k = 1,kmax-1
         pj(:) = pjk(:,k)
         sjj(:,:) = sjk(:,:,k)
         tj(:) = tjk(:,k)

        do i = 1,jd-2
          yj(i) = 0.
        do kk = 1,jd-2
          yj(i) = yj(i)+sjj(i,kk)*pj(kk)
        enddo
          pjk(i,k+1) = yj(i)+tj(i)
        enddo
!        call gemv(sjj,pj,yj)
!        pjk(:,k+1) = yj(:) + tj(:)
       enddo 
      
! **** Recover u *****
       do k = 1,kmax
        do j = 2,jd-1
          u(j,k) = pjk(j-1,k)
        enddo
       enddo

! *** Corner boundary conditions ***
         u(1,1) = 0.
         u(jd,1) = 0.
!        u(1,kmax) = ubar(1+jb,kmax)*cos(dp*float(jb))
         u(1,kmax) = ckref(1+jb,kmax)/(2.*pi*a)-om*a*cos(dp*float(jb))
         u(jd,kmax) = 0.

! *** Divide by cos phi to revover Uref ****
       do jj = jb+1,nd-1
        j = jj-jb
        phi0 = dp*float(jj-1)
        u(j,:) = u(j,:)/cos(phi0)
       enddo
        u(jd,:) = 2.*u(jd-1,:)-u(jd-2,:)

! ******** compute tref *******

        do k = 2,96
         t00 = 0.
         zz = dz*float(k-1)
           tref(1,k) = t00
           tref(2,k) = t00
         do j = 2,jd-1
           phi0 = dp*float(j-1+jb)
           cor = 2.*om*sin(phi0)  
           uz = (u(j,k+1)-u(j,k-1))/(2.*dz)
           ty = -cor*uz*a*h*exp(rkappa*zz/h)
           ty = ty/r
           tref(j+1,k) = tref(j-1,k)+2.*ty*dp
         enddo
         do j = 1,nd
           phi0 = dp*float(j-1)
           qref(j,k) = qref(j,k)*sin(phi0)
         enddo
            tg(k) = 0.
            wt = 0.
            do jj = 6,91
             j = jj-5
             phi0 = dp*float(jj-1)
             tg(k) = tg(k)+cos(phi0)*tref(j,k)
             wt = wt + cos(phi0)
            enddo
             tg(k) = tg(k)/wt
            tres = tb(k)-tg(k)
            tref(:,k) = tref(:,k)+tres
        enddo
            tref(:,1) = tref(:,2)-tb(2)+tb(1)
            tref(:,97) = tref(:,96)-tb(96)+tb(97)
            
        write(38) qref,u,tref,fawa,ubar,tbar

        write(6,*) m,n,mm 

! ********************************
          enddo

        close(35)
        close(36)
        close(37)
        close(38)

        enddo
        enddo

        stop
        end
