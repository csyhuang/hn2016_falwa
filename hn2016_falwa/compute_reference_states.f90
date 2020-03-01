SUBROUTINE compute_reference_states(pv,uu,pt,stat,nlon,nlat,kmax,jd,npart,maxits,&
               a,om,dz,eps,h,r,cp,rjac,qref,u,tref,num_of_iter)


    integer, intent(in) :: nlon,nlat,kmax,jd,npart,maxits
    real, intent(in) :: pv(nlon,nlat,kmax),uu(nlon,nlat,kmax),pt(nlon,nlat,kmax),stat(kmax)
    real, intent(in) :: a, om, dz,eps,h, r, cp, rjac
    real, intent(out) :: qref(jd,kmax),u(jd,kmax),tref(jd,kmax)
    integer, intent(out) :: num_of_iter


    real :: pv2(nlon,nlat)
    real :: u2(nlon,nlat)
    real :: pt2(nlon,nlat)
    real :: qn(npart),an(npart),aan(npart),tb(kmax),tg(kmax)
    real :: cn(npart),ccn(npart)
    real :: alat(jd),phi(jd),z(kmax),cbar(jd,kmax)
    real :: ajk(jd,kmax),bjk(jd,kmax),cjk(jd,kmax)
    real :: djk(jd,kmax),ejk(jd,kmax),fjk(jd,kmax)

    real :: cref(jd,kmax)
    real :: qbar(jd,kmax),ubar(jd,kmax),tbar(jd,kmax)
    real :: pi, dp, zero, half, qtr, one, rkappa

    pi = acos(-1.)
    dp = pi/float(nlat-1)
    zero = 0.
    half = 0.5
    qtr = 0.25
    one = 1.
    rkappa = r/cp        


    do nn = 1,jd
        phi(nn) = dp*float(nn-1)
        alat(nn) = 2.*pi*a*a*(1.-sin(phi(nn)))
    enddo

    do k = 1,kmax
        z(k) = dz*float(k-1)
    enddo


    ! **** Zonal-mean field ****
    do j = jd,nlat 
        qbar(j-(jd-1),:) = 0.
        tbar(j-(jd-1),:) = 0.
        ubar(j-(jd-1),:) = 0.
        do i = 1,nlon
            qbar(j-(jd-1),:) = qbar(j-(jd-1),:)+pv(i,j,:)/float(nlon)
            tbar(j-(jd-1),:) = tbar(j-(jd-1),:)+pt(i,j,:)/float(nlon)
            ubar(j-(jd-1),:) = ubar(j-(jd-1),:)+uu(i,j,:)/float(nlon)
        enddo
    enddo

    ! **** hemispheric-mean potential temperature ****
    tb(:) = 0.
    wt = 0.
    do j = jd,nlat
        phi0 = dp*float(j-1)-0.5*pi
        !tb(:) = tb(:)+cos(phi0)*tbar(j,:)
        tb(:) = tb(:)+cos(phi0)*tbar(j-(jd-1),:)
        wt = wt + cos(phi0)
    enddo
    tb(:) = tb(:)/wt

    do k = 2,KMAX-1
        pv2(:,:) = pv(:,:,k)    
        u2(:,:) = uu(:,:,k)    
        pt2(:,:) = pt(:,:,k)    

        !  **** area analysis ****
        qmax = maxval(pv2)
        qmin = minval(pv2)
        dq = (qmax-qmin)/float(npart-1)
        qn(:) = 0.
        an(:) = 0.
        cn(:) = 0.
        do nn = 1,npart
            qn(nn) = qmax - dq*float(nn-1)
        enddo            
        do j = 1,nlat
            phi0 = -0.5*pi+dp*float(j-1)
            do i = 1,nlon
                ind = 1+int((qmax-pv2(i,j))/dq)
                da = a*a*dp*dp*cos(phi0)
                an(ind) = an(ind) + da
                cn(ind) = cn(ind) + da*pv2(i,j)
            enddo
        enddo
        aan(1) = 0.
        ccn(1) = 0.
        do nn = 2,npart
            aan(nn) = aan(nn-1)+an(nn) 
            ccn(nn) = ccn(nn-1)+cn(nn) 
        enddo
        do j = 1,(jd-1)
            do nn = 1,npart-1
                if(aan(nn).le.alat(j).and.aan(nn+1).gt.alat(j)) then
                    dd = (alat(j)-aan(nn))/(aan(nn+1)-aan(nn))
                    qref(j,k) = qn(nn)*(1.-dd)+qn(nn+1)*dd
                    cref(j,k) = ccn(nn)*(1.-dd)+ccn(nn+1)*dd
                endif
            enddo 
        enddo

        qref(jd,k) = qmax

        cbar(jd,k) = 0.
        do j=(jd-1),1,-1
            phi0 = dp*(float(j)-0.5)
            cbar(j,k) = cbar(j+1,k)+0.5*(qbar(j+1,k)+qbar(j,k)) &
            *a*dp*2.*pi*a*cos(phi0)
        enddo 

    enddo


    ! ***** normalize QGPV by the Coriolis parameter ****

    do j = 2,jd
        phi0 = dp*float(j-1)
        cor = 2.*om*sin(phi0)
        qref(j,:) = qref(j,:)/cor
    enddo

    do k = 2,KMAX-1
        qref(1,k) = 2.*qref(2,k)-qref(3,k)
    enddo


    ! **** SOR (elliptic solver a la Numerical Recipes) ****

    do j = 2,(jd-1)
        phi0 = float(j-1)*dp 
        phip = (float(j)-0.5)*dp
        phim = (float(j)-1.5)*dp
        cos0 = cos(phi0)
        cosp = cos(phip)
        cosm = cos(phim)
        sin0 = sin(phi0)
        sinp = sin(phip)
        sinm = sin(phim)
        do k = 2,KMAX-1
            zp = 0.5*(z(k+1)+z(k))
            zm = 0.5*(z(k-1)+z(k))
            statp = 0.5*(stat(k+1)+stat(k))
            statm = 0.5*(stat(k-1)+stat(k))
            fact = 4.*om*om*h*a*a*sin0*dp*dp/(dz*dz*r*cos0)
            amp = exp(-zp/h)*exp(rkappa*zp/h)/statp
            amp = amp*fact*exp(z(k)/h)
            amm = exp(-zm/h)*exp(rkappa*zm/h)/statm
            amm = amm*fact*exp(z(k)/h)
            ajk(j,k) = 1./(sinp*cosp)
            bjk(j,k) = 1./(sinm*cosm)
            cjk(j,k) = amp
            djk(j,k) = amm
            ejk(j,k) = -ajk(j,k)-bjk(j,k)-cjk(j,k)-djk(j,k)
            fjk(j,k) = -om*a*dp*(qref(j+1,k)-qref(j-1,k))
        enddo
    enddo

    anormf = zero

    do j = 2,(jd-1)
        do k = 2,KMAX-1
            anormf = anormf + abs(fjk(j,k)) 
        enddo
    enddo

    omega = one

    do nnn = 1,maxits
        anorm = zero
        do j = 2,(jd-1)
            do k = 2,KMAX-1
                if(mod(j+k,2).eq.mod(nnn,2)) then
                    resid = ajk(j,k)*u(j+1,k)+bjk(j,k)*u(j-1,k)+    &
                     cjk(j,k)*u(j,k+1)+djk(j,k)*u(j,k-1)+    &
                     ejk(j,k)*u(j,k)-fjk(j,k)
                    anorm = anorm + abs(resid)
                    if(ejk(j,k).ne.0.) u(j,k) = u(j,k) - omega*resid/ejk(j,k)
                    if(ejk(j,k).eq.0.) u(j,k) = 0.
                endif
            enddo
            u(j,1) = 0.
            phi0 = dp*float(j-1)
            uz = dz*r*cos(phi0)*exp(-z(KMAX-1)*rkappa/h)
            uz = uz*(tbar(j+1,KMAX-1)-tbar(j-1,KMAX-1))/(2.*om*sin(phi0)*dp*h*a)
            u(j,KMAX) = u(j,KMAX-2)-uz
        enddo

        u(jd,:) = 0.
        u(1,:) = ubar(1,:)+(cref(1,:)-cbar(1,:))/(2.*pi*a)
        !        u(1,:) = ubar(1,:)

        if(nnn.eq.1) then
            omega = one/(one-half*rjac**2)
        else
            omega = one/(one-qtr*rjac**2*omega)
        endif

        if((nnn.gt.1).and.(anorm.lt.eps*anormf)) then
            if(nheck.eq.1) nheck = 0
            goto 233
        endif

    enddo

    ! **** In the case of non-convergence ****

    write(6,*) 'Maxits exceeded'

    u(:,:) = 0.      ! no correction for the non-converged case

    233  continue

    ! **** Converged! (u is the mass correction) ****

    write(6,*)  n,mm,kk,' converged at n = ',nnn
    num_of_iter = nnn

    do j = 2,(jd-1)
        phi0 = dp*float(j-1)
        u(j,:) = u(j,:)/cos(phi0)
    enddo
    u(1,:) = ubar(1,:)
    u(jd,:) = 2.*u((jd-1),:)-u((jd-2),:)

    ! ******** compute tref *******

    do k = 2,KMAX-1
        t00 = 0.
        zz = dz*float(k-1)
        tref(1,k) = t00
        tref(2,k) = t00
        do j = 2,(jd-1)
            phi0 = dp*float(j-1)
            cor = 2.*om*sin(phi0)  
            uz = (u(j,k+1)-u(j,k-1))/(2.*dz)
            ty = -cor*uz*a*h*exp(rkappa*zz/h)
            ty = ty/r
            tref(j+1,k) = tref(j-1,k)+2.*ty*dp
        enddo
        tg(k) = 0.
        wt = 0.
        do j = 1,jd
            phi0 = dp*float(j-1)
            tg(k) = tg(k)+cos(phi0)*tref(j,k)
            wt = wt + cos(phi0)
        enddo
        tg(k) = tg(k)/wt
        tres = tb(k)-tg(k)
        tref(:,k) = tref(:,k)+tres
    enddo
    tref(:,1) = tref(:,2)-tb(2)+tb(1)
    tref(:,KMAX) = tref(:,KMAX-1)-tb(KMAX-1)+tb(KMAX)


end subroutine
