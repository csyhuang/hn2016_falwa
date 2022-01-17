        program main

        use NETCDF

!   **** convert barotropic LWA and fluxes for 
!   NH into netCDF files ***

        integer,parameter :: imax = 360
        integer,parameter :: nd = 91
        common /frray/ ep4(imax,nd)
        common /grray/ ubaro(imax,nd),urefbaro(nd),astarbaro(imax,nd)
        common /hrray/ ua1baro(imax,nd),ua2baro(imax,nd)
        common /hrray/ ep1baro(imax,nd),ep2baro(imax,nd)
        common /hrray/ ep3baro(imax,nd),wa2(nd,124)
        common /array/ astar(imax,nd,124),ub(imax,nd,124)
        common /brray/ urb(nd,124),ua1(imax,nd,124)
        common /crray/ ua2(imax,nd,124),ep1(imax,nd,124)
        common /drray/ ep2(imax,nd,124),ep3(imax,nd,124)
        common /erray/ ep44(imax,nd,124),wa1(imax,nd,124)
        integer :: md(12)

        character*35 fn,fn0,fn1
        character*34 fu
        character*34 ft,fv
        character*38 fx
        character*4  fn2(12),fy,fy1,fy2
        character*19 f3
        character*36 fr
        character*37 fm
        character*39 fd,fe,ff,fg,fh,fi
        character*40 fa
        character*41 fc
        character*38 fb

        integer :: ncid, status,nDim,nVar,nAtt,uDimID,inq
        integer :: lonID,latID,vid2,varID
        integer :: l1,l2,l3,l4,l5,l6,xtype,len,attnum

        a = 6378000.
        pi = acos(-1.)
        om = 7.29e-5
        dp = pi/180.
        dz = 500.
        h = 7000.
        r = 287.
        rkappa = r/1004.

        do m = 2021,2021

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

        do n = 6,6
         fm = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'BARO_N' 
         fa = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'LWAb_N.nc'
         fb = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'Ub_N.nc'
         fc = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'Urefb_N.nc'
         fd = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ua1_N.nc'
         fe = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ua2_N.nc'
         ff = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ep1_N.nc'
         fg = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ep2_N.nc'
         fh = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ep3_N.nc'
         fi = '/data2/nnn/ERA5/'//fy//'/'//fy//fn2(n)//'ep4_N.nc'
         write(6,*) fn,md(n)
        open(41,file =fm,  &
                form='unformatted',status = 'old')

          do mm = 1,md(n)*4

       read(41) astarbaro,ubaro,urefbaro,ua1baro,ua2baro,ep1baro,& 
                 ep2baro,&
                 ep3baro,ep4

               astar(:,:,mm) = astarbaro(:,:)
               ub(:,:,mm) = ubaro(:,:)
               urb(:,mm) = urefbaro(:)
               ua1(:,:,mm) = ua1baro(:,:)
               ua2(:,:,mm) = ua2baro(:,:)
               ep1(:,:,mm) = ep1baro(:,:)
               ep2(:,:,mm) = ep2baro(:,:)
               ep3(:,:,mm) = ep3baro(:,:)
               ep44(:,:,mm) = ep4(:,:)
 
! ********************************

     write(6,*)  fy,n,mm

! ********************************
          enddo
       
       status = nf90_create(fa,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"lwa",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fa)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,astar)
       status = nf90_close(ncid2)

       status = nf90_open(fa,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"lwa",varID)
       write(6,*) 'Variable ID for LWA = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) astar(200,47,30),wa1(200,47,30)
                                                              
       status = nf90_create(fb,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"u",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fb)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ub)
       status = nf90_close(ncid2)

       status = nf90_open(fb,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"u",varID)
       write(6,*) 'Variable ID for U = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ub(200,47,30),wa1(200,47,30)
                                                             
       status = nf90_create(fd,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ua1",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fd)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ua1)
       status = nf90_close(ncid2)

       status = nf90_open(fd,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ua1",varID)
       write(6,*) 'Variable ID for ua1 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ua1(200,47,30),wa1(200,47,30)
                                                             
       status = nf90_create(fe,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ua2",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fe)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ua2)
       status = nf90_close(ncid2)

       status = nf90_open(fe,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ua2",varID)
       write(6,*) 'Variable ID for ua2 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ua2(200,47,30),wa1(200,47,30)
                                                             
       status = nf90_create(ff,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ep1",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",ff)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ep1)
       status = nf90_close(ncid2)

       status = nf90_open(ff,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ep1",varID)
       write(6,*) 'Variable ID for ep1 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ep1(200,47,30),wa1(200,47,30)
                                                              
       status = nf90_create(fg,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ep2",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fg)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ep2)
       status = nf90_close(ncid2)

       status = nf90_open(fg,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ep2",varID)
       write(6,*) 'Variable ID for ep2 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ep2(200,47,30),wa1(200,47,30)
                                                              
       status = nf90_create(fh,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ep3",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fh)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ep3)
       status = nf90_close(ncid2)

       status = nf90_open(fh,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ep3",varID)
       write(6,*) 'Variable ID for ep3 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ep3(200,47,30),wa1(200,47,30)
                                                              
       status = nf90_create(fi,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"longitude",imax,ix)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"ep4",nf90_float,   &
                (/ix,iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fi)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,ep44)
       status = nf90_close(ncid2)

       status = nf90_open(fi,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"ep4",varID)
       write(6,*) 'Variable ID for ep4 = ',varID
       status = nf90_get_var(ncid,varID,wa1)
       status = nf90_close(ncid)

       write(6,*) ep44(200,47,30),wa1(200,47,30)
                                                            
       status = nf90_create(fc,nf90_noclobber,ncid2)
       status = nf90_def_dim(ncid2,"latitude",nd,iy)
       status = nf90_def_dim(ncid2,"time",124,it)
       status = nf90_def_var(ncid2,"uref",nf90_float,   &
                (/iy,it/), vid2)
       status = nf90_put_att(ncid2,vid2,"title",fc)
       status = nf90_enddef(ncid2)
       status = nf90_put_var(ncid2,vid2,urb)
       status = nf90_close(ncid2)

       status = nf90_open(fc,nf90_nowrite,ncid)
       status = nf90_inquire(ncid,nDim,nVar,nAtt,uDimID)
       write(6,*) 'ndim,nvar,natt,uDimID =',nDim,nVar,nAtt,uDimID
       status = nf90_inq_varid(ncid,"uref",varID)
       write(6,*) 'Variable ID for uref = ',varID
       status = nf90_get_var(ncid,varID,wa2)
       status = nf90_close(ncid)

       write(6,*) urb(47,30),wa2(47,30)
                                                             
       close(41)

        enddo
        enddo

        stop
        end
