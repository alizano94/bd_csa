	  PROGRAM SBD

	  common / num_par / np
	  common / num_tot / n
	  common / np_3 / np3
      common / boxlen_xy / boxlenx, boxleny
	  common / radius / a, radii
	  common / delta_fn / delta
	  common / seed / iDummy
	  common / grav_pot / Fgrav, Fhw, rcut, tempr
	  common / es_pot / pfpp, pfpw, kappa
	  common / f_pot / lambda, fcm, dg,re
	  COMMON / ORDER_PAR / CONN6AVG, RMIN,PSI6,RGMEAN,RC
	  COMMON / EXP_PARA / expbox, VAR
      common / calds/ rgdsmin,delrgdsmin,
     +  distmin,deldist
      common / calds2/ rgdssbin,distdssbin
      common/ ecorr/ ecorrectflag

c    ******************************************************************
c    ** This program runs the actual simulation and writes           **
c    ** coordinates into an output file.                             **
c    ******************************************************************
	  
	  integer, parameter :: m = 100, nmax=1000
	  integer          i, j, k, l, np, np3, l1,nout
	  integer		   nstep, iprint, istart, iDummy, nbond, 
     +  bonds(2,nmax)
      integer, allocatable :: nxyz(:, :)
	  double precision, parameter :: pi = 3.1416, sqpi = 1.7725, 
     +								 twopi = 6.2831853, tol=1d-5
	  double precision, allocatable :: F(:), D(:), randisp(:), 
     +								   r0(:), u0(:), u(:),
     +								   ud(:), r(:)
	  double precision t, a, tempr, dt, boxlenx, dt_m, hlev,
     +				   m_2, fac1, fac2, istart_dt, t_min_isdt, vol, 
     +				   phi, Fgrav, Fhw, pw(3), temp, pfpp, pfpw,
     +				    kappa, pwfactor, Fvdw, pvdw, hhw, 
     +				   expbox(1:2), VAR, boxleny, swr, swd, 
     +				   radii(nmax), rcut, lambda, fcm, dg, 
     +				   CONN6AVG, RMIN, xm, ym, rgx, rgy, rg,gasdev
      double precision delrg, delpsi, rghist(700), psihist(120), RGMIN,
     +  conhist(100), delcon, rgmean, ophmax, psi6,
     + rgarray(nmax),distarray(nmax),dssarray(nmax,nmax),
     + dsscalcu(nmax),re,RC,
     + rgdsmin,delrgdsmin,distmin,deldist,dssmin,dssmax,dpf
      character        par_in*30, par_out*30, check, bondfile*30,
     + polymono*4, pdfile*30,RGHISTFILE*30, psihistfile*30,
     + conhistfile*30,temp3*3,rgdsfile*150

	  REAL          TIMER, T1, T2

	  INTEGER       HOUR, MINUTE, SECOND, cycles, cycleindex, jumpstep1,
     +	             jumpstep2
	  integer conbin, psibin, rgdssbin,distdssbin,cyclenum,rgarraybin,
     +  dsscount(nmax,nmax),ecorrectflag,flag,controP


	 INTEGER :: num_args
	 CHARACTER(len=12), DIMENSION(:), ALLOCATABLE :: args

C    ** BEGIN CLOCKING THE PROGRAM **
        CALL CPU_TIME(T1)  

	  num_args = COMMAND_ARGUMENT_COUNT()
	  ALLOCATE(args(num_args))
	  DO i = 1, num_args
		CALL GET_COMMAND_ARGUMENT(i, args(i))
	  END DO

	  READ(args(1), *) lambda
	  READ(args(2), *) iDummy
	  
	  open(1,file='run.txt')
	  read(1,*)
	  read(1,*)np
	  read(1,*)
	  read(1,*)nstep
	  read(1,*)
	  read(1,*)iprint
	  read(1,*)
	  read(1,*)istart
	  read(1,*)
	read(1,*)par_in
	read(1,*)
	read(1,*)par_out
	read(1,*)
	read(1,*)a
	read(1,*)
	read(1,*)tempr
	read(1,*)
	read(1,*)phi
	read(1,*)
	read(1,*)dt
	read(1,*)
	read(1,*)t
	read(1,*)
	read(1,*)check
	read(1,*)
	read(1,*)fac1
	read(1,*)
	read(1,*)fac2
	read(1,*)
	read(1,*)pwfactor
	read(1,*)
	read(1,*)Fgrav
	read(1,*)
	read(1,*)rcut
	read(1,*)
	read(1,*) re
	read(1,*)
	read(1,*)kappa
	read(1,*)
	read(1,*)pfpp
	read(1,*)
	read(1,*)pfpw
	read(1,*)
	read(1,*)
	read(1,*)
	read(1,*)fcm
	read(1,*)
	read(1,*)dg
	read(1,*)
	read(1,*)hlev
	read(1,*)
	read(1,*)rmin
	read(1,*)
	read(1,*)expbox(1),expbox(2)
	read(1,*)
	read(1,*)var
	read(1,*)
	read(1,*)polymono
	read(1,*)
	read(1,*)pdfile
	read(1,*)
	read(1,*)
      READ(1,*) 
	  READ(1,*) RGHISTFILE, DELRG, RGMIN
	  READ(1,*)
	  READ(1,*) PSIHISTFILE, DELPSI
	  read(1,*)
	  read(1,*) conhistfile, delcon	  
        read(1,*)
        read(1,*) cyclenum
        read(1,*)
        read(1,'(a150)') rgdsfile
        read(1,*)
        read(1,*) rgdsmin,delrgdsmin, rgdssbin
        read(1,*)
        read(1,*) distmin,deldist,distdssbin
        read(1,*) 
        read(1,*) dssmin, dssmax
        read(1,*)
        read(1,*) dpf
        read(1,*)
        read(1,*) ecorrectflag
        
      lambda=lambda*dpf
      open(131,file=rgdsfile)
        
      do i=1,rgdssbin
      do j=1,distdssbin
      read(131,*) dum,dum,rgarray(i),distarray(j),dssarray(i,j),
     + dsscount(i,j)
      end do
      end do

        rgarraybin=i-1
        
	  pfpp=pfpp*a

      rcut = rcut * a
	  re=re*a
	  dg = dg * a
	  hlev = a + a*hlev
	  conn6avg = 0.0
	  expbox = expbox * a
	  

	  conhist=0
	  rghist=0
  	  psihist=0

	  np3=np * 3

	  allocate ( F(np3), D(np3), randisp(np3),
     +			 r0(np3), u0(np3), u(np3), ud(np3), 
     +			 r(np3), nxyz(3, np))

	  if (polymono.eq.'poly') then
		open(1111,file=pdfile)
		a = 0.0
		do i=1, np
			read(1111,*) radii(i)
			radii(i)=radii(i)
			a = a + radii(i)
		end do
		close(1111)
		a = a/real(np)
	  else
		do i=1,np
			radii(i)=a
		end do
	  end if

	  do i = 1, np
		
	    nxyz(1, i) = 3 * (i-1) + 1
	    
		nxyz(2, i) = nxyz(1, i) + 1

		nxyz(3, i) = nxyz(2, i) + 1

	  end do  


	  boxlenx = dg
	  boxleny = dg

        	  dt_m = dt / real(m)

	  m_2 = real(m) / 2.d0
		        fac1=fac1/a
      fac2=fac2*sqrt((273+tempr)/a)
      fac2 = fac2 / sqrt(dt)

	  istart_dt = real(istart) * dt

	  do cycleindex=1,cyclenum
	  call readcn( par_in, check, r, nxyz, hlev )


          Fhw   = 0.417                  
c
c	since the wall particles are always fixed, we can calculate
c	mobility terms of wall-wall interactions beforehand and use
c	them throughout the simulation
c

	  	
c    **	Open file for writing simulation output   **

        r0=r

        if(cycleindex .lt. 10) then
        write(temp3,'(i1)') cycleindex
        else if(cycleindex .lt. 100) then
        write(temp3,'(i2)') cycleindex
        else
	    write(temp3,'(i3)') cycleindex
	    end if




      t=0.0
      
	  open ( unit = 20, file = trim(par_out)//trim(temp3)//'.txt' )
	  open ( unit = 40, file = 'op'//trim(temp3)//'.txt')
	  
c	  call CONN6CALC(R,NXYZ,IDUMMY)
c	  call caldss(dssarray,dsscalcu,dsscount,r,nxyz,dssmin,dssmax)
c	  write(40,'(5f15.5)') (t/1E3),conn6avg,rgmean, psi6,RC	 
c
c    **   Now start the dynamics   **
c 
      flag=1
      nout=0
      do l=1, nstep
      
	  if(mod(l,iprint).eq.0 .or. t.eq.0)then
	  call CONN6CALC(R,NXYZ,IDUMMY)
	  call caldss(dssarray,dsscalcu,dsscount,r,nxyz,dssmin,dssmax)
	  write(40,'(5f12.5,i5,f12.5)') (t/1E3),conn6avg,rgmean,
     +    psi6,RC,ControP,lambda
      call writcn(t,r,nxyz)	 
	  end if
	  
	  if(mod(l,iprint).eq.0 .or. t.eq.0)then
	  call CONN6CALC(R,NXYZ,IDUMMY)
	  call caldss(dssarray,dsscalcu,dsscount,r,nxyz,dssmin,dssmax)
	  call QuadControl(l,Psi6,lambda,ControP,conn6avg,flag,IDUMMY)
	  end if
	  
	  nout=nout+1
      t=t+dt
      t_min_isdt=t-istart_dt

c    ******************************************************************
c    ** call subroutines for calculating diffusion coefficients,     ** 
c    ** forces and random displacement terms                         **
c    ******************************************************************

c
c	calculate exact particle-wall diffusion tensor (3x3)
c	
	    do j = 1, np

c		  call pwexact(pw, j, r, nxyz)
	
		  do k = 1, 3

			r0(nxyz(k, j)) = r(nxyz(k, j))
		    u0(nxyz(k, j)) = 0d0
		    u(nxyz(k, j))  = 0d0

		    D(nxyz(k, j)) = dsscalcu(j)

			temp = gasdev(idummy)

			if (k.ne.3) then
			  randisp(nxyz(k, j)) = temp * sqrt(1.0 / dsscalcu(j))
			else
			  randisp(nxyz(k, j)) = 0.0
			end if

		  end do

		end do

	    call forces(F, r, nxyz)
c
c
		do j = 1, np3

		  u0(j) = u0(j) + D(j) * ( F(j)*fac1 + randisp(j)*fac2 )	
		  
		  r(j) = r0(j) + u0(j) * dt

			if( mod(j-1,3).eq.0 ) then

     	   		 r(j) = r(j) - anint ( r(j) / boxlenx ) * boxlenx

		  elseif( mod(j-2,3).eq.0 ) then

     			    r(j) = r(j) - anint ( r(j) / boxleny ) * boxleny

		  end if

		end do		
      end do
c    **	end loop over time steps   **

c

	  close(10)
	  close(20)
	 

	  end do  
 
	  deallocate ( F, D, randisp, r0, u0, u, ud, r, nxyz )

C    ** FINISH CLOCKING THE PROGRAM **
 400    CALL CPU_TIME(T2)
        TIMER  = T2 - T1
        HOUR   = 0
        MINUTE = 0
        SECOND = 0
        IF ( TIMER .GE. 3600 ) HOUR = INT( TIMER / 3600 )
        IF ( TIMER .GE. 60 ) MINUTE = INT( (TIMER - 3600 * HOUR) / 60 )
        SECOND = INT( ( TIMER - 3600 * HOUR - 60 * MINUTE ) )
        WRITE(*,60) HOUR, ':', MINUTE, ':', SECOND,
     &              ' ELAPSED (HH:MM:SS)'
60      FORMAT(I3,A,I2.2,A,I2.2,A)

	  stop
	  end
c
c
