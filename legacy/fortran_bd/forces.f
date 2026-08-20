	SUBROUTINE forces(F, r, nxyz)
 
	  common / num_par / np
	  common / np_3 / np3

c	  common / num_bond / nbond 
c	  common / bonds / bonds
	  
	  common / boxlen_xy / boxlenx, boxleny
	  
	  common / radius / a, radii

	  common / grav_pot / Fgrav, Fhw, rcut, tempr

c	  common / vdw_pot / Fvdw, pvdw

	  common / es_pot / pfpp, pfpw, kappa

	  common / f_pot / lambda, fcm, dg,re

c	  common / cad_pot / swd, swr
	  

c    ******************************************************************
c    ** evlauates the different force components                     **
c    ******************************************************************

	  integer, parameter :: rmax = 1, nmax=1000
	  
	  integer    i, j, np, np3, n, n3, nbond
	  
	  integer	 nxyz(3, np), bonds(2,nmax)

	  double precision		 a, r(np3), F(np3), rij(3), sep, boxlenx, 
     +						 rijsep, Fss, rijtemp(3), Fpp, sep2, 
     +						 Fgrav, Fsw, Fpw, Fhw, rcut, tempr, 
     +						 pfpp, pfpw, kappa, Fvdw, pvdw, boxleny,
     +						 swd, swr, radii(nmax), lambda, fcm, dg,
     +						 dE2x, dE2y, Fdepx, Fdepy, rdep

	  double precision		 Exi, Eyi, Ezi, Exj, Eyj, Ezj, Fo, 
     +						 F1, F2, F3, felx, fely, felz

	  double precision		 RT, EX1, EY1, EX2, EY2, EX3, EY3,
     +						 EX4, EY4, RX, RY, STEP, EMAGI, EMAGJ,EMAG,re,
     + Exidx,Exjdx,Eyidx,Eyjdx,Exidy,Exjdy,Eyidy,Eyjdy,Udx,Udy,rijsepdx,
     + rijsepdy,fddpx,fddpy,step2,felxnew,felynew,felznew,U,de2Xnew,Upp,
     + Uppdx,Uppdy,step3,Fppx,Fppy,Fppxnew,Fppynew,Exidx2,Exjdx2,Eyidx2,
     + Eyjdx2,Exidy2,Exjdy2,Eyidy2,Eyjdy2,Udx2,Udy2,rijsepdx2,
     + rijsepdy2,Fddpx2,Fddpy2,felxnew2,felynew2,term1,term2,term3,
     + term1dx,term2dx,term3dx,term1dx2,term2dx2,term3dx2,
     + term1dy,term2dy,term3dy,term1dy2,term2dy2,term3dy2,Uppdx2,Uppdy2,
     + Fppxnew2,Fppynew2

	  double precision, parameter :: kb = 1.380658E-23, pi = 3.1416
	

	  F = 0.d0

	  Fsw = 0.0

	  Fo = 1e18*0.75*lambda*kb*(273+tempr)/a

	  rdepc = 16.5*a

	  STEP = 1E-3
	  step2=1
	  step3=1e-3

c
c	particle-particle forces
c
	  do i=1,np-1

		Exi = -4*r(nxyz(1,i))/dg

		Eyi = 4*r(nxyz(2,i))/dg

		Ezi = 0

	   do j=i+1,np

		  rijtemp(1) = r(nxyz(1,j)) - r(nxyz(1,i))
		  rijtemp(2) = r(nxyz(2,j)) - r(nxyz(2,i))
		  rijtemp(3) = r(nxyz(3,j)) - r(nxyz(3,i))
			

c		  rij(1) = rijtemp(1) - boxlenx*anint(rijtemp(1)/boxlenx)
c		  rij(2) = rijtemp(2) - boxleny*anint(rijtemp(2)/boxleny)
	      rij(1) = rijtemp(1)
		rij(2) = rijtemp(2)
		rij(3) = rijtemp(3)

		  rijsep = sqrt( rij(1)**2 + rij(2)**2 + rij(3)**2 )
		
		
		  if( rijsep .le. (radii(i)+radii(j)) ) then

		    Fpp = Fhw
			
			felx = 0.0

			fely = 0.0

			felz = 0.0
			
c			write(*,*) i,j,'overlap','rij=',rijsep

		  elseif( rijsep  .lt. rcut ) then
			
			Fpp = 1e18*kb*(tempr+273)*kappa*pfpp*
     +			exp(-kappa*(rijsep - (radii(i)+radii(j)))/a)/a
     
c            Upp=pfpp*exp(-kappa/a*(rijsep-(radii(i)+radii(j))))
            
c            rijsepdx=sqrt( (rij(1)-step3)**2 + rij(2)**2)
            
c            Uppdx=pfpp*exp(-kappa/a*(rijsepdx-(radii(i)+radii(j))))
            
c            rijsepdy=sqrt( rij(1)**2 + (rij(2)-step3)**2)
            
c            Uppdy=pfpp*exp(-kappa/a*(rijsepdy-(radii(i)+radii(j))))
            
c             rijsepdx2=sqrt( (rij(1)+step3)**2 + rij(2)**2)
            
c            Uppdx2=pfpp*exp(-kappa/a*(rijsepdx2-(radii(i)+radii(j))))
            
c            rijsepdy2=sqrt( rij(1)**2 + (rij(2)-step3)**2)
            
c            Uppdy2=pfpp*exp(-kappa/a*(rijsepdy2-(radii(i)+radii(j))))

            else
            			Fpp = 0.d0
c            			Upp=0.0
c            			Uppdx=0.0
c            			Uppdy=0.0
c            			Uppdx2=0.0
c            			Uppdy2=0.0
           end if
           
           if(rijsep .gt. (radii(i)+radii(j)) .and. rijsep .lt. re) then
			Exj = -4*r(nxyz(1,j))/dg

			Eyj = 4*r(nxyz(2,j))/dg

			Ezj = 0
			
c			term1=Exi*(RIJ(1)/RIJsep)+EYi*(RIJ(2)/RIJsep)
c     			term2=Exj*(RIJ(1)/RIJsep)+EYj*(RIJ(2)/RIJsep)
c     			term3=Exi*Exj+Eyi*Eyj
     			
c     			U=-0.5*LAMBDA*((2*a)/RIJsep)**3*
c     +(3*TERM1*TERM2-TERM3)
     
     
c            Exidx=-4*(r(nxyz(1,i))+step2)/dg
c            Eyidx = 4*r(nxyz(2,i))/dg
c            Exjdx = -4*(r(nxyz(1,j)))/dg
c			Eyjdx = 4*r(nxyz(2,j))/dg
c			rijsepdx=sqrt( (rij(1)-step2)**2 + rij(2)**2)
c      	term1dx=Exidx*((RIJ(1)-step2)/RIJsepdx)+EYidx*(RIJ(2)/RIJsepdx)
c        term2dx=Exjdx*((RIJ(1)-step2)/RIJsepdx)+EYjdx*(RIJ(2)/RIJsepdx)
c     		term3dx=Exidx*Exjdx+Eyidx*Eyjdx
     		
c     		    Exidx2= -4*(r(nxyz(1,i)))/dg
c            Eyidx2 = 4*(r(nxyz(2,i)))/dg
c            Exjdx2 = -4*(r(nxyz(1,j))+step2)/dg
c			Eyjdx2 = 4*r(nxyz(2,j))/dg
c			rijsepdx2=sqrt( (rij(1)+step2)**2 + rij(2)**2)
c      term1dx2=Exidx2*((RIJ(1)+step2)/RIJsepdx2)+
c     + EYidx2*(RIJ(2)/RIJsepdx2)
c      term2dx2=Exjdx2*((RIJ(1)+step2)/RIJsepdx2)+
c     + EYjdx2*(RIJ(2)/RIJsepdx2)
c     	term3dx2=Exidx2*Exjdx2+Eyidx2*Eyjdx2
     		
c     		Udx=-0.5*LAMBDA*((2*a)/RIJsepdx)**3*
c     +(3*TERM1dx*TERM2dx-TERM3dx)
c        Udx2=-0.5*LAMBDA*((2*a)/RIJsepdx2)**3*
c     +(3*TERM1dx2*TERM2dx2-TERM3dx2)

c            Exidy=-4*r(nxyz(1,i))/dg
c            Eyidy = 4*(r(nxyz(2,i))+step2)/dg
c            Exjdy = -4*r(nxyz(1,j))/dg
c			Eyjdy = 4*r(nxyz(2,j))/dg
c			rijsepdy=sqrt( rij(1)**2 + (rij(2)-step2)**2)
c      	term1dy=Exidy*(RIJ(1)/RIJsepdy)+EYidy*((RIJ(2)-step2)/RIJsepdy)
c        term2dy=Exjdy*(RIJ(1)/RIJsepdy)+EYjdy*((RIJ(2)-step2)/RIJsepdy)
c     		term3dy=Exidy*Exjdy+Eyidy*Eyjdy
c         Udy=-0.5*LAMBDA*((2*a)/RIJsepdy)**3*
c     +(3*TERM1dy*TERM2dy-TERM3dy)
     
c            Exidy2=-4*r(nxyz(1,i))/dg
c            Eyidy2 = 4*(r(nxyz(2,i)))/dg
c            Exjdy2 = -4*r(nxyz(1,j))/dg
c			Eyjdy2 = 4*(r(nxyz(2,j))+step2)/dg
c			rijsepdy2=sqrt( rij(1)**2 + (rij(2)+step2)**2)
c      	term1dy2=Exidy2*(RIJ(1)/RIJsepdy2)+
c     + 	EYidy2*((RIJ(2)+step2)/RIJsepdy2)
c        term2dy2=Exjdy*(RIJ(1)/RIJsepdy2)+
c     +   EYjdy2*((RIJ(2)+step2)/RIJsepdy2)
c     		term3dy2=Exidy2*Exjdy2+Eyidy2*Eyjdy2
c         Udy=-0.5*LAMBDA*((2*a)/RIJsepdy)**3*
c     +(3*TERM1dy*TERM2dy-TERM3dy)
c          Udy2=-0.5*LAMBDA*((2*a)/RIJsepdy2)**3*
c     +(3*TERM1dy2*TERM2dy2-TERM3dy2)

c            fddpx=-(Udx-U)/step2*kb*(273+tempr)*1e18
c            fddpx2=-(Udx2-U)/step2*kb*(273+tempr)*1e18
c            fddpy=-(Udy-U)/step2*kb*(273+tempr)*1e18
c            fddpy2=-(Udy2-U)/step2*kb*(273+tempr)*1e18
c        if(rijsep .lt. 3000) then
c       write(*,*) i,j,'overlap','rij=',rijsep
c        end if
        
     
c		Electrostatic Dipole Interactions
c		see Aubry and Singh, Europhysics Lett., 2006
		  F1 = Exi*Exj + Eyi*Eyj + Ezi*Ezj

		  F2 = rij(1)*Exi/rijsep + rij(2)*Eyi/rijsep 
     +		+ rij(3)*Ezi/rijsep

	      F3 = rij(1)*Exj/rijsep + rij(2)*Eyj/rijsep 
     +		+ rij(3)*Ezj/rijsep

c		  felx = Fo*(2*a/rijsep)**4*(F1*rij(1)/rijsep +
c     +		Exi*F3 + Exj*F2 - 5*F2*F3*rij(1)/rijsep)

c		  fely = Fo*(2*a/rijsep)**4*(F1*rij(2)/rijsep +
c     +		Eyi*F3 + Eyj*F2 - 5*F2*F3*rij(2)/rijsep)

		  felz = Fo*(2*a/rijsep)**4*(F1*rij(3)/rijsep +
     +		Ezi*F3 + Ezj*F2 - 5*F2*F3*rij(3)/rijsep) 
     
            felxnew=Fo*(2*a/rijsep)**4*(F1*rij(1)/rijsep +
     +		Exi*F3 + Exj*F2 - 5*F2*F3*rij(1)/rijsep+
     +  rijsep*16*r(nxyz(1,j))/dg**2/3.0-F3*4*(-rij(1))/dg)
            felynew=Fo*(2*a/rijsep)**4*(F1*rij(2)/rijsep +
     +		Eyi*F3 + Eyj*F2 - 5*F2*F3*rij(2)/rijsep+
     +  rijsep*16*r(nxyz(2,j))/dg**2/3.0-F3*4*(rij(2))/dg)
            
           
            
            felxnew2=Fo*(2*a/rijsep)**4*(F1*rij(1)/rijsep +
     +		Exi*F3 + Exj*F2 - 5*F2*F3*rij(1)/rijsep-
     +  rijsep*16*r(nxyz(1,i))/dg**2/3.0+F2*4*(-rij(1))/dg)
            felynew2=Fo*(2*a/rijsep)**4*(F1*rij(2)/rijsep +
     +		Eyi*F3 + Eyj*F2 - 5*F2*F3*rij(2)/rijsep-
     +  rijsep*16*r(nxyz(2,i))/dg**2/3.0+F2*4*(rij(2))/dg)
            
             felznew=felz
		  else



			felx = 0.0

			fely = 0.0

			felz = 0.0
			
			fddpx=0.0
			fddpy=0.0
			felxnew=0.0
			felynew=0.0
			felxnew2=0.0
			felynew2=0.0
			fddpx2=0.0
			fddpy2=0.0

		  endif
c	
		  Fss = Fpp		
		  
c		  Fppx=	Fss*rij(1)/rijsep
c		  Fppy=	Fss*rij(2)/rijsep
		  
		  
		  
c		  Fppxnew=-(Uppdx-Upp)/step3*kb*(273+tempr)*1e18
c		  Fppynew=-(Uppdy-Upp)/step3*kb*(273+tempr)*1e18
		  
c		  Fppxnew2=-(Uppdx2-Upp)/step3*kb*(273+tempr)*1e18
c		  Fppynew2=-(Uppdy2-Upp)/step3*kb*(273+tempr)*1e18
		  	
c
		  F(nxyz(1,i))=F(nxyz(1,i)) -felxnew - Fss*rij(1)/rijsep
		  F(nxyz(2,i))=F(nxyz(2,i)) -felynew - Fss*rij(2)/rijsep
		  F(nxyz(3,i))=F(nxyz(3,i)) - felz - Fss*rij(3)/rijsep

		  F(nxyz(1,j))=F(nxyz(1,j)) +felxnew2 + Fss*rij(1)/rijsep
		  F(nxyz(2,j))=F(nxyz(2,j)) +felynew2+ Fss*rij(2)/rijsep
		  F(nxyz(3,j))=F(nxyz(3,j)) + felz + Fss*rij(3)/rijsep

	   end do

	  end do

c	add field and wall forces
	
	do i = 1,np

		RX = r(nxyz(1,i))
		RY = r(nxyz(2,i))

		EMAGI = EMAG(RX,RY)

		RX = r(nxyz(1,i)) + STEP
		RY = r(nxyz(2,i))

		EMAGJ = EMAG(RX,RY)
		dE2x = (EMAGJ**2-EMAGI**2)/STEP

c        de2Xnew=32.0/dg**2*r(nxyz(1,i)) for verification of numerical method
		
		RX = r(nxyz(1,i))
		RY = r(nxyz(2,i)) + STEP

		EMAGJ = EMAG(RX,RY)
		dE2y =  (EMAGJ**2-EMAGI**2)/STEP
	
c		Dimensionless E-field
C		rdep = sqrt(r(nxyz(1,i))**2+r(nxyz(2,i))**2)
C		if(rdep.lt.rdepc)then
C			dE2x = 32*r(nxyz(1,i))/dg**2
C			dE2y = 32*r(nxyz(2,i))/dg**2
C		else
C			dE2x = 72* r(nxyz(1,i))/dg**2
C			dE2y = 72*r(nxyz(2,i))/dg**2
C		endif

c		dE2z = 0

c		Dielectrophoresis force components
		Fdepx = (2*1e18*kb*(tempr+273)*lambda/fcm)*dE2x

		Fdepy = (2*1e18*kb*(tempr+273)*lambda/fcm)*dE2y

c		Fdepz = (2*1e18*kb*(tempr+273)*lambda*dpf/fcm)*dE2z

c		Fwall =	1e18*kb*(tempr+273)*kappa_a
c     +			*bpw*exp(-kappa_a*(r(nxyz(3,i))/a-1))/a

		F(nxyz(1,i))=F(nxyz(1,i)) + Fdepx*(radii(i)/a)**3

		F(nxyz(2,i))=F(nxyz(2,i)) + Fdepy*(radii(i)/a)**3

c		F(nxyz(3,i))=F(nxyz(3,i)) + Fdepz - 
c     +		Fgrav*1e18*kb*(tempr+273)/a + Fwall

	enddo

c
c	add forces with the wall
c
c	  do i = 1, np

c		sep = r(nxyz(3,i)) - a

c		if( sep .le. 0.0 ) then

c		    Fpw = Fhw

c		elseif ( sep / 30.5192 .lt. 700.0 ) then
					
c		    Fpw = 2.0*pfpp*exp ( - sep / kappa )

c		else

c			Fpw = 0.d0

c		endif

c		F(nxyz(3,i)) = F(nxyz(3,i)) + Fpw

c	  end do

	  return
	  end
