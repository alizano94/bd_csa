		FUNCTION DDPOT( RXI, RYI, RZI,  
     +			 RXJ, RYJ, RZJ, i, j)

		COMMON / FIELD / LAMBDA, RE, DG, FCM, DPF, FE, POTREF
		COMMON / BXLN / BOXLENX, BOXLENY
		COMMON / DDMM / UDPMAX, UDPMIN
		COMMON / RAD / RADIUS, radii
		common/cyclenum/ cycleindex

C    *******************************************************************
C    ** CALCULATES DIPOLE-DIPOLE INTERACTION POTENTIAL                **
C    *******************************************************************

	integer, parameter :: nmax=1000
	integer i, j,cycleindex
	DOUBLE PRECISION LAMBDA, RE, DG, FCM, DPF(nmax), FE, POTREF
	DOUBLE PRECISION RXI, RYI, RZI, EMAGI, EMAGJ, EMAGAVG
	DOUBLE PRECISION RXJ, RYJ, RZJ, PHI1, PHI2, EMAG
	DOUBLE PRECISION POIX, POIY, POIZ, POJX, POJY, POJZ
	DOUBLE PRECISION RIJ, TERM1, TERM2, TERM3, FL, FT
	DOUBLE PRECISION RXIJ, RYIJ, RZIJ, DDPOT, RATIO
	DOUBLE PRECISION UDPMAX, UDPMIN, THETA, RTEMP
	DOUBLE PRECISION BOXLENX, BOXLENY
	DOUBLE PRECISION PI, radius, radii(nmax)

	PI = 3.1415927

c	Calculate Dipole Moment components for sphere I
	
	POIX = -4*RXI/DG

	POIY = 4*RYI/DG

	POIZ = 0.0

C	Calculate Dipole Moment components for sphere J
	
	POJX = -4*RXJ/DG

	POJY = 4*RYJ/DG

	POJZ = 0.0

	RXIJ  = RXI - RXJ
      RYIJ  = RYI - RYJ
	RZIJ  = RZI - RZJ

c	RXIJ  = RXIJ - ANINT ( RXIJ / DG ) * DG
c      RYIJ  = RYIJ - ANINT ( RYIJ / BOXLENY ) * BOXLENY

      RIJ = SQRT ( RXIJ**2 + RYIJ**2 + RZIJ**2)

c	Terms for dipole-dipole potential
	TERM1 = POIX*(RXIJ/RIJ)+POIY
     +*(RYIJ/RIJ)+POIZ*(RZIJ/RIJ)

	TERM2 = POJX*(RXIJ/RIJ)+POJY
     +*(RYIJ/RIJ)+POJZ*(RZIJ/RIJ)

	TERM3 = POIX*POJX+POIY*POJY
     ++POIZ*POJZ

c	Time-averaged dipole-dipole potential
c	for non-uniform fields
c	see Furst and Gast, PRL, 1998
	DDPOT = -0.5*LAMBDA*((radii(i)+radii(j))/RIJ)**3*
     +(3*TERM1*TERM2-TERM3)*DPF(cycleindex)* 
     + (0.5*(radii(i)+radii(j)))**3

c	DDPOT = -0.5*LAMBDA*(2*A/RIJ/RADIUS)**3*
c     +DPF*(3*TERM1*TERM2-TERM3)*(A/RADIUS)**3

c	DDPOT = 0

c	DDPOT = -0.5*LAMBDA*(2/RIJ)**3*
c     +DPF*(3*COS(THETA)**2-1)
c     +*EMAGI*EMAGJ

C	DDPOT = -0.5*LAMBDA*(2*A/RIJ/RADIUS)**3
C     +*(A/RADIUS)**3*(3*COS(THETA)**2-1)
C     +*EMAGI*EMAGJ

	IF(DDPOT.GT.UDPMAX)UDPMAX=DDPOT
	IF(DDPOT.LT.UDPMIN)UDPMIN=DDPOT

	END