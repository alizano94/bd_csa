		FUNCTION EMAG( RX, RY )
		
		common / f_pot / lambda, fcm, dg,re
		common/ ecorr/ ecorrectflag

C    *******************************************************************
C    ** CALCULATES ELECTRIC FIELD MAGNITUDE                           **
C    *******************************************************************
        integer ecorrectflag
		DOUBLE PRECISION lambda, fcm, dg
		DOUBLE PRECISION PI, RX, RY, RT, EMAG,re
		DOUBLE PRECISION EX1, EX2, EX3, EX4, EX, EY
		DOUBLE PRECISION EY1, EY2, EY3, EY4,correctfactor

		PI = 3.1415927

C	E-field magnitude for interdigitated electrode.
c	FE is a geometric factor that accounts for
c	finite electrode size.
c	For exact solution for infinite electrodes see
c	Marcuse, IEEE Quantum Electronics, 1982

c		EMAG = FE/PI/((0.25+(RZ/dg)**2 
c     +		- (RX/dg)**2)**2 + (2*RZ*RX/dg**2)**2)**0.25

C	E-field magnitude for quadpole electrode.
c	For exact solution see
c	Huang and Pethig, Meas. Sci. Tech., 1991

		RT = SQRT((RX-0.5*DG)**2+RY**2)
		EX1 = (RX-0.5*DG)/RT**2
		EY1 = RY/RT**2

		RT = SQRT((RX+0.5*DG)**2+RY**2)
		EX2 = (RX+0.5*DG)/RT**2
		EY2 = RY/RT**2

		RT = SQRT((RY-0.5*DG)**2+RX**2)
		EY3 = (RY-0.5*DG)/RT**2
		EX3 = RX/RT**2

		RT = SQRT((RY+0.5*DG)**2+RX**2)
		EY4 = (RY+0.5*DG)/RT**2
		EX4 = RX/RT**2

		EX = EX1 + EX2 + EX3 + EX4
		EX = EX*DG

		EY = EY1 + EY2 + EY3 + EY4
		EY = EY*DG

		EMAG = SQRT(EX**2 + EY**2)

		RT = SQRT(RX**2 + RY**2)

C		IF(RT.LT.16.5)THEN
			EMAG = 4*RT/DG
		if(ecorrectflag .eq. 1) then
		
		correctfactor=2.081e-7*(RT/1000.0)**4-1.539e-9*(RT/1000.0)**3+
     +	8.341e-5*(RT/1000)**2+1.961e-5*(RT/1000)+1.028
		Emag=Emag*correctfactor
		
		end if	
			
C		ELSE
C			EMAG = 6*RT/DG
C		ENDIF


		END