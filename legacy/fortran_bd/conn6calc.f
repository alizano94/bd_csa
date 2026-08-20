	SUBROUTINE CONN6CALC(R,NXYZ,IDUMMY)

	common / num_par / np
	common / np_3 / np3
	COMMON / ORDER_PAR / CONN6AVG, RMIN, PSI6, RGMEAN, RC
	COMMON / EXP_PARA / expbox, VAR
	common / boxlen_xy / boxlenx, boxleny

	INTEGER					NP, I, J, IDUMMY, NPTEMP
      INTEGER, PARAMETER  ::  NMAX = 1000
	integer					nxyz(3, np)
	INTEGER					nb(nmax), CON(NMAX)
      DOUBLE PRECISION        BOXLENX, BOXLENY
      DOUBLE PRECISION        RX(NMAX), RY(NMAX), RZ(NMAX)
	DOUBLE PRECISION		RXIJ, RYIJ, THETA, psir(nmax),
     +						psii(nmax), numer, denom, r(np3),
     +						testv, ctestv, CONN6AVG, RMIN, 
     +						expbox(1:2),VAR, rxtemp, rytemp,
     +                  psi6, rgmean, xmean, ymean,gasdev,
     +                accumpsi6r, accumpsi6i,RC
      DOUBLE PRECISION RgLB,RgUB,RgNew,Wrc,Crc,Ra
	CTESTV = 0.32
	nb=0
	con=0
	psir=0.0
	psii=0.0
	NPTEMP = 0.0
        psiavg=0.0
        rgmean=0.0
	do i=1,np

		rxtemp = r(nxyz(1,i))
		rytemp = r(nxyz(2,i))

		IF(ABS(rxtemp).LE.0.5*EXPBOX(1)
     +	.AND.ABS(rytemp).LE.0.5*EXPBOX(2))THEN
			rx(i) = rxtemp
			rx(i) = rx(i) + var*gasdev(idummy)


			ry(i) = rytemp
			ry(i) = ry(i) + var*gasdev(idummy)
			
			rz(i) = r(nxyz(3,i))
c			rz(i) = rz(i) + var*gasdev(idummy)
			NPTEMP = NPTEMP + 1
		ENDIF
	enddo


!  GET PSI6 FOR ALL PARTICLES
	do i=1, NPTEMP
		do j=1, NPTEMP
			if (i.ne.j) then 
				rxij=rx(j)-rx(i)
				rxij=rxij-boxlenx*anint(rxij/boxlenx)
				ryij=ry(j)-ry(i)
				ryij=ryij-boxleny*anint(ryij/boxleny)
				RP=sqrt(RXIJ**2+RYIJ**2)
				if (RP.le.rmin) then
					nb(i)=nb(i)+1
					THETA=atan2(RYIJ,RXIJ)
					psir(i)=psir(i)+cos(6*theta)
					psii(i)=psii(i)+sin(6*theta)
				end if
			end if
		end do
		if (nb(i).ne.0) then
			psir(i)=psir(i)/dble(nb(i))
			psii(i)=psii(i)/dble(nb(i))
			
		end if
	end do

       
        psi6=0
		accumpsi6r=0
		accumpsi6i=0
		do i=1,np
		accumpsi6r=accumpsi6r+psir(i)
		accumpsi6i=accumpsi6i+psii(i)
		end do
		accumpsi6r=accumpsi6r/np
		accumpsi6i=accumpsi6i/np
		
		psi6=sqrt(accumpsi6r**2+accumpsi6i**2)
c		GET CONN6 FOR ALL PARTICLES

		conn6avg = 0.0

		do i=1, NPTEMP
			do j=1, NPTEMP
				rxij=rx(j)-rx(i)
				rxij=rxij-boxlenx*anint(rxij/boxlenx)
				ryij=ry(j)-ry(i)
				ryij=ryij-boxleny*anint(ryij/boxleny)
				rp=sqrt(rxij**2+ryij**2)
				if ((i.ne.j).and.(rp.le.rmin)) then  

					numer=psir(i)*psir(j)+psii(i)*psii(j)
					denom=sqrt(numer**2+(psii(i)*psir(j)-
     +					psii(j)*psir(i))**2)
					testv=numer/denom
					if (testv.ge.ctestv) then
						con(i)=con(i)+1
					end if
				end if
			end do
			conn6avg = conn6avg + real(con(i))
		end do

		conn6avg = conn6avg/real(nptemp)

	
c      calculate Rg
        xmean=0
        ymean=0
        
        do i=1,NPTEMP
        
        	xmean=xmean+rx(i)
        	ymean=ymean+ry(i)
        end do
        
        xmean=xmean/real(nptemp)

        ymean=ymean/real(nptemp)

        
        rgmean=0
        do i=1,nptemp
        
        rgmean=rgmean+(rx(i)-xmean)**2
        rgmean=rgmean+(ry(i)-ymean)**2
        
        end do
        
        rgmean=rgmean/real(nptemp)

        rgmean=sqrt(rgmean)
        
c     Calculate RC
      RgLB=18526.0
      RgUB=26500.0
      RgNew=rgmean
      if (RgNew .gt. RgUB)then
          RgNew=RgUB
      end if
      Ra=1-(RgNew-RgLB)/(RgUB-RgLB)
      Crc=conn6avg/5.6
      Wrc=1/(exp((Crc-0.5)*18)+1)
      RC=Wrc*Ra+(1-Wrc)*Crc   
        
        return
	end