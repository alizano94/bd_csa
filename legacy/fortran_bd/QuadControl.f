      SUBROUTINE QuadControl(l,Psi6,lambda,ControP,C6,flag,IDUMMY)
      
      
c     parameter defination 
      character OneDcontroller*30

      character(300) :: out_line
      
      integer,parameter ::N1=50,N2=120,N=N1*N2
      
      integer Controlpolicy(N),RealDPsi6,RealDC6,ControP,flag,l,IDUMMY
      
      double precision Psi6,dPsi6,Psi6_0,C6,C6_0,dC6,disPsi6,disC6
      
      double precision Voltage,lambda
      
      double precision conn6avg
      Psi6_0=0.0
      dPsi6=0.02
      C6_0=0.0
      dC6=0.05
      
      
      if (Psi6 .lt. Psi6_0) then
      Psi6=Psi6_0
      end if
      if (C6 .lt. C6_0) then
       C6=C6_0
      end if

      if (Psi6 .gt. 1) then
       Psi6=1
      end if
            
      if (C6 .gt. 6) then
      C6=6
      end if

      ! out_line = '{step:' // l //',lambda:' // lambda //
      ! +           ',psi_6:' // Psi6 // ',C6:' // C6 // '}'

      open(2,file='out_param.json')
      write(2,*) '{'
      write(2,*) '      "step":',l,','
      write(2,*) '      "lambda":',lambda,','
      write(2,*) '      "psi6":',Psi6,','
      write(2,*) '      "c6":',C6,','
      write(2,*) '      "seed":',IDUMMY
      write(2,*) '}'
      close(2)   
      
 
      return 
      end