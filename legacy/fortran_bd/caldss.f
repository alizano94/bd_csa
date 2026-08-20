        subroutine caldss(dssarray,dsscalcu,dsscount,r,nxyz,
     +  dssmin,dssmax   )
        
        COMMON / ORDER_PAR / CONN6AVG, RMIN,PSI6,RGMEAN
        common / calds/ rgdsmin,delrgdsmin,
     +  distmin,deldist
        common / calds2/ rgdssbin,distdssbin
        common / num_par / np
        
        integer, parameter:: nmax=1000
        integer rgarraybin,i,j,rgbinindex,distbinindex,
     +   rgdssbin,distdssbin
        double precision rgarray(nmax),dssarray(nmax,nmax),
     +   dsscalcu(nmax),
     +  conn6avg,rmin,Psi6,rgmean,dssmin,dssmax
        double precision rgdsmin,delrgdsmin,distmin,deldist
        integer dsscount(nmax,nmax)
        integer np, nxyz(3,np)
        double precision r(3*np)
        
        rgbinindex=int((rgmean-rgdsmin)/delrgdsmin)+1
        
        if(rgbinindex .le. 0) then
        rgbinindex=1
        end if
        
        calcudss=0.5*(dssmax+dssmin)
        xmean=0.0
        ymean=0.0
        
        do i=1,np
        
        xmean=xmean+r(nxyz(1,i))
        ymean=ymean+r(nxyz(2,i))
        
        end do
        
        xmean=xmean/dble(np)
        ymean=ymean/dble(np)
        
        do i=1,np
        
        disttemp=(r(nxyz(1,i))-xmean)**2+(r(nxyz(2,i))-ymean)**2
        
        disttemp=sqrt(disttemp)
        
        distbinindex=int((disttemp-distmin)/deldist)+1
        
        if(rgbinindex .ge. 1 .and. rgbinindex .le. rgdssbin) then
        
        if(distbinindex .ge. 1 .and. distbinindex .le. distdssbin) then
        
        if(dsscount(rgbinindex,distbinindex) .ge. 1) then
        dsscalcu(i)=dssarray(rgbinindex,distbinindex)
        else
        dsscalcu(i)=dssmax
        end if
        
        else
        
        dsscalcu(i)=dssmax
        end if
        
        elseif(rgbinindex .ge. rgdssbin) then
        dsscalcu(i)=dssmin
        
        
        end if
        
        end do
        
        end
         
        
       