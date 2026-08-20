	SUBROUTINE readcn( par_in, check, r, nxyz, hlev )
 
        common / num_par / np
	  common / np_3 / np3

	  common / radius / a
        
	  common / boxlen_xy / boxlen

c    *******************************************************************
c    ** subroutine to read in the configuration from unit 10          **
c    *******************************************************************

        integer, parameter :: cnunit1 = 10, cnunit2 = 20

	  integer				np3, np, i
	  
	  integer				nxyz(3, np)

        character				par_in*(*), check

        double precision		r(np3), a, dum, brush_length, boxlen,
     +						hlev


c   ********************************************************************

      open ( cnunit1, file = par_in )

	if(check.eq.'n')then

        do i = 1, np
	
          read ( cnunit1, * ) dum, r(nxyz(1,i)), r(nxyz(2,i)), 
     +							 r(nxyz(3,i))

		r(nxyz(1,i)) = r(nxyz(1,i)) * a
		r(nxyz(2,i)) = r(nxyz(2,i)) * a
c		r(nxyz(3,i)) = r(nxyz(3,i)) * a
		r(nxyz(3,i)) = hlev

        enddo

	else

	  do i=1, np

          read ( cnunit1, * ) dum, dum, r(nxyz(1,i)), r(nxyz(2,i)),
     +                                 r(nxyz(3,i))

        enddo

	endif

      close ( unit = cnunit1 )

      return
      end
