	subroutine pwexact(pw, n1, r, nxyz)
	
	  common / num_par / np
	  common / np_3 / np3

	  common / boxlen_z / boxlenz

	  common / radius / a

	  common / brsh / brush_length
c
c	subroutine to calculate the exact particle-wall mobility tensor
c

c
c	declare variables
c
	integer		n1, i, j, np, np3

	integer		nxyz(3, np)

	double precision  a, r(np3), PW(3), v, Aw, Bw
c

	PW = 0.d0

	v = ( r(nxyz(3,n1)) - a ) / a

	if(v.lt.0.d0) v = 1.0d-8

c
c	calculate the A and B constants in the expression for the mobility tensor
c
	Aw = (12420*v**2 + 5654*v**2 + 100*v) 
     +   / (12420*v**3 + 12233*v**2 + 431*v)

	Bw = (6.0*v**2+2.0*v) / (6.0*v**2+9.0*v+2.0)
c
c	start loops over particle coordinates
c

	pw(1) = Aw
	pw(2) = Aw
	pw(3) = Bw
c
c
	return
	end
c