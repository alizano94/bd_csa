# depends on FORTRAN compilers, command to invloke the fortran compiler
FC = gfortran

#please check your compiler for these options
FFLAGS = -O


2d  =   caldss.o conn6calc.o ddpot.o emag.o\
        forces.o gasdev.o main.o QuadControl.o\
        pwexct.o ran2.o readcn.o writcn.o\
        $(libs)

.f.o:;  $(FC) $(FFLAGS) -c $<

bdpd: $(2d)
	  $(FC) $(FFLAGS) -o $@ $(2d)


clean:   
	rm -f core* bdpd *.o *.il




