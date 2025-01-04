#
# Makefile for CS 70 Rock/Paper/Scissors
# 
# As usual, you can build this software by typing
#
#     make
#

.SUFFIXES: .cpp .o .f90
.f90.o:
	$(F90) $(F90FLAGS) -c $<

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INC) -c $<

.cc.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INC) -c $<

# ----- Make Macros -----
F90	=	gfortran
F90FLAGS	= -g -O3 -cpp  #-fopenmp  # -fcheck=all -fbacktrace
#F90FLAGS	= -g -O3 -cpp  # -fcheck=all -fbacktrace

OPTFLAGS = -O3
CXX	=  g++

INC = -I/usr/include

LIB = -larmadillo -lgsl -lgslcblas 
CXXFLAGS = -g -std=c++1y $(OPTFLAGS) #-fopenmp #-Wall -Wextra

CUR_DIR = `pwd`
TARGETS = run	
OBJECTS =	run.o mc_ising.o field.o rate_table.o
# ----- Make Rules -----
all:	$(TARGETS) 

run: $(OBJECTS)  	
	$(CXX) $(OBJECTS) $(CXXFLAGS) -o run $(INC) $(LIB)

clean:
	rm -f $(TARGETS) $(OBJECTS) *.o wisdom*.txt

