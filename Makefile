CC=gcc
CPP=g++
NVCC=nvcc
# Foi trocado REAL por REALN pois o open MPI já define um REAL
# o que estava causando conflitos na compilação
REALN ?= float
EXE ?= dmbrot
CFLAGS=-ansi -Wall -DREALN=$(REALN)
CFLAGS_CUDA= -DREALN=$(REALN)
PNG_FLAGS=-ansi -pedantic -Wall -Wextra -O3
LDFLAGS=-fopenmp -lm -lmpi
LDFLAGS_CUDA=-lm -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
INCLUDES=-I/usr/lib/x86_64-linux-gnu/openmpi/include/

main: dmbrot

dmbrot: mandelbrot.o lodepng.o
	$(NVCC) $(CFLAGS_CUDA) $(INCLUDES) -o $(EXE) $^ $(LDFLAGS_CUDA)

mandelbrot.o: mandelbrot.cu lodepng.h
	$(NVCC) $(CFLAGS_CUDA) $(INCLUDES) -c $<

lodepng.o: lodepng.cpp lodepng.h
	$(CPP) $(PNG_FLAGS) -c $<

clean:
	rm -f *.o dmbrot
