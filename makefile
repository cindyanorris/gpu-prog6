NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30

#Optimization flags. Don't use this for debugging.
NVCCFLAGS = -c -m64 -O2 --compiler-options -Wall -Xptxas -O2,-v

#No optimizations. Debugging flags. Use this for debugging.
#NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = wrappers.o matMultiply.o h_matMultiply.o d_matMultiply.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

matMultiply: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o matMultiply

matMultiply.o: matMultiply.cu h_matMultiply.h d_matMultiply.h config.h

h_matMultiply.o: h_matMultiply.cu h_matMultiply.h CHECK.h

d_matMultiply.o: d_matMultiply.cu d_matMultiply.h CHECK.h config.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm matMultiply *.o
