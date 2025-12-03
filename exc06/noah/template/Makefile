CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc

.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm -f ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/main.cpp ./src/kernel.cu
	$(NVCC) -arch=compute_60 -code=sm_70 -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
