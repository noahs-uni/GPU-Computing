NVCC=nvcc
CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64

.PHONY: build
build: ./bin/memCpy, ./bin/Data_trans_D2D

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/main.cpp ./src/kernel.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/Data_trans_D2D: ./src/Data_trans_D2D.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
