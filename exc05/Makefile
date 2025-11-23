CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS= -Xptxas -flcm=cg -Xptxas -fscm=cg -O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/matMul

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/matMul: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ ./src/main.cu ./src/mmult_cpu.c $(INC) $(LIB)
