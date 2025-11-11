CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/nullKernelAsync ./bin/nullKernelSync ./bin/Break-even_miro ./bin/Data_trans_D2H_miro ./bin/Data_trans_H2D_miro

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/nullKernelAsync: ./src/nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/nullKernelSync: ./src/nullKernelSync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/Break-even_miro: ./src/Break-even_miro.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/Data_trans_D2H_miro: ./src/Data_trans_D2H_miro.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/Data_trans_H2D_miro: ./src/Data_trans_H2D_miro.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)