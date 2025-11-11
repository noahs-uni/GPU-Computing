NVCC=nvcc
CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64

.PHONY: build
build: ./bin/memCpy ./bin/Data_trans_D2D_miro

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/main.cpp ./src/kernel.cu
	@mkdir -p ./bin
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
	
./bin/Data_trans_D2D_miro: ./src/Data_trans_D2D_miro.cu
	@mkdir -p ./bin
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
