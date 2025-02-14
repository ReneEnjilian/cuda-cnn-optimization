# Compiler settings
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = clang
CXXFLAGS = -O3 -std=c++17

# Include directories
INCLUDE_DIRS = -I$(CUDA_HOME)/include

# Libraries
LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda

# Source and executable
SRC = cnn_optimized.cu
EXEC = cnn_optimized

# Flags for compiling
NVCC_FLAGS = -arch=sm_80 -lineinfo -Xcompiler $(CXXFLAGS)

# Target to build the executable
all: $(EXEC)

# Rule for compiling the CUDA source file
$(EXEC): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(EXEC) $(LIBS)

# Clean target to remove object files and executable
clean:
	rm -f $(EXEC)

# Run the program
run: $(EXEC)
	./$(EXEC)

# Phony targets (not actual files)
.PHONY: all clean run