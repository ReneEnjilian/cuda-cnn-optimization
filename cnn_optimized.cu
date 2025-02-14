/*****************************************************************************
* File: cnn_optimized.cu
*
* This file contains an optimized implementation of a convolutional neural 
* network developed entirely from scratch using CUDA. The optimized version 
* incorporates various techniques—such as overlapping host-device transfers,
* kernel fusion, and efficient memory management—to significantly improve 
* performance.
*
* Network Architecture:
*  - A convolutional layer fused with a ReLU activation for the forward pass.
*  - A max pooling layer fused with a flatten operation.
*  - A fully-connected layer.
*  - A softmax layer combined with cross-entropy loss.
*
* For the backward pass, a fused max pooling backward kernel is employed.
*
* Stochastic Gradient Descent (SGD) is used to update the network parameters.
*
* Compile: nvcc cnn_optimized.cu -o cnn_optimized
*****************************************************************************/


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------------
// Constants and hyperparameters
// ---------------------------------------------------------------------------------
#define TRAIN_IMAGES    60000
#define TEST_IMAGES     10000
#define IMAGE_ROWS      28
#define IMAGE_COLS      28
#define NUM_CLASSES     10

// Convolution layer
#define FILTER_SIZE     5
#define NUM_FILTERS     8
// Output from 5x5 kernel, stride=1 => 24x24
#define CONV_OUT_ROWS   (IMAGE_ROWS - FILTER_SIZE + 1)
#define CONV_OUT_COLS   (IMAGE_COLS - FILTER_SIZE + 1)

// MaxPool 2x2 => 12x12
#define POOL_SIZE       2
#define POOL_OUT_ROWS   (CONV_OUT_ROWS / POOL_SIZE)  
#define POOL_OUT_COLS   (CONV_OUT_COLS / POOL_SIZE)  
#define FLATTEN_SIZE    (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS)

// Training config
#define EPOCHS          5
#define BATCH_SIZE      64
#define LEARNING_RATE   0.01f

#define BLOCK_SIZE      256  // for generic kernels

// Triple buffering
#define NUM_BUFFERS     3

// ---------------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------------
inline void checkCudaError(const char *msg) {
#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if(err != cudaSuccess){                                                  \
            fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while(0)


// ---------------------------------------------------------------------------------
// MNIST Data Loading Functions
// ---------------------------------------------------------------------------------
void readMNISTImages(const char* fileName, float* data, int numImages) {
    FILE* f = fopen(fileName, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fileName);
        exit(1);
    }
    fseek(f, 16, SEEK_SET);
    for(int i = 0; i < numImages; i++){
        for(int p = 0; p < IMAGE_ROWS * IMAGE_COLS; p++){
            unsigned char pixel = 0;
            if (fread(&pixel, 1, 1, f) != 1) {
                fprintf(stderr, "Error reading pixel from %s\n", fileName);
                exit(1);
            }
            data[i * IMAGE_ROWS * IMAGE_COLS + p] = pixel / 255.0f;
        }
    }
    fclose(f);
}

void readMNISTLabels(const char* fileName, int* labels, int numLabels) {
    FILE* f = fopen(fileName, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fileName);
        exit(1);
    }
    fseek(f, 8, SEEK_SET);
    for(int i = 0; i < numLabels; i++){
        unsigned char lbl = 0;
        if (fread(&lbl, 1, 1, f) != 1) {
            fprintf(stderr, "Error reading label from %s\n", fileName);
            exit(1);
        }
        labels[i] = (int)lbl;
    }
    fclose(f);
}



// ---------------------------------------------------------------------------------
// Forward Kernels
// ---------------------------------------------------------------------------------

/***********************************************************************************
* Convolution Forward Kernel Fused with ReLU
*
* Optimizations Employed:
* 1. Shared Memory Tiling:
*    - Loads input data into shared memory (with tile dimensions 16×16 plus extra
*      padding for the 5×5 filter) to reduce redundant global memory accesses.
*
* 2. Constant Memory for Weights:
*    - Stores filter weights in constant memory for faster, cached access.
*
* 3. Loop Unrolling:
*    - Unrolls the inner convolution loops to reduce loop overhead and improve performance.
*
* 4. Fused Activation:
*    - Integrates the ReLU activation within the convolution computation, eliminating
*      the need for a separate kernel launch.
*
* 5. Boundary Checks:
*    - Ensures output indices remain within valid bounds to prevent illegal memory accesses.
***********************************************************************************/

__constant__ float constW[NUM_FILTERS * FILTER_SIZE * FILTER_SIZE];

__global__
void fusedConvReluKernel(const float* in, const float* w, const float* b,
                            float* out, int batchSize)
{
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int batchIdx = blockIdx.z / NUM_FILTERS;
    int filterIdx = blockIdx.z % NUM_FILTERS;
    const float* input = in + batchIdx * (IMAGE_ROWS * IMAGE_COLS);
    float* output = out + batchIdx * (NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS)
                        + filterIdx * (CONV_OUT_ROWS * CONV_OUT_COLS);
    extern __shared__ float sharedIn[];
    int sharedWidth = TILE_WIDTH + FILTER_SIZE - 1;
    int in_start_x = blockIdx.x * TILE_WIDTH;
    int in_start_y = blockIdx.y * TILE_HEIGHT;
    for (int i = threadIdx.y; i < TILE_HEIGHT + FILTER_SIZE - 1; i += blockDim.y) {
        for (int j = threadIdx.x; j < TILE_WIDTH + FILTER_SIZE - 1; j += blockDim.x) {
            int in_x = in_start_x + j;
            int in_y = in_start_y + i;
            float value = 0.0f;
            if(in_x < IMAGE_COLS && in_y < IMAGE_ROWS)
                value = input[in_y * IMAGE_COLS + in_x];
            sharedIn[i * sharedWidth + j] = value;
        }
    }
    __syncthreads();
    float result = 0.0f;
    if(out_x < CONV_OUT_COLS && out_y < CONV_OUT_ROWS){
        #pragma unroll
        for (int i = 0; i < FILTER_SIZE; i++){
            #pragma unroll
            for (int j = 0; j < FILTER_SIZE; j++){
                float inVal = sharedIn[(threadIdx.y + i) * sharedWidth + (threadIdx.x + j)];
                float wVal = constW[filterIdx * (FILTER_SIZE * FILTER_SIZE) + i * FILTER_SIZE + j];
                result += inVal * wVal;
            }
        }
        result += b[filterIdx];
        if(result < 0.0f) result = 0.0f;
        output[out_y * CONV_OUT_COLS + out_x] = result;
    }
}



/***********************************************************************************
* Fused Max Pooling and Flattening Forward Kernel
*
* Optimizations Employed:
* 1. Kernel Fusion:
*    - Combines max pooling and flattening into a single kernel, reducing global
*      memory traffic and kernel launch overhead.
*
* 2. Efficient Index Arithmetic:
*    - Computes multi-dimensional indices using modular arithmetic, minimizing
*      branches and control overhead.
*
* 3. Minimal Memory Overhead:
*    - Directly reads from and writes to global memory without extra temporary buffers.
***********************************************************************************/

__global__
void fusedMaxPoolFlattenKernel(const float* in, float* out, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if(index >= total) return;
    int b = index / (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem = index % (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int f = rem / (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem2 = rem % (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rOut = rem2 / POOL_OUT_COLS;
    int cOut = rem2 % POOL_OUT_COLS;
    int in_r = rOut * POOL_SIZE;
    int in_c = cOut * POOL_SIZE;
    int convStride = CONV_OUT_COLS;
    int base = b * (NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS)
             + f * (CONV_OUT_ROWS * CONV_OUT_COLS)
             + in_r * convStride + in_c;
    float v0 = in[base];
    float v1 = in[base + 1];
    float v2 = in[base + convStride];
    float v3 = in[base + convStride + 1];
    float maxVal = v0;
    if(v1 > maxVal) maxVal = v1;
    if(v2 > maxVal) maxVal = v2;
    if(v3 > maxVal) maxVal = v3;
    out[index] = maxVal;
}



/***********************************************************************************
* Fully Connected Forward Kernel 
*
* Optimizations Employed:
* 1. GEMM-Style Shared Memory Tiling:
*    - Implements a GEMM-style matrix multiplication by dividing the input (A) and 
*      weight (B) matrices into 16×16 tiles, significantly reducing redundant global 
*      memory accesses and enhancing data reuse.
*
* 2. Kernel Fusion for Bias Addition:
*    - Integrates the bias addition directly within the kernel, eliminating the need 
*      for a separate post-processing step.
*
* 3. Loop Unrolling:
*    - Unrolls the inner loop over the tile dimension to reduce loop overhead and 
*      improve instruction-level parallelism.
*
* 4. Bounds Checking:
*    - Uses conditional loads to ensure that out-of-bound indices are handled gracefully.
***********************************************************************************/


__global__
void fcForwardKernel(const float* in, const float* w, const float* b,
                     float* out, int batchSize)
{
    const int TILE_SIZE = 16;  // tile dimension
    // Determine global row (batch index) and column (class index)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Shared memory for input tile (A) and weight tile (B)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Compute number of tiles needed along the reduction dimension
    int numTiles = (FLATTEN_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (row < batchSize && tiledCol < FLATTEN_SIZE)
            As[threadIdx.y][threadIdx.x] = in[row * FLATTEN_SIZE + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        if (tiledRow < FLATTEN_SIZE && col < NUM_CLASSES)
            Bs[threadIdx.y][threadIdx.x] = w[tiledRow * NUM_CLASSES + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < batchSize && col < NUM_CLASSES)
        out[row * NUM_CLASSES + col] = sum + b[col];
}



/***********************************************************************************
* Softmax and Cross-Entropy Forward Kernel (Optimized)
*
* Optimizations Employed:
* 1. Single-Pass Exponential Computation:
*    - Computes the maximum logit and then, in a single loop, uses the fast __expf
*      intrinsic to calculate the exponentials while accumulating their sum.
*
* 2. Fused Computation:
*    - Stores the computed exponentials in a local array, enabling simultaneous 
*      calculation of the normalized probabilities and the loss.
*
* 3. Minimal Memory Footprint:
*    - Uses a small local array (size NUM_CLASSES) to hold intermediate results, 
*      minimizing global memory traffic.
*
* 4. Numerical Stability:
*    - Subtracts the maximum logit from each value before exponentiation to avoid 
*      overflow and maintain numerical precision.
***********************************************************************************/

__global__
void softmaxCrossEntropyForwardKernel(const float* logits, const int* labels,
                                      float* outLoss, float* outProb,
                                      int batchSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batchSize) return;
    int offset = i * NUM_CLASSES;
    float maxLogit = -1e30f;
    for (int c = 0; c < NUM_CLASSES; c++){
        float val = logits[offset + c];
        if(val > maxLogit)
            maxLogit = val;
    }
    float exp_vals[NUM_CLASSES];
    float sumExp = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++){
        float ex = __expf(logits[offset + c] - maxLogit);
        exp_vals[c] = ex;
        sumExp += ex;
    }
    int lbl = labels[i];
    float prob = exp_vals[lbl] / sumExp;
    float lossVal = -logf(prob + 1e-10f);
    for (int c = 0; c < NUM_CLASSES; c++){
        outProb[offset + c] = exp_vals[c] / sumExp;
    }
    outLoss[i] = lossVal;
}



// ---------------------------------------------------------------------------------
// Backward Kernels
// ---------------------------------------------------------------------------------

/***********************************************************************************
* Softmax and Cross-Entropy Backward Kernel
*
* Implementation:
*  - Computes the gradient by subtracting the one-hot encoded labels from the 
*    computed probabilities.
***********************************************************************************/

__global__
void softmaxCrossEntropyBackwardKernel(float* gradLogits, const float* prob,
                                       const int* labels, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batchSize * NUM_CLASSES) return;
    int sampleIdx = idx / NUM_CLASSES;
    int c = idx % NUM_CLASSES;
    int lbl = labels[sampleIdx];
    float p = prob[idx];
    float y = (c == lbl) ? 1.0f : 0.0f;
    gradLogits[idx] = p - y;
}



/***********************************************************************************
* Fully Connected Backward Gradient Parameter Kernel
*
* Optimizations Employed:
* 1. Tiled Processing:
*    - Each block processes a tile of 16 parameters, reducing kernel launch overhead
*      by handling multiple parameters per block.
*
* 2. Strided Accumulation:
*    - Threads in the x-dimension accumulate partial sums over the batch dimension
*      using a fixed stride.
*
* 3. Shared Memory Reduction:
*    - A 2D shared memory array (32×16) is used to perform a parallel reduction
*      across the warp, minimizing synchronization overhead.
*
* 4. Unified Handling:
*    - Both weight and bias gradients are computed in a single kernel.
***********************************************************************************/

__global__
void fcBackwardGradParamKernel(const float* gradOut, const float* in,
                               float* gradW, float* gradB,
                               int batchSize)
{
    // Define tile size (number of parameters processed per block)
    const int tileSize = 16;  // number of parameters per block (along y)
    // Total number of weight parameters:
    int totalW = FLATTEN_SIZE * NUM_CLASSES;
    // Total parameters = weights + biases
    int totalParams = totalW + NUM_CLASSES;
    
    // Each block processes 'tileSize' parameters.
    // Compute the global parameter index for this thread's column.
    int paramIdx = blockIdx.x * tileSize + threadIdx.y;
    if(paramIdx >= totalParams)
        return;
    
    float sum = 0.0f;
    // For weight gradients, compute k and c; for bias, just c.
    bool isWeight = (paramIdx < totalW);
    int k = 0, c = 0;
    if(isWeight) {
        k = paramIdx / NUM_CLASSES;
        c = paramIdx % NUM_CLASSES;
    } else {
        c = paramIdx - totalW; // bias index
    }
    
    // Each thread (within the block, along x dimension) reduces a portion of the batch.
    int stride = blockDim.x; // 32
    for (int b = threadIdx.x; b < batchSize; b += stride) {
        if(isWeight) {
            sum += in[b * FLATTEN_SIZE + k] * gradOut[b * NUM_CLASSES + c];
        } else {
            sum += gradOut[b * NUM_CLASSES + c];
        }
    }
    
    // Declare shared memory for reduction.
    __shared__ float sdata[32][tileSize]; 
    sdata[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();
    
    // Perform reduction along the x-dimension.
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if(threadIdx.x < s) {
            sdata[threadIdx.x][threadIdx.y] += sdata[threadIdx.x + s][threadIdx.y];
        }
        __syncthreads();
    }
    
    // Thread 0 in each column writes the result.
    if(threadIdx.x == 0) {
        if(isWeight)
            gradW[paramIdx] = sdata[0][threadIdx.y];
        else
            gradB[paramIdx - totalW] = sdata[0][threadIdx.y];
    }
}



/***********************************************************************************
* Fully Connected Backward Gradient Input Kernel
*
* Optimizations Employed:
* 1. Loop Unrolling:
*    - The inner loop over the NUM_CLASSES (10 iterations) is fully unrolled.
*
* 2. Fused Multiply-Add:
*    - Uses the fmaf intrinsic to perform multiplication and addition in a single step,
*      reducing rounding errors and instruction count.
*
* 3. Register-Level Accumulation:
*    - The entire reduction is performed in registers, minimizing memory accesses and overhead.
***********************************************************************************/

__global__
void fcBackwardGradInKernel(const float* gradOut, const float* w,
                            float* gradIn, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * FLATTEN_SIZE;
    if(idx >= total) return;
    int b = idx / FLATTEN_SIZE;
    int k = idx % FLATTEN_SIZE;
    float sumVal = 0.0f;
    int baseGrad = b * NUM_CLASSES;
    int baseW = k * NUM_CLASSES;
    // Fully unroll the 10 iterations using fmaf for fused multiply-add.
    sumVal = fmaf(gradOut[baseGrad + 0], w[baseW + 0], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 1], w[baseW + 1], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 2], w[baseW + 2], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 3], w[baseW + 3], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 4], w[baseW + 4], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 5], w[baseW + 5], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 6], w[baseW + 6], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 7], w[baseW + 7], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 8], w[baseW + 8], sumVal);
    sumVal = fmaf(gradOut[baseGrad + 9], w[baseW + 9], sumVal);
    
    gradIn[idx] = sumVal;
}



/***********************************************************************************
* Fused Max Pooling Backward Kernel (only one definition)
*
* Optimizations Employed:
* 1. Kernel Fusion:
*    - Merges the functionality of the original unflattenPoolKernel and 
*      maxPoolBackwardKernel into one, reducing global memory traffic and kernel 
*      launch overhead.
*
* 2. Efficient Index Mapping:
*    - Computes multi-dimensional indices from a flattened index for direct access
*      to the corresponding convOut elements without extra buffers.
*
* 3. Direct Gradient Write:
*    - Determines the winning index in the pooling window and writes the gradient
*      directly, avoiding atomic operations.
*
* 4. Minimal Overhead:
*    - Fuses the backward pooling operations into a single pass, reducing memory
*      accesses and synchronization requirements.
***********************************************************************************/

__global__
void fusedMaxPoolBackwardKernel(const float* convOut, const float* gradFlat,
                                float* gradConvOut, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if(index >= total) return;
    int b = index / (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem = index % (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int f = rem / (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem2 = rem % (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rOut = rem2 / POOL_OUT_COLS;
    int cOut = rem2 % POOL_OUT_COLS;
    int convStride = CONV_OUT_COLS;
    int in_r = rOut * POOL_SIZE;
    int in_c = cOut * POOL_SIZE;
    int base = b * (NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS)
             + f * (CONV_OUT_ROWS * CONV_OUT_COLS)
             + in_r * convStride + in_c;
    float v0 = convOut[base];
    float v1 = convOut[base + 1];
    float v2 = convOut[base + convStride];
    float v3 = convOut[base + convStride + 1];
    int maxIdx = 0;
    float maxVal = v0;
    if(v1 > maxVal){ maxVal = v1; maxIdx = 1; }
    if(v2 > maxVal){ maxVal = v2; maxIdx = 2; }
    if(v3 > maxVal){ maxVal = v3; maxIdx = 3; }
    float gradVal = gradFlat[index];
    int writeIdx = base;
    if(maxIdx == 1) writeIdx = base + 1;
    else if(maxIdx == 2) writeIdx = base + convStride;
    else if(maxIdx == 3) writeIdx = base + convStride + 1;
    gradConvOut[writeIdx] = gradVal;
}



/***********************************************************************************
* ReLU Backward Kernel
*
* Optimizations Employed:
* 1. Read-Only Caching:
*    - Uses __ldg to load x from global memory, leveraging the read-only cache for faster access.
*
* 2. Branchless Computation:
*    - Replaces explicit branching with a simple multiplication by the boolean expression 
*      (x > 0.0f), which evaluates to 1.0f or 0.0f, reducing divergence.
***********************************************************************************/

__global__
void reluBackwardKernel(const float* gradOut, const float* x,
                        float* gradIn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        // Use __ldg to load from read-only global memory and use branchless multiplication.
        float x_val = __ldg(&x[i]);
        // (x_val > 0.0f) evaluates to 1.0f if true, 0.0f if false.
        gradIn[i] = gradOut[i] * (x_val > 0.0f);
    }
}



/***********************************************************************************
* Convolution Backward Weight Kernel (Optimized with Warp-Level Reduction)
*
* Optimizations Employed:
* 1. Warp-Level Reduction:
*    - Each block (with 32 threads) computes one parameter's gradient using 
*      __shfl_down_sync for an efficient in-warp reduction without shared memory.
*
* 2. Strided Data Access:
*    - Threads iterate over the combined batch and spatial dimension in strides of 32,
*      improving memory coalescing and workload distribution.
*
* 3. Unified Parameter Handling:
*    - Computes both weight and bias gradients in a single kernel based on the parameter index.
***********************************************************************************/

__global__
void convBackwardWeightKernel(const float* in, const float* gradConvOut,
                              float* gradW, float* gradB, int batchSize)
{
    // Each block computes one parameter gradient.
    int convSize = CONV_OUT_ROWS * CONV_OUT_COLS;
    int totalW = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
    int totalParams = totalW + NUM_FILTERS;
    
    // Launch one warp (32 threads) per parameter.
    int paramIdx = blockIdx.x;  
    if(paramIdx >= totalParams) return;
    
    float sum = 0.0f;
    const int warpSize = 32;  // blockDim.x must be 32
    int N = batchSize * convSize;
    
    if(paramIdx < totalW) {
        // Weight gradient.
        int f = paramIdx / (FILTER_SIZE * FILTER_SIZE);
        int rem = paramIdx % (FILTER_SIZE * FILTER_SIZE);
        int kr = rem / FILTER_SIZE;
        int kc = rem % FILTER_SIZE;
        for (int i = threadIdx.x; i < N; i += warpSize) {
            int b = i / convSize;
            int s = i % convSize;
            int orow = s / CONV_OUT_COLS;
            int ocol = s % CONV_OUT_COLS;
            float g = gradConvOut[b * (NUM_FILTERS * convSize) + f * convSize + orow * CONV_OUT_COLS + ocol];
            float inp = in[b * (IMAGE_ROWS * IMAGE_COLS) + (orow + kr) * IMAGE_COLS + (ocol + kc)];
            sum += inp * g;
        }
    } else {
        // Bias gradient for filter f = paramIdx - totalW.
        int f = paramIdx - totalW;
        for (int i = threadIdx.x; i < N; i += warpSize) {
            int b = i / convSize;
            int s = i % convSize;
            int orow = s / CONV_OUT_COLS;
            int ocol = s % CONV_OUT_COLS;
            sum += gradConvOut[b * (NUM_FILTERS * convSize) + f * convSize + orow * CONV_OUT_COLS + ocol];
        }
    }
    
    // Perform warp-level reduction.
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(mask, sum, offset);
    
    if (threadIdx.x == 0) {
        if(paramIdx < totalW)
            gradW[paramIdx] = sum;
        else
            gradB[paramIdx - totalW] = sum;
    }
}



/***********************************************************************************
* Convolution Backward Input Kernel (Optimized with Per-Sample Tiling)
*
* Optimizations Employed:
* 1. Per-Sample Tiling:
*    - Each block processes one sample by loading the entire gradConvOut for that
*      sample into statically allocated shared memory, reducing redundant global accesses.
*
* 2. Flattened Loop for Filter/Kernel Offsets:
*    - Collapses the three nested loops (over filters and kernel offsets) into a single
*      loop over totalKernel iterations, simplifying index calculations.
*
* 3. Cooperative Data Loading:
*    - Threads in a block cooperatively load gradConvOut into shared memory to improve 
*      memory coalescing and lower latency.
***********************************************************************************/

__global__
void convBackwardInputKernel(const float* gradConvOut, const float* w,
                             float* gradIn, int batchSize)
{
    // Each block processes one sample.
    int b = blockIdx.x;
    if(b >= batchSize) return; 

    __shared__ float sgrad[NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS];
    int totalConv = NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS;
    
    // Each thread in the block loads a portion of gradConvOut into shared memory.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    for (int i = tid; i < totalConv; i += blockSize) {
        sgrad[i] = gradConvOut[b * totalConv + i];
    }
    __syncthreads();
    
    // Map the block's 2D threads to the input image dimensions.
    int c = threadIdx.x;  // input column
    int r = threadIdx.y;  // input row
    if(r < IMAGE_ROWS && c < IMAGE_COLS) {
        float sumVal = 0.0f;
        // Instead of three nested loops over filters and kernel offsets,
        // use a single loop over totalKernel iterations.
        int totalKernel = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
        for (int i = 0; i < totalKernel; i++) {
            // Compute filter index f and kernel offsets kr, kc.
            int f = i / (FILTER_SIZE * FILTER_SIZE);
            int rem = i % (FILTER_SIZE * FILTER_SIZE);
            int kr = rem / FILTER_SIZE;
            int kc = rem % FILTER_SIZE;
            int orow = r - kr;
            int ocol = c - kc;
            // Check bounds for the output index.
            if (orow >= 0 && orow < CONV_OUT_ROWS &&
                ocol >= 0 && ocol < CONV_OUT_COLS) {
                int idxConv = f * (CONV_OUT_ROWS * CONV_OUT_COLS) + orow * CONV_OUT_COLS + ocol;
                float gOut = sgrad[idxConv];
                int idxW = f * (FILTER_SIZE * FILTER_SIZE) + kr * FILTER_SIZE + kc;
                float wVal = w[idxW];
                sumVal += gOut * wVal;
            }
        }
        gradIn[b * (IMAGE_ROWS * IMAGE_COLS) + r * IMAGE_COLS + c] = sumVal;
    }
}



/***********************************************************************************
* SGD Update Kernel
*
* Optimizations Employed:
* 1. Baseline Implementation:
*    - Performs a straightforward element-wise update: param -= lr * grad.
*
* 2. Minimal Overhead:
*    - No additional optimizations were applied given the simplicity of the operation.
***********************************************************************************/

__global__
void sgdUpdateKernel(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        param[i] -= lr * grad[i];
}



/***********************************************************************************
* Main Function
*
* Overview:
*  - Implements triple buffering to overlap host-to-device data transfers with 
*    kernel execution, maximizing GPU utilization.
*
* Training Phase:
*  1. Forward Pass:
*     - Executes optimized convolution, pooling, and fully connected kernels.
*  2. Backward Pass:
*     - Computes gradients using optimized backward kernels.
*  3. Parameter Update:
*     - Updates model parameters via SGD.
*
* Testing Phase:
*  - Evaluates the trained model on unseen test data to compute accuracy.
*
* Triple Buffering:
*  - Utilizes three buffers with CUDA streams, allowing concurrent data transfers,
*    computation, and result copying to minimize idle time.
***********************************************************************************/

int main(){
    // ---------------------------------------------------------------------------
    // 1) Load data (host)
    // ---------------------------------------------------------------------------
    float* h_trainImages = (float*)malloc(TRAIN_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   h_trainLabels = (int*)malloc(TRAIN_IMAGES * sizeof(int));
    float* h_testImages  = (float*)malloc(TEST_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   h_testLabels  = (int*)malloc(TEST_IMAGES * sizeof(int));
 
    printf("Reading MNIST data...\n");
    readMNISTImages("train-images.idx3-ubyte", h_trainImages, TRAIN_IMAGES);
    readMNISTLabels("train-labels.idx1-ubyte", h_trainLabels, TRAIN_IMAGES);
    readMNISTImages("t10k-images.idx3-ubyte", h_testImages, TEST_IMAGES);
    readMNISTLabels("t10k-labels.idx1-ubyte", h_testLabels, TEST_IMAGES);
 
    // ---------------------------------------------------------------------------
    // 2) Allocate device memory for parameters/activations and triple-buffered batch data
    // ---------------------------------------------------------------------------
    size_t imageBytesPerBatch = BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float);
    float* d_trainImages[NUM_BUFFERS];
    int*   d_labels[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++){
        CHECK_CUDA(cudaMalloc(&d_trainImages[i], imageBytesPerBatch));
        CHECK_CUDA(cudaMalloc(&d_labels[i], BATCH_SIZE * sizeof(int)));
    }
    int convW_size = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
    float *d_convW, *d_convB;
    CHECK_CUDA(cudaMalloc(&d_convW, convW_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_convB, NUM_FILTERS * sizeof(float)));
    float *d_convOut;
    CHECK_CUDA(cudaMalloc(&d_convOut, BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS * sizeof(float)));
    float *d_poolOut;
    CHECK_CUDA(cudaMalloc(&d_poolOut, BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    int *d_poolIdx;
    CHECK_CUDA(cudaMalloc(&d_poolIdx, BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(int)));
    float *d_flat;
    CHECK_CUDA(cudaMalloc(&d_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    float *d_fcW, *d_fcB;
    CHECK_CUDA(cudaMalloc(&d_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fcB, NUM_CLASSES * sizeof(float)));
    float *d_fcOut;
    CHECK_CUDA(cudaMalloc(&d_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    float *d_prob;
    CHECK_CUDA(cudaMalloc(&d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    float *d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    float *d_grad_fcOut;
    CHECK_CUDA(cudaMalloc(&d_grad_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    float *d_grad_fcW, *d_grad_fcB;
    CHECK_CUDA(cudaMalloc(&d_grad_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_fcB, NUM_CLASSES * sizeof(float)));
    float *d_grad_flat;
    CHECK_CUDA(cudaMalloc(&d_grad_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    float *d_grad_convOut;
    CHECK_CUDA(cudaMalloc(&d_grad_convOut, BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS * sizeof(float)));
    float *d_grad_convW, *d_grad_convB;
    CHECK_CUDA(cudaMalloc(&d_grad_convW, convW_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_convB, NUM_FILTERS * sizeof(float)));
    float *d_grad_in;
    CHECK_CUDA(cudaMalloc(&d_grad_in, BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float)));
 
    // ---------------------------------------------------------------------------
    // 3) Initialize weights on host -> copy to device
    // ---------------------------------------------------------------------------
    srand(123);
    auto randf = [](){ return 0.01f * ((float)rand() / RAND_MAX - 0.5f); };
    float* h_convW = (float*)malloc(convW_size * sizeof(float));
    float* h_convB_ = (float*)malloc(NUM_FILTERS * sizeof(float));
    for (int i = 0; i < convW_size; i++)
        h_convW[i] = randf();
    for (int i = 0; i < NUM_FILTERS; i++)
        h_convB_[i] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_convW, h_convW, convW_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_convB, h_convB_, NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
    free(h_convW); free(h_convB_);
    float* h_fcW = (float*)malloc(FLATTEN_SIZE * NUM_CLASSES * sizeof(float));
    float* h_fcB = (float*)malloc(NUM_CLASSES * sizeof(float));
    for (int i = 0; i < FLATTEN_SIZE * NUM_CLASSES; i++)
        h_fcW[i] = randf();
    for (int i = 0; i < NUM_CLASSES; i++)
        h_fcB[i] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_fcW, h_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fcB, h_fcB, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fcW); free(h_fcB);
 
    // ---------------------------------------------------------------------------
    // 4) Create streams and pinned host buffers (triple buffering)
    // ---------------------------------------------------------------------------
    cudaStream_t stream[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++){
        CHECK_CUDA(cudaStreamCreate(&stream[i]));
    }
    float* h_pinnedImages[NUM_BUFFERS];
    int*   h_pinnedLabels[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++){
        CHECK_CUDA(cudaMallocHost((void**)&h_pinnedImages[i], imageBytesPerBatch));
        CHECK_CUDA(cudaMallocHost((void**)&h_pinnedLabels[i], BATCH_SIZE * sizeof(int)));
    }
 
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    CHECK_CUDA(cudaEventRecord(startEvent, 0));
 
    // ---------------------------------------------------------------------------
    // 5) Training with triple-buffering
    // ---------------------------------------------------------------------------
    int nBatches = TRAIN_IMAGES / BATCH_SIZE;
    printf("Starting training for %d epochs...\n", EPOCHS);
    for (int e = 0; e < EPOCHS; e++){
        double epochLoss = 0.0;
        int warmBatches = (nBatches < NUM_BUFFERS) ? nBatches : NUM_BUFFERS;
        for (int i = 0; i < warmBatches; i++){
            int dataOffset = i * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
            memcpy(h_pinnedImages[i], &h_trainImages[dataOffset], imageBytesPerBatch);
            memcpy(h_pinnedLabels[i], &h_trainLabels[i * BATCH_SIZE], BATCH_SIZE * sizeof(int));
            CHECK_CUDA(cudaMemcpyAsync(d_trainImages[i], h_pinnedImages[i],
                                       imageBytesPerBatch, cudaMemcpyHostToDevice,
                                       stream[i]));
            CHECK_CUDA(cudaMemcpyAsync(d_labels[i], h_pinnedLabels[i],
                                       BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice,
                                       stream[i]));
        }
        int b = 0;
        for(; b < nBatches; b++){
            int curIdx = b % NUM_BUFFERS;
            int nextIdx = (b + 1) % NUM_BUFFERS;
            CHECK_CUDA(cudaStreamSynchronize(stream[curIdx]));
            // Update constant memory with current conv weights
            CHECK_CUDA(cudaMemcpyToSymbol(constW, d_convW, convW_size * sizeof(float)));
            // Convolution (fused with ReLU)
            {
                dim3 blockDim(16, 16, 1);
                dim3 gridDim((CONV_OUT_COLS + 16 - 1) / 16,
                             (CONV_OUT_ROWS + 16 - 1) / 16,
                             BATCH_SIZE * NUM_FILTERS);
                size_t sharedMemSize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float);
                fusedConvReluKernel<<<gridDim, blockDim, sharedMemSize, stream[curIdx]>>>(
                    d_trainImages[curIdx], d_convW, d_convB, d_convOut, BATCH_SIZE);
            }
            // Fused Max Pooling + Flattening (forward)
            {
                int totalFlatten = BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
                int grid = (totalFlatten + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fusedMaxPoolFlattenKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(
                    d_convOut, d_flat, BATCH_SIZE);
            }
            // Fully Connected forward (tiled GEMM)
            {
                dim3 fcGrid((NUM_CLASSES + 16 - 1)/16, (BATCH_SIZE + 16 - 1)/16);
                dim3 fcBlock(16, 16, 1);
                int sharedMemBytes = 2 * (16*16) * sizeof(float);
                fcForwardKernel<<<fcGrid, fcBlock, sharedMemBytes, stream[curIdx]>>>(d_flat, d_fcW, d_fcB, d_fcOut, BATCH_SIZE);
            }
            // Softmax and loss forward (optimized)
            {
                int grid = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
                softmaxCrossEntropyForwardKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_fcOut, d_labels[curIdx], d_loss, d_prob, BATCH_SIZE);
            }
            // Backward for FC
            {
                int grid = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
                softmaxCrossEntropyBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_grad_fcOut, d_prob, d_labels[curIdx], BATCH_SIZE);
            }
            {
                CHECK_CUDA(cudaMemsetAsync(d_grad_fcW, 0, FLATTEN_SIZE * NUM_CLASSES * sizeof(float), stream[curIdx]));
                CHECK_CUDA(cudaMemsetAsync(d_grad_fcB, 0, NUM_CLASSES * sizeof(float), stream[curIdx]));
                int totalParams = FLATTEN_SIZE * NUM_CLASSES + NUM_CLASSES;
                int gridX = (totalParams + 16 - 1) / 16;  // each block processes 16 parameters
                fcBackwardGradParamKernel<<< gridX, dim3(32, 16), 0, stream[curIdx] >>>(d_grad_fcOut, d_flat, d_grad_fcW, d_grad_fcB, BATCH_SIZE);


            }
            {
                int total = BATCH_SIZE * FLATTEN_SIZE;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fcBackwardGradInKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_grad_fcOut, d_fcW, d_grad_flat, BATCH_SIZE);
            }
                // Fused max pooling backward: clear d_grad_convOut first
                CHECK_CUDA(cudaMemsetAsync(d_grad_convOut, 0, BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS * sizeof(float), stream[curIdx]));
            {
                int totalFlatten = BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
                int grid = (totalFlatten + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fusedMaxPoolBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_convOut, d_grad_flat, d_grad_convOut, BATCH_SIZE);
            }
            // Backward for conv layer
            {
                int convSize = BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS;
                int grid = (convSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                reluBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_grad_convOut, d_convOut, d_grad_convOut, convSize);
            }
            {
                CHECK_CUDA(cudaMemsetAsync(d_grad_convW, 0, convW_size * sizeof(float), stream[curIdx]));
                CHECK_CUDA(cudaMemsetAsync(d_grad_convB, 0, NUM_FILTERS * sizeof(float), stream[curIdx]));
                int totalParams = FLATTEN_SIZE * NUM_CLASSES + NUM_CLASSES;
                convBackwardWeightKernel<<< totalParams, 32, 0, stream[curIdx] >>>(d_trainImages[curIdx], d_grad_convOut, d_grad_convW, d_grad_convB, BATCH_SIZE);

            }
            {
                dim3 blockDim(IMAGE_COLS, IMAGE_ROWS);
                convBackwardInputKernel<<<BATCH_SIZE, blockDim, 0, stream[curIdx]>>>(d_grad_convOut, d_convW, d_grad_in, BATCH_SIZE);

            }
            {
                int total = convW_size;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sgdUpdateKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_convW, d_grad_convW, LEARNING_RATE, total);
                total = NUM_FILTERS;
                grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sgdUpdateKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_convB, d_grad_convB, LEARNING_RATE, total);
            }
            {
                int total = FLATTEN_SIZE * NUM_CLASSES;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sgdUpdateKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_fcW, d_grad_fcW, LEARNING_RATE, total);
                total = NUM_CLASSES;
                grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sgdUpdateKernel<<<grid, BLOCK_SIZE, 0, stream[curIdx]>>>(d_fcB, d_grad_fcB, LEARNING_RATE, total);
            }
 
            if(b + 1 < nBatches){
                int nextOffset = (b + 1) * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
                memcpy(h_pinnedImages[nextIdx], &h_trainImages[nextOffset], imageBytesPerBatch);
                memcpy(h_pinnedLabels[nextIdx], &h_trainLabels[(b + 1) * BATCH_SIZE], BATCH_SIZE * sizeof(int));
                CHECK_CUDA(cudaMemcpyAsync(d_trainImages[nextIdx], h_pinnedImages[nextIdx],
                                           imageBytesPerBatch, cudaMemcpyHostToDevice,
                                           stream[nextIdx]));
                CHECK_CUDA(cudaMemcpyAsync(d_labels[nextIdx], h_pinnedLabels[nextIdx],
                                           BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice,
                                           stream[nextIdx]));
            }
 
            CHECK_CUDA(cudaStreamSynchronize(stream[curIdx]));
            float h_loss[BATCH_SIZE];
            CHECK_CUDA(cudaMemcpy(h_loss, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float batchLoss = 0.f;
            for (int i = 0; i < BATCH_SIZE; i++){
                batchLoss += h_loss[i];
            }
            epochLoss += batchLoss / BATCH_SIZE;
        }
        epochLoss /= nBatches;
        printf("Epoch [%d/%d], avg loss=%.6f\n", e + 1, EPOCHS, epochLoss);
    }
 
    for (int i = 0; i < NUM_BUFFERS; i++){
        CHECK_CUDA(cudaStreamSynchronize(stream[i]));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    float elapsedTime = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    // ---------------------------------------------------------------------------------
    // 6) Testing Phase: Evaluate trained model on unseen MNIST test data
    // ---------------------------------------------------------------------------------
    {
        dim3 fcGrid((NUM_CLASSES + 16 - 1)/16, (BATCH_SIZE + 16 - 1)/16);
        dim3 fcBlock(16, 16, 1);
        int fcSharedBytes = 2 * (16*16) * sizeof(float);
 
        int correct = 0;
        int testBatches = TEST_IMAGES / BATCH_SIZE;
        for (int b = 0; b < testBatches; b++){
            CHECK_CUDA(cudaMemcpy(d_trainImages[0],
                       &h_testImages[b * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS)],
                       imageBytesPerBatch, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_labels[0],
                       &h_testLabels[b * BATCH_SIZE],
                       BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            {
                CHECK_CUDA(cudaMemcpyToSymbol(constW, d_convW, convW_size * sizeof(float)));
                dim3 blockDim(16, 16, 1);
                dim3 gridDim((CONV_OUT_COLS + 16 - 1) / 16,
                             (CONV_OUT_ROWS + 16 - 1) / 16,
                             BATCH_SIZE * NUM_FILTERS);
                size_t sharedMemSize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float);
                fusedConvReluKernel<<<gridDim, blockDim, sharedMemSize>>>(d_trainImages[0], d_convW, d_convB, d_convOut, BATCH_SIZE);
                int totalFlatten = BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
                int grid = (totalFlatten + BLOCK_SIZE - 1) / BLOCK_SIZE;
                fusedMaxPoolFlattenKernel<<<grid, BLOCK_SIZE>>>(d_convOut, d_flat, BATCH_SIZE);
                fcForwardKernel<<<fcGrid, fcBlock, fcSharedBytes>>>(d_flat, d_fcW, d_fcB, d_fcOut, BATCH_SIZE);
                int gridB = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
                softmaxCrossEntropyForwardKernel<<<gridB, BLOCK_SIZE>>>(d_fcOut, d_labels[0], d_loss, d_prob, BATCH_SIZE);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            float* h_prob = (float*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
            CHECK_CUDA(cudaMemcpy(h_prob, d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < BATCH_SIZE; i++){
                float maxp = -1.f;
                int pred = -1;
                for (int c = 0; c < NUM_CLASSES; c++){
                    float p = h_prob[i * NUM_CLASSES + c];
                    if(p > maxp){
                        maxp = p;
                        pred = c;
                    }
                }
                if(pred == h_testLabels[b * BATCH_SIZE + i])
                    correct++;
            }
            free(h_prob);
        }
        float accuracy = (float)correct / (testBatches * BATCH_SIZE);
        printf("Test accuracy = %.2f%%\n", accuracy * 100.f);
    }
 
    // ---------------------------------------------------------------------------------
    // Cleanup: Free memory 
    // ---------------------------------------------------------------------------------
    free(h_trainImages);
    free(h_trainLabels);
    free(h_testImages);
    free(h_testLabels);
    for (int i = 0; i < NUM_BUFFERS; i++){
        CHECK_CUDA(cudaFree(d_trainImages[i]));
        CHECK_CUDA(cudaFree(d_labels[i]));
        CHECK_CUDA(cudaFreeHost(h_pinnedImages[i]));
        CHECK_CUDA(cudaFreeHost(h_pinnedLabels[i]));
        CHECK_CUDA(cudaStreamDestroy(stream[i]));
    }
    CHECK_CUDA(cudaFree(d_convW));
    CHECK_CUDA(cudaFree(d_convB));
    CHECK_CUDA(cudaFree(d_convOut));
    CHECK_CUDA(cudaFree(d_poolOut));
    CHECK_CUDA(cudaFree(d_poolIdx));
    CHECK_CUDA(cudaFree(d_flat));
    CHECK_CUDA(cudaFree(d_fcW));
    CHECK_CUDA(cudaFree(d_fcB));
    CHECK_CUDA(cudaFree(d_fcOut));
    CHECK_CUDA(cudaFree(d_prob));
    CHECK_CUDA(cudaFree(d_loss));
    CHECK_CUDA(cudaFree(d_grad_fcOut));
    CHECK_CUDA(cudaFree(d_grad_fcW));
    CHECK_CUDA(cudaFree(d_grad_fcB));
    CHECK_CUDA(cudaFree(d_grad_flat));
    CHECK_CUDA(cudaFree(d_grad_convOut));
    CHECK_CUDA(cudaFree(d_grad_convW));
    CHECK_CUDA(cudaFree(d_grad_convB));
    CHECK_CUDA(cudaFree(d_grad_in));
 
    printf("Done.\n");
    printf("Total elapsed time (ms): %f\n", elapsedTime);
    return 0;
}
