/*****************************************************************************
 * File: cnn_baseline.cu
 *
 * A from-scratch CNN on MNIST with no advanced CUDA optimizations.
 *
 * Architecture:
 *  - Convolution layer 
 *  - ReLU
 *  - 2x2 MaxPool
 *  - Fully Connected 
 *  - Softmax + Cross Entropy
 *
 * Stochastic Gradient Descent is used to update weights.
 *
 * Expected to achieve >90% accuracy (with enough epochs).
 *
 * Compile: nvcc cnn_baseline.cu -o cnn_baseline or via Makefile 
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
// Output from 5x5 kernel, stride=1, no padding => 24x24
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

// ---------------------------------------------------------------------------------
// CUDA error checking 
// ---------------------------------------------------------------------------------
#define DEBUG
inline void checkCudaError(const char *msg) {
#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}


#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %d: %s\n", __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)


// ---------------------------------------------------------------------------------
// MNIST Data Loading Functions
// ---------------------------------------------------------------------------------
void readMNISTImages(const char* fileName, float* data, int numImages) {
    // Each image 28x28 = 784 bytes
    // 16-byte header
    FILE* f = fopen(fileName, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fileName);
        exit(1);
    }
    fseek(f, 16, SEEK_SET); 
    for(int i = 0; i < numImages; i++) {
        for(int p = 0; p < IMAGE_ROWS * IMAGE_COLS; p++) {
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
* Convolution Forward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Straightforward Computation:
*    - Iterates over the filter dimensions using two nested loops to perform the 
*      convolution, with each thread computing one output element.
*
* 2. Global Memory Access:
*    - Directly accesses input and weight data from global memory without the use 
*      of shared or constant memory for caching, resulting in redundant memory 
*      accesses.
*
* 3. No Fusion or Tiling:
*    - Does not employ tiling or kernel fusion (e.g., fusing ReLU), keeping the 
*      implementation simple but less efficient.
*
* 4. Simple Index Mapping:
*    - Uses modular arithmetic to compute the corresponding batch, filter, and 
*      spatial indices for each output element.
***********************************************************************************/

__global__
void convForwardKernel(const float* in, const float* w, const float* b,
                       float* out, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS;
    if (index >= total) return;

    int batchIdx = index / (NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS);
    int rem      = index % (NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS);
    int f        = rem / (CONV_OUT_ROWS * CONV_OUT_COLS);
    int rem2     = rem % (CONV_OUT_ROWS * CONV_OUT_COLS);
    int outRow   = rem2 / CONV_OUT_COLS;
    int outCol   = rem2 % CONV_OUT_COLS;

    float val = 0.0f;
    for(int kr = 0; kr < FILTER_SIZE; kr++){
        for(int kc = 0; kc < FILTER_SIZE; kc++){
            int inRow = outRow + kr; 
            int inCol = outCol + kc;
            float inp = in[batchIdx * (IMAGE_ROWS * IMAGE_COLS)
                           + inRow * IMAGE_COLS
                           + inCol];
            float wgt = w[f * (FILTER_SIZE * FILTER_SIZE)
                          + kr * FILTER_SIZE
                          + kc];
            val += inp * wgt;
        }
    }
    val += b[f];
    out[index] = val;
}



/***********************************************************************************
* ReLU Forward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Simple Element-wise Operation:
*    - Each thread examines its corresponding element in global memory and sets it
*      to zero if it is negative.
*
* 2. Lack of Kernel Fusion:
*    - This kernel is implemented separately. In a more efficient design, it could be
*      fused with the convolution forward kernel to reduce memory traffic and kernel
*      launch overhead.
*
* 3. Direct Global Memory Access:
*    - Operates directly on global memory without using shared memory caching or
*      vectorized memory accesses.
***********************************************************************************/

__global__
void reluForwardKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] < 0.0f)
            data[idx] = 0.0f;
    }
}



/***********************************************************************************
* Max Pooling Forward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Sequential Scanning:
*    - Each thread iterates over the POOL_SIZE×POOL_SIZE window with nested loops
*      to determine the maximum value and its index.
*
* 2. Direct Global Memory Access:
*    - All data is read directly from global memory without caching in shared memory,
*      resulting in redundant memory accesses.
*
* 3. Implicit Flattening:
*    - Index arithmetic is used to map multi-dimensional data (batch, filters, spatial)
*      to a flat index, without any fusion or additional optimization.
*
* 4. No Kernel Fusion:
*    - The pooling and index tracking are implemented separately, leaving potential
*      optimizations (such as fusing with subsequent flattening) unexplored.
***********************************************************************************/

__global__
void maxPoolForwardKernel(const float* in, float* out, int* maxIdx, int batchSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if (index >= total) return;

    int b = index / (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem = index % (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int f = rem / (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem2 = rem % (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rOut = rem2 / POOL_OUT_COLS;
    int cOut = rem2 % POOL_OUT_COLS;

    int rIn = rOut * POOL_SIZE;
    int cIn = cOut * POOL_SIZE;
    float maxVal = -1e30f;
    int maxPos = 0;
    for(int i=0; i<POOL_SIZE; i++){
        for(int j=0; j<POOL_SIZE; j++){
            int inR = rIn + i;
            int inC = cIn + j;
            int inIdx = b*(NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS)
                       + f*(CONV_OUT_ROWS*CONV_OUT_COLS)
                       + inR*CONV_OUT_COLS
                       + inC;
            float v = in[inIdx];
            if(v > maxVal){
                maxVal = v;
                maxPos = inIdx;
            }
        }
    }
    out[index] = maxVal;
    maxIdx[index] = maxPos;
}



/***********************************************************************************
* Flatten Pooling Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Direct Flattening:
*    - Converts the multi-dimensional pooled output [B, NUM_FILTERS, POOL_OUT_ROWS, POOL_OUT_COLS]
*      into a 2D flattened matrix [B, FLATTEN_SIZE] using straightforward index arithmetic.
*
* 2. Global Memory Access:
*    - Each thread directly reads from and writes to global memory without caching,
*      keeping the implementation simple.
*
* 3. Minimal Overhead:
*    - No advanced optimizations (e.g., shared memory or loop unrolling) are employed.
***********************************************************************************/

__global__
void flattenPoolKernel(const float* in, float* out, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if(idx >= total) return;

    int b = idx / (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem = idx % (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int f = rem / (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem2 = rem % (POOL_OUT_ROWS * POOL_OUT_COLS);
    int r = rem2 / POOL_OUT_COLS;
    int c = rem2 % POOL_OUT_COLS;

    int flatIdx = f*(POOL_OUT_ROWS*POOL_OUT_COLS) + r*(POOL_OUT_COLS) + c;
    out[b * FLATTEN_SIZE + flatIdx] = in[idx];
}



/***********************************************************************************
* Fully Connected Forward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Simple Dot Product:
*    - Each thread computes one output element by iterating over the entire input
*      vector (of size FLATTEN_SIZE) and performing a dot product with the corresponding
*      weight column.
*
* 2. Direct Global Memory Access:
*    - Reads from global memory for the input, weights, and bias without caching,
*      leading to high memory traffic and latency.
*
* 3. No Tiling or Fusion:
*    - The kernel is implemented in a straightforward loop without any shared memory
*      tiling, loop unrolling, or fusion techniques.
*
* Note:
*    - A GEMM-style implementation using shared memory tiling would be significantly
*      faster. By loading sub-tiles of the input and weight matrices into shared memory,
*      a GEMM approach reduces redundant global memory accesses, increases data reuse,
*      and minimizes latency, thereby improving overall throughput.
***********************************************************************************/

__global__
void fcForwardKernel(const float* in, const float* w, const float* b,
                     float* out, int batchSize)
{
    // out shape = [batchSize, NUM_CLASSES]
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_CLASSES;
    if(index >= total) return;

    int bIdx = index / NUM_CLASSES;
    int c    = index % NUM_CLASSES;

    float sumVal = 0.0f;
    for(int k=0; k<FLATTEN_SIZE; k++){
        sumVal += in[bIdx*FLATTEN_SIZE + k] * w[k*NUM_CLASSES + c];
    }
    sumVal += b[c];
    out[index] = sumVal;
}



/***********************************************************************************
* Softmax and Cross-Entropy Forward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Per-Sample Processing:
*    - Each thread processes one sample, computing the maximum logit for numerical
*      stability, the sum of exponentials, and the resulting probabilities and loss.
*
* 2. Multiple Passes Over Classes:
*    - The kernel iterates over the class dimension three separate times: first to
*      determine the maximum logit, then to compute the sum of exponentials, and finally
*      to calculate the normalized probabilities and loss.
*
* 3. Redundant Exponential Computation:
*    - The expf function is invoked repeatedly in separate loops, leading to redundant
*      calculations that could be optimized.
*
* 4. Simplicity:
*    - Designed for clarity, this baseline version avoids advanced optimizations such as
*      caching intermediate results or fusing loops.
***********************************************************************************/

__global__
void softmaxCrossEntropyForwardKernel(const float* logits, const int* labels,
                                      float* outLoss, float* outProb,
                                      int batchSize)
{
    // 1 thread per sample
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batchSize) return;

    // find max for stability
    float maxLogit = -1e30f;
    for(int c=0; c<NUM_CLASSES; c++){
        float val = logits[i * NUM_CLASSES + c];
        if(val > maxLogit) maxLogit = val;
    }
    // sum exp
    float sumExp = 0.0f;
    for(int c=0; c<NUM_CLASSES; c++){
        float ex = expf(logits[i*NUM_CLASSES + c] - maxLogit);
        sumExp += ex;
    }
    int lbl = labels[i];
    float lossVal = 0.0f;
    for(int c=0; c<NUM_CLASSES; c++){
        float ex = expf(logits[i*NUM_CLASSES + c] - maxLogit);
        float prob = ex / sumExp;
        outProb[i*NUM_CLASSES + c] = prob;
        if(c == lbl){
            lossVal = -logf(prob + 1e-10f);
        }
    }
    outLoss[i] = lossVal;
}



// ---------------------------------------------------------------------------------
// Backward Kernels
// ---------------------------------------------------------------------------------

/***********************************************************************************
* Softmax and Cross-Entropy Backward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Element-wise Computation:
*    - Each thread computes the gradient for one class of one sample by subtracting
*      the one-hot label from the corresponding probability.
*
* 2. Direct Global Memory Access:
*    - Reads probabilities and labels directly from global memory with no caching,
*      keeping the implementation simple.
*
* 3. Minimal Overhead:
*    - The kernel uses a straightforward approach without loop unrolling or warp-level
*      optimizations, emphasizing clarity over performance.
***********************************************************************************/

__global__
void softmaxCrossEntropyBackwardKernel(float* gradLogits, const float* prob,
                                       const int* labels, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * NUM_CLASSES) return;

    int sampleIdx = idx / NUM_CLASSES;
    int c = idx % NUM_CLASSES;
    int lbl = labels[sampleIdx];

    float p = prob[idx];
    float y = (c == lbl) ? 1.0f : 0.0f;
    gradLogits[idx] = p - y;
}



/***********************************************************************************
* Fully Connected Backward Gradient Parameter Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Straightforward Accumulation:
*    - Each thread computes one gradient parameter (weight or bias) by iterating 
*      over the entire batch dimension and summing the contributions.
*
* 2. Direct Global Memory Access:
*    - No caching or shared memory is used; data is read directly from global memory 
*      in each iteration.
*
* 3. Simple Looping:
*    - Uses a basic for-loop over the batch dimension without any parallel reduction
*      techniques.
*
* 4. Separate Handling:
*    - Processes weight gradients and bias gradients separately in a single kernel, 
*      without advanced fusion optimizations.
***********************************************************************************/

__global__
void fcBackwardGradParamKernel(const float* gradOut, const float* in,
                               float* gradW, float* gradB,
                               int batchSize)
{
    // Each thread handles one element in [FLATTEN_SIZE*NUM_CLASSES] for gradW
    // or up to NUM_CLASSES for gradB. We'll handle them separately:
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalW = FLATTEN_SIZE * NUM_CLASSES;
    int totalParams = totalW + NUM_CLASSES;

    if(idx >= totalParams) return;

    if(idx < totalW) {
        // gradW
        int k = idx / NUM_CLASSES;   // which input index
        int c = idx % NUM_CLASSES;   // which class
        float sumVal = 0.0f;
        for(int b=0; b<batchSize; b++){
            float go = gradOut[b*NUM_CLASSES + c];
            float inp = in[b*FLATTEN_SIZE + k];
            sumVal += inp * go;
        }
        gradW[idx] = sumVal;
    } else {
        // gradB
        int c = idx - totalW;
        float sumVal = 0.0f;
        for(int b=0; b<batchSize; b++){
            sumVal += gradOut[b*NUM_CLASSES + c];
        }
        gradB[c] = sumVal;
    }
}



/***********************************************************************************
* Fully Connected Backward Gradient Input Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Straightforward Dot Product:
*    - Each thread computes one element of gradIn by iterating over all NUM_CLASSES,
*      multiplying corresponding gradOut and weight values, and accumulating the result.
*
* 2. Direct Global Memory Access:
*    - Data is read directly from global memory without using caching mechanisms like 
*      shared memory or constant memory.
*
* 3. Simple Looping:
*    - A basic loop over the NUM_CLASSES dimension is used without any loop unrolling or
*      tiling optimizations.
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
    for(int c=0; c<NUM_CLASSES; c++){
        float go = gradOut[b*NUM_CLASSES + c];
        float wVal = w[k*NUM_CLASSES + c];
        sumVal += go * wVal;
    }
    gradIn[idx] = sumVal;
}



/***********************************************************************************
* Unflatten Pooling Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Direct Index Mapping:
*    - Converts the flattened pooling output [B, FLATTEN_SIZE] back into its original
*      4D shape [B, NUM_FILTERS, POOL_OUT_ROWS, POOL_OUT_COLS] using simple modular
*      arithmetic.
*
* 2. Global Memory Operations:
*    - Reads from the flattened gradient (gradFlat) and writes directly to the unflattened
*      output (gradPoolOut) without any caching or additional optimizations.
*
* Note:
*    - In the optimized version, this kernel is fused with the maxPoolBackwardKernel, 
*      thereby reducing redundant global memory accesses and kernel launch overhead.
***********************************************************************************/

__global__
void unflattenPoolKernel(const float* gradFlat, float* gradPoolOut, int batchSize)
{
    // Inverse of flatten: [B, FLATTEN_SIZE] -> [B, NUM_FILTERS, 12, 12]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if(idx >= total) return;

    int b = idx / (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem = idx % (NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS);
    int f = rem / (POOL_OUT_ROWS * POOL_OUT_COLS);
    int rem2 = rem % (POOL_OUT_ROWS * POOL_OUT_COLS);
    int r = rem2 / POOL_OUT_COLS;
    int c = rem2 % POOL_OUT_COLS;

    int flatIdx = f*(POOL_OUT_ROWS*POOL_OUT_COLS) + r*(POOL_OUT_COLS) + c;
    gradPoolOut[idx] = gradFlat[b * FLATTEN_SIZE + flatIdx];
}



/***********************************************************************************
* Max Pooling Backward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Atomic Accumulation:
*    - Uses atomicAdd to accumulate gradient contributions at the selected max index,
*      which can cause contention and is less efficient.
*
* 2. Direct Global Memory Access:
*    - Reads the gradient output and max index directly from global memory without caching.
*
* 3. Straightforward Approach:
*    - Implements the backward pooling operation in a simple manner for clarity, without
*      advanced optimizations.
***********************************************************************************/

__global__
void maxPoolBackwardKernel(const float* gradOut, float* gradIn,
                           const int* maxIdx, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS;
    if(idx >= total) return;

    float val = gradOut[idx];
    int inPos = maxIdx[idx];
    atomicAdd(&gradIn[inPos], val);
}



/***********************************************************************************
* ReLU Backward Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Element-wise Conditional Update:
*    - Each thread checks if x[i] is positive and, if so, passes gradOut[i] to gradIn[i];
*      otherwise, it sets gradIn[i] to 0.
*
* 2. Direct Global Memory Access:
*    - Reads and writes occur directly to global memory without caching or vectorized
*      operations, keeping the implementation simple.
***********************************************************************************/

__global__
void reluBackwardKernel(const float* gradOut, const float* x,
                        float* gradIn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        gradIn[i] = (x[i] > 0.0f) ? gradOut[i] : 0.0f;
    }
}



/***********************************************************************************
* Convolution Backward Weight Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Triple Nested Loop:
*    - For each weight (or bias) parameter, the kernel iterates over the entire batch 
*      and spatial dimensions using three nested loops, which results in significant 
*      redundant computation.
*
* 2. Direct Global Memory Access:
*    - Accesses input and gradConvOut data directly from global memory without any 
*      caching mechanisms, leading to increased memory latency.
*
* 3. Separate Handling:
*    - Processes weight gradients and bias gradients in distinct code paths using 
*      basic index arithmetic.
*
* 4. Lack of Advanced Optimizations:
*    - This implementation does not exploit techniques such as parallel (warp-level)
*      reduction, shared memory tiling, or loop unrolling, all of which could further 
*      reduce redundant computations and improve performance.
***********************************************************************************/

__global__
void convBackwardWeightKernel(const float* in, const float* gradConvOut,
                              float* gradW, float* gradB, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wCount = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE; 
    int totalParams = wCount + NUM_FILTERS;
    if(idx >= totalParams) return;

    if(idx < wCount){
        // weight
        int f = idx / (FILTER_SIZE * FILTER_SIZE);
        int rem = idx % (FILTER_SIZE * FILTER_SIZE);
        int kr = rem / FILTER_SIZE;
        int kc = rem % FILTER_SIZE;

        float sumVal = 0.0f;
        for(int b=0; b<batchSize; b++){
            for(int orow=0; orow<CONV_OUT_ROWS; orow++){
                for(int ocol=0; ocol<CONV_OUT_COLS; ocol++){
                    float g = gradConvOut[b*(NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS)
                                          + f*(CONV_OUT_ROWS*CONV_OUT_COLS)
                                          + orow*CONV_OUT_COLS
                                          + ocol];
                    float inp = in[b*(IMAGE_ROWS*IMAGE_COLS)
                                   + (orow+kr)*IMAGE_COLS
                                   + (ocol+kc)];
                    sumVal += inp*g;
                }
            }
        }
        gradW[idx] = sumVal;
    } else {
        // bias
        int f = idx - wCount;
        float sumVal = 0.0f;
        for(int b=0; b<batchSize; b++){
            for(int orow=0; orow<CONV_OUT_ROWS; orow++){
                for(int ocol=0; ocol<CONV_OUT_COLS; ocol++){
                    float g = gradConvOut[b*(NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS)
                                          + f*(CONV_OUT_ROWS*CONV_OUT_COLS)
                                          + orow*CONV_OUT_COLS
                                          + ocol];
                    sumVal += g;
                }
            }
        }
        gradB[f] = sumVal;
    }
}



/***********************************************************************************
* Convolution Backward Input Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Triple Nested Loops:
*    - Iterates over each filter and kernel offset using three nested loops,
*      with bounds checks performed for every iteration.
*
* 2. Direct Global Memory Access:
*    - Reads gradConvOut and weight values directly from global memory for each
*      computation, resulting in redundant data accesses.
*
* 3. Lack of Advanced Optimizations:
*    - Does not employ shared memory tiling, loop fusion, or parallel reduction
*      techniques that could reduce memory latency and improve performance.
***********************************************************************************/

__global__
void convBackwardInputKernel(const float* gradConvOut, const float* w,
                             float* gradIn, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * IMAGE_ROWS * IMAGE_COLS;
    if(idx >= total) return;

    int b = idx / (IMAGE_ROWS * IMAGE_COLS);
    int rem = idx % (IMAGE_ROWS * IMAGE_COLS);
    int r = rem / IMAGE_COLS;
    int c = rem % IMAGE_COLS;

    float sumVal = 0.0f;
    for(int f=0; f<NUM_FILTERS; f++){
        for(int kr=0; kr<FILTER_SIZE; kr++){
            for(int kc=0; kc<FILTER_SIZE; kc++){
                int orow = r - kr;
                int ocol = c - kc;
                if(orow>=0 && orow<CONV_OUT_ROWS && ocol>=0 && ocol<CONV_OUT_COLS){
                    float gOut = gradConvOut[b*(NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS)
                                             + f*(CONV_OUT_ROWS*CONV_OUT_COLS)
                                             + orow*CONV_OUT_COLS
                                             + ocol];
                    float wVal = w[f*(FILTER_SIZE*FILTER_SIZE)
                                   + kr*FILTER_SIZE
                                   + kc];
                    sumVal += gOut * wVal;
                }
            }
        }
    }
    gradIn[idx] = sumVal;
}



/***********************************************************************************
* SGD Update Kernel (Baseline Implementation)
*
* Naive Implementation Details:
* 1. Element-wise Update:
*    - Each thread performs a simple update (param[i] -= lr * grad[i]) for one element.
*
* 2. Direct Global Memory Access:
*    - Reads and writes occur directly to global memory without any caching or vectorization,
*      leading to straightforward but suboptimal performance.
***********************************************************************************/

__global__
void sgdUpdateKernel(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        param[i] -= lr * grad[i];
    }
}



/***********************************************************************************
* Main Function (Baseline Implementation)
*
* Overview:
*  - Executes the CNN in a straightforward, sequential manner without overlapping
*    host-to-device data transfers and kernel execution.
*
* Training Phase:
*  1. Forward Pass:
*     - Launches each kernel (convolution, pooling, fully-connected, softmax) one 
*       after the other, with no concurrent data transfers.
*
*  2. Backward Pass:
*     - Computes gradients and updates parameters sequentially, waiting for each 
*       kernel to finish before proceeding.
*
* Testing Phase:
*  - Evaluates the trained model on unseen test data after the training completes.
*
* Lack of Overlapping:
*  - Unlike optimized implementations that use triple buffering and CUDA streams to 
*    overlap data transfers with computation, this baseline approach performs all 
*    operations sequentially, resulting in lower GPU utilization.
***********************************************************************************/

int main(){
    // ---------------------------------------------------------------------------
    // 1) Load data (host)
    // ---------------------------------------------------------------------------
    float* h_trainImages = (float*)malloc(TRAIN_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   h_trainLabels = (int*)  malloc(TRAIN_IMAGES * sizeof(int));
    float* h_testImages  = (float*)malloc(TEST_IMAGES  * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   h_testLabels  = (int*)  malloc(TEST_IMAGES  * sizeof(int));

    printf("Reading MNIST data...\n");
    readMNISTImages("train-images.idx3-ubyte", h_trainImages, TRAIN_IMAGES);
    readMNISTLabels("train-labels.idx1-ubyte", h_trainLabels, TRAIN_IMAGES);
    readMNISTImages("t10k-images.idx3-ubyte",  h_testImages,  TEST_IMAGES);
    readMNISTLabels("t10k-labels.idx1-ubyte",  h_testLabels,  TEST_IMAGES);

    // ---------------------------------------------------------------------------
    // 2) Allocate GPU memory for a batch + network
    // ---------------------------------------------------------------------------
    float* d_trainImages;
    int*   d_labels;
    size_t imageBytes = BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_trainImages, imageBytes));
    CHECK_CUDA(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));

    // Conv params
    int convW_size = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
    float *d_convW, *d_convB;
    CHECK_CUDA(cudaMalloc(&d_convW, convW_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_convB, NUM_FILTERS * sizeof(float)));

    // Conv output
    float *d_convOut;
    CHECK_CUDA(cudaMalloc(&d_convOut, BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS * sizeof(float)));

    // Pool output
    float *d_poolOut;
    CHECK_CUDA(cudaMalloc(&d_poolOut, BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    int *d_poolIdx;
    CHECK_CUDA(cudaMalloc(&d_poolIdx, BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(int)));

    // Flatten
    float *d_flat;
    CHECK_CUDA(cudaMalloc(&d_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));

    // FC params
    float *d_fcW, *d_fcB;
    CHECK_CUDA(cudaMalloc(&d_fcW, FLATTEN_SIZE*NUM_CLASSES*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fcB, NUM_CLASSES*sizeof(float)));

    // FC output
    float *d_fcOut;
    CHECK_CUDA(cudaMalloc(&d_fcOut, BATCH_SIZE*NUM_CLASSES*sizeof(float)));

    // Softmax prob + loss
    float *d_prob;
    CHECK_CUDA(cudaMalloc(&d_prob, BATCH_SIZE*NUM_CLASSES*sizeof(float)));
    float *d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, BATCH_SIZE*sizeof(float)));

    // Gradients
    float *d_grad_fcOut;
    CHECK_CUDA(cudaMalloc(&d_grad_fcOut, BATCH_SIZE*NUM_CLASSES*sizeof(float)));
    float *d_grad_fcW, *d_grad_fcB;
    CHECK_CUDA(cudaMalloc(&d_grad_fcW, FLATTEN_SIZE*NUM_CLASSES*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_fcB, NUM_CLASSES*sizeof(float)));
    float *d_grad_flat;
    CHECK_CUDA(cudaMalloc(&d_grad_flat, BATCH_SIZE*FLATTEN_SIZE*sizeof(float)));
    float *d_grad_poolOut;
    CHECK_CUDA(cudaMalloc(&d_grad_poolOut, BATCH_SIZE * NUM_FILTERS * POOL_OUT_ROWS * POOL_OUT_COLS*sizeof(float)));
    float *d_grad_convOut;
    CHECK_CUDA(cudaMalloc(&d_grad_convOut, BATCH_SIZE * NUM_FILTERS * CONV_OUT_ROWS * CONV_OUT_COLS*sizeof(float)));
    float *d_grad_convW, *d_grad_convB;
    CHECK_CUDA(cudaMalloc(&d_grad_convW, convW_size*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_convB, NUM_FILTERS*sizeof(float)));
    float *d_grad_in; 
    CHECK_CUDA(cudaMalloc(&d_grad_in, BATCH_SIZE*IMAGE_ROWS*IMAGE_COLS*sizeof(float)));

    // ---------------------------------------------------------------------------
    // 3) Initialize weights (host random) -> copy to device
    // ---------------------------------------------------------------------------
    srand(123);
    auto randf = [](){ return 0.01f * ((float)rand() / RAND_MAX - 0.5f); };

    float* h_convW = (float*)malloc(convW_size*sizeof(float));
    float* h_convB = (float*)malloc(NUM_FILTERS*sizeof(float));
    for(int i=0; i<convW_size; i++)
        h_convW[i] = randf();
    for(int i=0; i<NUM_FILTERS; i++)
        h_convB[i] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_convW, h_convW, convW_size*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_convB, h_convB, NUM_FILTERS*sizeof(float), cudaMemcpyHostToDevice));
    free(h_convW); free(h_convB);

    float* h_fcW = (float*)malloc(FLATTEN_SIZE*NUM_CLASSES*sizeof(float));
    float* h_fcB = (float*)malloc(NUM_CLASSES*sizeof(float));
    for(int i=0; i<FLATTEN_SIZE*NUM_CLASSES; i++)
        h_fcW[i] = randf();
    for(int i=0; i<NUM_CLASSES; i++)
        h_fcB[i] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_fcW, h_fcW, FLATTEN_SIZE*NUM_CLASSES*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fcB, h_fcB, NUM_CLASSES*sizeof(float), cudaMemcpyHostToDevice));
    free(h_fcW); free(h_fcB);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0);  // Record the start event


    // ---------------------------------------------------------------------------
    // 4) Training
    // ---------------------------------------------------------------------------
    int nBatches = TRAIN_IMAGES / BATCH_SIZE;
    printf("Starting training for %d epochs...\n", EPOCHS);

    for(int e=0; e<EPOCHS; e++){
        double epochLoss = 0.0;

        for(int b=0; b<nBatches; b++){
            // Copy batch
            CHECK_CUDA(cudaMemcpy(d_trainImages,
                       &h_trainImages[(b * BATCH_SIZE) * (IMAGE_ROWS*IMAGE_COLS)],
                       imageBytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_labels,
                       &h_trainLabels[b * BATCH_SIZE],
                       BATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice));

            // Forward pass
            {
                int total = BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                convForwardKernel<<<grid,BLOCK_SIZE>>>(d_trainImages, d_convW, d_convB,
                                                       d_convOut, BATCH_SIZE);
            }
            {
                int total = BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                reluForwardKernel<<<grid,BLOCK_SIZE>>>(d_convOut, total);
            }
            {
                int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                maxPoolForwardKernel<<<grid,BLOCK_SIZE>>>(d_convOut, d_poolOut, d_poolIdx, BATCH_SIZE);
            }
            {
                CHECK_CUDA(cudaMemset(d_flat, 0, BATCH_SIZE*FLATTEN_SIZE*sizeof(float)));
                int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                flattenPoolKernel<<<grid,BLOCK_SIZE>>>(d_poolOut, d_flat, BATCH_SIZE);
            }
            {
                int total = BATCH_SIZE*NUM_CLASSES;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                fcForwardKernel<<<grid,BLOCK_SIZE>>>(d_flat, d_fcW, d_fcB, d_fcOut, BATCH_SIZE);
            }
            {
                int grid = (BATCH_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE;
                softmaxCrossEntropyForwardKernel<<<grid,BLOCK_SIZE>>>(d_fcOut, d_labels, d_loss, d_prob, BATCH_SIZE);
            }

            // Compute batch loss on host
            float h_loss[BATCH_SIZE];
            CHECK_CUDA(cudaMemcpy(h_loss, d_loss, BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
            float batchLoss = 0.0f;
            for(int i=0; i<BATCH_SIZE; i++)
                batchLoss += h_loss[i];
            epochLoss += batchLoss / BATCH_SIZE;

            // Backward pass
            // Softmax -> FC
            {
                int total = BATCH_SIZE;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                softmaxCrossEntropyBackwardKernel<<<grid,BLOCK_SIZE>>>(
                    d_grad_fcOut, d_prob, d_labels, BATCH_SIZE);
            }
            {
                CHECK_CUDA(cudaMemset(d_grad_fcW, 0, FLATTEN_SIZE*NUM_CLASSES*sizeof(float)));
                CHECK_CUDA(cudaMemset(d_grad_fcB, 0, NUM_CLASSES*sizeof(float)));

                int totalParams = FLATTEN_SIZE*NUM_CLASSES + NUM_CLASSES; 
                int grid = (totalParams + BLOCK_SIZE - 1)/BLOCK_SIZE;
                fcBackwardGradParamKernel<<<grid,BLOCK_SIZE>>>(
                    d_grad_fcOut, d_flat,
                    d_grad_fcW, d_grad_fcB,
                    BATCH_SIZE);
            }
            {
                int total = BATCH_SIZE*FLATTEN_SIZE;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                fcBackwardGradInKernel<<<grid,BLOCK_SIZE>>>(d_grad_fcOut, d_fcW, d_grad_flat, BATCH_SIZE);
            }
            // unflatten
            {
                int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                unflattenPoolKernel<<<grid,BLOCK_SIZE>>>(d_grad_flat, d_grad_poolOut, BATCH_SIZE);
            }
            // maxPool backward
            {
                CHECK_CUDA(cudaMemset(d_grad_convOut, 0, BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS*sizeof(float)));
                int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                maxPoolBackwardKernel<<<grid,BLOCK_SIZE>>>(d_grad_poolOut, d_grad_convOut, d_poolIdx, BATCH_SIZE);
            }
            // ReLU backward
            {
                int total = BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                reluBackwardKernel<<<grid,BLOCK_SIZE>>>(d_grad_convOut, d_convOut, d_grad_convOut, total);
            }
            // conv backward
            {
                CHECK_CUDA(cudaMemset(d_grad_convW, 0, convW_size*sizeof(float)));
                CHECK_CUDA(cudaMemset(d_grad_convB, 0, NUM_FILTERS*sizeof(float)));
                int totalParams = convW_size + NUM_FILTERS;
                int grid = (totalParams + BLOCK_SIZE - 1)/BLOCK_SIZE;
                convBackwardWeightKernel<<<grid,BLOCK_SIZE>>>(
                    d_trainImages, d_grad_convOut,
                    d_grad_convW, d_grad_convB,
                    BATCH_SIZE
                );
            }
            {
                CHECK_CUDA(cudaMemset(d_grad_in, 0, BATCH_SIZE*IMAGE_ROWS*IMAGE_COLS*sizeof(float)));
                int total = BATCH_SIZE*IMAGE_ROWS*IMAGE_COLS;
                int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                convBackwardInputKernel<<<grid,BLOCK_SIZE>>>(
                    d_grad_convOut, d_convW, d_grad_in, BATCH_SIZE
                );
            }
            // Update
            {
                // conv params
                {
                    int total = convW_size;
                    int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                    sgdUpdateKernel<<<grid,BLOCK_SIZE>>>(d_convW, d_grad_convW, LEARNING_RATE, total);

                    total = NUM_FILTERS;
                    grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                    sgdUpdateKernel<<<grid,BLOCK_SIZE>>>(d_convB, d_grad_convB, LEARNING_RATE, total);
                }
                // fc params
                {
                    int total = FLATTEN_SIZE*NUM_CLASSES;
                    int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                    sgdUpdateKernel<<<grid,BLOCK_SIZE>>>(d_fcW, d_grad_fcW, LEARNING_RATE, total);

                    total = NUM_CLASSES;
                    grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
                    sgdUpdateKernel<<<grid,BLOCK_SIZE>>>(d_fcB, d_grad_fcB, LEARNING_RATE, total);
                }
            }
        } // end batches

        epochLoss /= nBatches;
        printf("Epoch [%d/%d], avg loss=%.6f\n", e+1, EPOCHS, epochLoss);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  // Wait for the stop event to complete
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    // ---------------------------------------------------------------------------
    // 5) Evaluate on test set
    // ---------------------------------------------------------------------------
    int correct = 0;
    int testBatches = TEST_IMAGES / BATCH_SIZE;
    for(int b=0; b<testBatches; b++){
        CHECK_CUDA(cudaMemcpy(d_trainImages,
                   &h_testImages[(b * BATCH_SIZE) * (IMAGE_ROWS*IMAGE_COLS)],
                   imageBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_labels,
                   &h_testLabels[b * BATCH_SIZE],
                   BATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice));

        // Forward
        {
            int total = BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS;
            int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            convForwardKernel<<<grid,BLOCK_SIZE>>>(d_trainImages, d_convW, d_convB, d_convOut, BATCH_SIZE);
        }
        {
            int total = BATCH_SIZE*NUM_FILTERS*CONV_OUT_ROWS*CONV_OUT_COLS;
            int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            reluForwardKernel<<<grid,BLOCK_SIZE>>>(d_convOut, total);
        }
        {
            int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
            int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            maxPoolForwardKernel<<<grid,BLOCK_SIZE>>>(d_convOut, d_poolOut, d_poolIdx, BATCH_SIZE);
        }
        {
            CHECK_CUDA(cudaMemset(d_flat, 0, BATCH_SIZE*FLATTEN_SIZE*sizeof(float)));
            int total = BATCH_SIZE*NUM_FILTERS*POOL_OUT_ROWS*POOL_OUT_COLS;
            int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            flattenPoolKernel<<<grid,BLOCK_SIZE>>>(d_poolOut, d_flat, BATCH_SIZE);
        }
        {
            int total = BATCH_SIZE*NUM_CLASSES;
            int grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            fcForwardKernel<<<grid,BLOCK_SIZE>>>(d_flat, d_fcW, d_fcB, d_fcOut, BATCH_SIZE);
        }
        {
            int grid = (BATCH_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE;
            softmaxCrossEntropyForwardKernel<<<grid,BLOCK_SIZE>>>(
                d_fcOut, d_labels, d_loss, d_prob, BATCH_SIZE
            );
        }

        // Get predictions
        float* h_prob = (float*)malloc(BATCH_SIZE*NUM_CLASSES*sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_prob, d_prob, BATCH_SIZE*NUM_CLASSES*sizeof(float), cudaMemcpyDeviceToHost));

        int* h_lbl = (int*)malloc(BATCH_SIZE*sizeof(int));
        memcpy(h_lbl, &h_testLabels[b * BATCH_SIZE], BATCH_SIZE*sizeof(int));

        for(int i=0; i<BATCH_SIZE; i++){
            float maxp = -1.0f;
            int pred = -1;
            for(int c=0; c<NUM_CLASSES; c++){
                float p = h_prob[i*NUM_CLASSES + c];
                if(p > maxp){
                    maxp = p;
                    pred = c;
                }
            }
            if(pred == h_lbl[i]) correct++;
        }
        free(h_prob);
        free(h_lbl);
    }
    float accuracy = (float)correct / (float)(testBatches * BATCH_SIZE);
    printf("Test accuracy = %.2f%%\n", accuracy*100.f);

    // ---------------------------------------------------------------------------
    // Cleanup: Free memory 
    // ---------------------------------------------------------------------------
    free(h_trainImages);
    free(h_trainLabels);
    free(h_testImages);
    free(h_testLabels);

    CHECK_CUDA(cudaFree(d_trainImages));
    CHECK_CUDA(cudaFree(d_labels));
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
    CHECK_CUDA(cudaFree(d_grad_poolOut));
    CHECK_CUDA(cudaFree(d_grad_convOut));
    CHECK_CUDA(cudaFree(d_grad_convW));
    CHECK_CUDA(cudaFree(d_grad_convB));
    CHECK_CUDA(cudaFree(d_grad_in));

    printf("Done.\n");
    printf("Total elapsed time: %f ms\n", elapsedTime);

    return 0;
}
