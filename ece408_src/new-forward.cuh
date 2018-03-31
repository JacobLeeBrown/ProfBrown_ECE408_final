
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 1024
#define TILE_WIDTH 16

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    

#undef y4d
#undef x4d
#undef k4d
}

__global__ void unroll_Kernel(const int C, const int W_out, const int W_unroll
                              const int K, const float* X, float* X_unroll)
{
  /* This kernel takes a *single* input image as an array of input maps. It will
   * convert this input data into an unrolled matrix, which will have duplicate
   * elements.
   */

  /**** LEGEND ****
   C        = number of input maps
   W_out    = width of output map
   W_unroll = *total* width of unrolled matrix
   K        = Width of filter banks
   X        = Image as input maps
   X_unroll = Target location for unrolled matrix
   */

  // Some local values
  int cur_channel, cur_outCol, start_inRow, start_inCol, start_outRow, cur_outRow, p, q;
  // Grab the thread's index
  int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // Check that the thread index maps to a column section within bounds
  if (t < C * W_unroll)
  {
    // Determine which channel the thread maps to
    cur_channel = t / W_unroll;
    // Determine which ouput element the thread maps to, which is equivalent to
    // the current row we're calculating for the unrolled matrix
    cur_outCol  = t % W_unroll;

    // These values are the top left corner of the section we're applying the
    // filter bank to within the input channels 
    start_inRow = cur_output / W_out; // Starting row in channel maps
    start_inCol = cur_output % W_out; // Starting column in channel maps

    // This value gives us the starting row of the current channel's sections
    // within the unrolled matrix
    start_outRow = cur_channel * K * K;

    // For each element in the filter bank
    for(p = 0; p < K; p++)
      for(q = 0; q < K; q++)
      {
        // Now we calculate the exact row index for the unrolled matrix
        cur_outRow = start_outRow + p * K + q; 
        X_unroll[cur_outRow, cur_outCol] = X[cur_channel, start_inRow + p, start_inCol + q];
      }
  }
}


__global__ void matrixMultiply(float *A, float *B, float *C, 
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the C element to work on
  int Row = blockIdx.y*TILE_WIDTH + ty;
  int Col = blockIdx.x*TILE_WIDTH + tx;
  
  // We need to calculate the phases to account for different shaped inputs
  int phases = ceil(numAColumns/(TILE_WIDTH*1.0));

  float Cvalue = 0;
  
  // Loop over the A and B tiles required to compute the C element
  for (int m = 0; m < phases; ++m) 
  {
    // Store this useful value
    int curTile = m * TILE_WIDTH;
    // Collaborative loading of A and B tiles into shared memory
    // We can only load data into our tiles if the data is there to load
    
    // First we check if we are within the bounds of input A
    if (Row < numARows && tx + curTile < numAColumns) {
      subTileA[ty][tx] = A[Row*numAColumns + curTile + tx];
    }
    else { // We are outside the bounds of input A
      subTileA[ty][tx] = 0.;
    }
    
    // Now we repeat for the bounds of input B
    if (Col < numBColumns  && curTile + ty < numBRows) {
      subTileB[ty][tx] = B[(curTile + ty)*numBColumns + Col];
    }
    else { // We are outside the bounds of input B
      subTileB[ty][tx] = 0.;
    }
    
    __syncthreads();
    // Now we can update our local partial inner product
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Cvalue += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }
  
  // Make sure we are within the bounds of output C
  if (Row < numCRows && Col < numCColumns) {
    C[Row*numCColumns + Col] = Cvalue;
  }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];      // Batch size
    const int C = x.shape_[1];      // Input Feature Maps
    const int H = x.shape_[2];      // Height of input maps
    const int W = x.shape_[3];      // Width  of input maps
    const int M = y.shape_[1];      // Output Feature Maps
    const int K = k.shape_[3];      // Filter Dimensions

    const int W_out = W – K + 1;    // Height of output maps
    const int H_out = H – K + 1;    // Width  of output maps

    const int H_unroll = H_out * W_out; // Height of unrolled matrix
    const int W_unroll = C * K * K;     // Width  of unrolled matrix

    // Allocated the space for the unrolled input (which will get re-written)
    float* X_unrolled = malloc(W_unroll * H_unroll * sizeof(float));

    // Set the kernel dimensions
    const int num_threads = C * H_out * W_out;
    const int num_blocks = ceil(num_threads / (BLOCK_SIZE * 1.0));

    // For each image in the batch
    for (int b = 0; b < B; b++)
    {
      unroll_Kernel<<<num_blocks, BLOCK_SIZE>>>(x[b]);
      gemm(H_unroll, M, W_unroll, X_unrolled, W, Y[b]);
    }


    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif