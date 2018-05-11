
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32
#define COMBINE 32   //number of batches to be combined together
#define STREAM 16    //number of streams 

//#define MAX_OUTPUT_CHANNELS 16
//#define MAX_INPUT_CHANNELS  6
#define K 5
#define FILTER_SIZE 25
#define MAX_FILTER_BANK 2400

#define SHARED_WIDTH 36
#define SHARED_SIZE 1296

// We define constant memory for the filter data
__constant__ float Kc[MAX_FILTER_BANK];

/*********************/
/* Layer 1 Constants */
/*********************/

#define C1   1    // Input Feature Maps
#define HI_1 64   // Height of input maps
#define WI_1 64   // Width  of input maps
#define IN_SIZE1 4096 // Size of input maps
#define TOTAL_IN_SIZE1 4096 // C1 * IN_SIZE1

#define M1   6    // Output Feature Maps
#define HO_1 60   // Height of output maps
#define WO_1 60   // Width  of output maps
#define OUT_SIZE1 3600  // Size of output maps
#define TOTAL_OUT_SIZE1 21600 // M1 * IN_SIZE1

#define Z1 4
#define WGRID1 2

/*********************/
/* Layer 2 Constants */
/*********************/

#define C2   6    // Input Feature Maps
#define HI_2 30   // Height of input maps
#define WI_2 30   // Width  of input maps
#define IN_SIZE2 900 // Size of input maps
#define TOTAL_IN_SIZE2 5400 // C2 * IN_SIZE2

#define M2   16   // Output Feature Maps
#define HO_2 26   // Height of output maps
#define WO_2 26   // Width  of output maps
#define OUT_SIZE2 676  // Size of output maps
#define TOTAL_OUT_SIZE2 10816 // M2 * IN_SIZE2

#define Z2 1
#define WGRID2 1

__global__ void convolution_layer_shared1(float* X, float* Y){

  __shared__ float X_s[SHARED_SIZE]; 

  int n, m, h0, w0, h_base, w_base, h, w;

  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.y;
  w0 = threadIdx.x;
  h_base = blockIdx.z / WGRID1 * TILE_WIDTH;
  w_base = blockIdx.z % WGRID1 * TILE_WIDTH;
  h = h_base + h0;
  w = w_base + w0;

  float acc = 0.;
  //#pragma unroll
  for (int i = h; i < h_base + SHARED_WIDTH; i+=TILE_WIDTH)
  {
    //#pragma unroll
    for (int j = w; j < w_base + SHARED_WIDTH; j+=TILE_WIDTH)
    {
      X_s[(i-h_base)*SHARED_WIDTH+j-w_base]=X[n*(TOTAL_IN_SIZE1)+i*WI_1+j]; //n,c,h(i),w(j)
    }
  }
  __syncthreads();

  if(h < HO_1 && w < WO_1)
  {
    #pragma unroll
    for (int p = 0; p < K; p++)
    {
      #pragma unroll
      for (int q = 0; q < K; q++)
      {
        acc += X_s[(h0+p)*SHARED_WIDTH+w0+q]*Kc[m*(C1*FILTER_SIZE)+p*K+q];
      }
    }
    __syncthreads();

    Y[n*(TOTAL_OUT_SIZE1)+m*(OUT_SIZE1)+h*WO_1+w] = acc;
  }
}

__global__ void convolution_layer_shared2(float* X, float* Y){

  __shared__ float X_s[SHARED_SIZE]; 

  int n, m, h0, w0, h_base, w_base, h, w;

  n = blockIdx.x;
  m = blockIdx.y;
  h0 = threadIdx.y;
  w0 = threadIdx.x;
  h_base = blockIdx.z / WGRID2 * TILE_WIDTH;
  w_base = blockIdx.z % WGRID2 * TILE_WIDTH;
  h = h_base + h0;
  w = w_base + w0;

  float acc = 0.;
  #pragma unroll
  for (int c = 0; c < C2; c++)
  {
    //#pragma unroll
    for (int i = h; i < h_base + SHARED_WIDTH; i+=TILE_WIDTH)
    {
      //#pragma unroll
      for (int j = w; j < w_base + SHARED_WIDTH; j+=TILE_WIDTH)
      {
        X_s[(i-h_base)*SHARED_WIDTH+j-w_base]=X[n*(TOTAL_IN_SIZE2)+c*(IN_SIZE2)+i*WI_2+j]; //n,c,h(i),w(j)
      }
    }
    __syncthreads();

    if(h < HO_2 && w < WO_2)
    {
      #pragma unroll
      for (int p = 0; p < K; p++)
      {
        #pragma unroll
        for (int q = 0; q < K; q++)
        {
          acc += X_s[(h0+p)*SHARED_WIDTH+w0+q]*Kc[m*(C2*FILTER_SIZE)+c*(FILTER_SIZE)+p*K+q];
        }
      }
    }
    __syncthreads();
  }
  if(h < HO_2 && w < WO_2)
  {
    Y[n*(TOTAL_OUT_SIZE2)+m*(OUT_SIZE2)+h*WO_2+w] = acc;
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
    const int M = y.shape_[1];      // Output Feature Maps

    const int BANK_SIZE = (M * C * FILTER_SIZE);

    // Set the constant memory for the filter banks
    cudaMemcpyToSymbol(Kc, k.dptr_, BANK_SIZE*sizeof(float));

    // Generate multiple streams
    cudaStream_t stream[STREAM];
    for (int i = 0; i < STREAM; ++i)
      cudaStreamCreate(&stream[i]);

    if( C == C1 )
    {
      // ~~~ Set the kernel dimensions ~~~
      const dim3 gridDim1(COMBINE, M1, Z1);
      const dim3 blockDim1(TILE_WIDTH, TILE_WIDTH, 1);

      // For each image in the batch
      for (int b = 0; b < B; b += STREAM * COMBINE)
      {
        for(int i = 0; i < STREAM; i++)
        {
          if(b+i*COMBINE < B)
          {
            float* x_ptr = &x.dptr_[(b+i*COMBINE)*TOTAL_IN_SIZE1];
            float* y_ptr = &y.dptr_[(b+i*COMBINE)*TOTAL_OUT_SIZE1];
            convolution_layer_shared1<<<gridDim1, blockDim1, 0, stream[i]>>>(x_ptr, y_ptr);
          }
        }
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
      }
    }
    else // C == C2
    {
      // ~~~ Set the kernel dimensions ~~~
      const dim3 gridDim2(COMBINE, M2, Z2);
      const dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

      // For each image in the batch
      for (int b = 0; b < B; b += STREAM * COMBINE)
      {
        for(int i = 0; i < STREAM; i++)
        {
          if(b+i*COMBINE < B)
          {
            float* x_ptr = &x.dptr_[(b+i*COMBINE)*TOTAL_IN_SIZE2];
            float* y_ptr = &y.dptr_[(b+i*COMBINE)*TOTAL_OUT_SIZE2];
            convolution_layer_shared2<<<gridDim2, blockDim2, 0, stream[i]>>>(x_ptr, y_ptr);
          }
        }
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
      }
    }
    // destroy cuda streams
    for (int i = 0; i < STREAM; ++i)
      cudaStreamDestroy(stream[i]);

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