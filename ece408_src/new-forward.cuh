
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
#define MAX_FILTER_BANK 16*6*5*5

#define IN_SIZE1 64*64

// We define constant memory for the filter data
__constant__ float Kc[MAX_FILTER_BANK];

__global__ void convMM(const float *X, float *Y, const int W,
                       const int H_out, const int W_out, const int M)
{

  __shared__ float Xs[IN_SIZE1];

  unsigned int idx = threadIdx.x;

  // First we need to load the input maps into shared memory
  if(idx < (IN_SIZE1 / 2))
  {
    unsigned int offset = idx + blockIdx.x * IN_SIZE1;
    Xs[idx] = X[offset];
    Xs[idx + (IN_SIZE1 / 2)] = X[offset + (IN_SIZE1 / 2)];
  }
  __syncthreads();

  float res = 0.0f;

  // These values are the top left corner of the section we're applying the
  // filter bank to within the input channels 
  unsigned int start_inRow = idx / W_out;
  unsigned int start_inCol = idx % W_out;

  // For each element in the filter bank
  #pragma unroll
  for(int p = 0; p < K; p++)
  {
    #pragma unroll
    for(int q = 0; q < K; q++)
    {
      res += Xs[(start_inRow + p)*W + (start_inCol+q)] * Kc[p * K + q];
    }
  }
  Y[blockIdx.x * H_out * W_out * M + idx] = res;
}

__global__ void convolution_layer(float* X, float* Y, const int C, 
                                  const int H, const int W, 
                                  const int W_out, const int M){
  int H_out = H - K + 1;
  int n, m, h, w, c, p, q;
  int W_grid = ceilf((float)W_out/TILE_WIDTH);

  n = blockIdx.x;
  m = blockIdx.y;
  h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
  w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

  float acc = 0;
  for (c = 0; c < C; c++) { 
    for (p = 0; p < K; p++) 
      for (q = 0; q < K; q++)
        if(h < H_out && w < W_out)
          acc += X[n*(C*H*W)+c*(H*W)+(h+p)*(W)+(w+q)] 
                                * Kc[m*(C*K*K)+c*(K*K)+p*(K)+q];
  }
  if(h < H_out && w < W_out)
  {
    Y[n*(M*H_out*W_out) + m*(H_out*W_out) + h*(W_out) + w] = acc;
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

    const int H_out = (H - K + 1);    // Height of output maps
    const int W_out = (W - K + 1);    // Width  of output maps

    const int FILTER_SIZE = (K * K);
    const int BANK_SIZE = (M * C * FILTER_SIZE);

    // Set the constant memory for the filter banks
    cudaMemcpyToSymbol(Kc, k.dptr_, BANK_SIZE*sizeof(float));

    /*if(C == 1)
    {
      // Generate the stream with the appropriate MXNet stream
      cudaStream_t s = y.stream_->stream_;

      //const int IN_SIZE = (H * W);
      const int OUT_SIZE = (H_out * W_out);

      // ~~~ Set the kernel dimensions ~~~
      // Each grid will have one block for each image
      const dim3 gridDim1(B, 1, 1);
      // Each block will be the size of an output map, and compute each output map for an image
      const dim3 blockDim1(OUT_SIZE, 1, 1);
      // ~~~ Now we begin the bread n butter of all this hard labor ~~~
      convMM<<<gridDim1, blockDim1, 0, s>>>(x.dptr_, y.dptr_, W,
                                            H_out, W_out, M);
    }*/
    //else
    //{
      // Generate multiple streams
      cudaStream_t stream[STREAM];
      for (int i = 0; i < STREAM; ++i)
        cudaStreamCreate(&stream[i]);

      // ~~~ Set the kernel dimensions ~~~
      int Z = ceil((float)W_out/TILE_WIDTH)*ceil((float)H_out/TILE_WIDTH);
      const dim3 gridDim2(COMBINE, M, Z);
      const dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

      // For each image in the batch
      for (int b = 0; b < B; b += STREAM * COMBINE)
      {
        for(int i = 0; i < STREAM; i++)
        {
          float* x_ptr = &x.dptr_[(b+i*COMBINE)*C*H*W];
          float* y_ptr = &y.dptr_[(b+i*COMBINE)*M*W_out*H_out];
          convolution_layer<<<gridDim2, blockDim2, 0, stream[i]>>>(x_ptr, y_ptr, C, H, W, W_out, M);
        }
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
      }
      // destroy cuda streams
      for (int i = 0; i < STREAM; ++i)
        cudaStreamDestroy(stream[i]);
    //}

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
