
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32
#define COMBINE 1 //number of batches to be combined together
#define STREAM 2    //number of streams 
#define SEGSIZE 2 //segments of batches to work on for each stream

//#define MAX_OUTPUT_CHANNELS 16
//#define MAX_INPUT_CHANNELS  6
#define K 5
#define MAX_FILTER_BANK 16*6*5*5

// We define constant memory for the filter data
__constant__ float Kc[MAX_FILTER_BANK];

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

    cudaMemcpyToSymbol(Kc, k.dptr_, BANK_SIZE*sizeof(float));

    // Generate multiple streams
    cudaStream_t stream[STREAM];

    for (int i = 0; i < STREAM; ++i)
      cudaStreamCreate(&stream[i]);

    // ~~~ Set the kernel dimensions ~~~
    const int Z = ceil((float)W_out/TILE_WIDTH)*ceil((float)H_out/TILE_WIDTH);
    const dim3 gridDim2(COMBINE, M, Z);
    const dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

	// Cuda Mallocs
	float *deviceInputMap0, *deviceInputMap1, *deviceOutputMap0, *deviceOutputMap1;

	cudaMalloc((void **)&deviceInputMap0, sizeof(float)*C*H*W*COMBINE);
	cudaMalloc((void **)&deviceInputMap1, sizeof(float)*C*H*W*COMBINE);
	cudaMalloc((void **)&deviceOutputMap0, sizeof(float)*M*H_out*W_out*COMBINE);
	cudaMalloc((void **)&deviceOutputMap1, sizeof(float)*M*H_out*W_out*COMBINE);

    // For each image in the batch
    for (int b = 0; b < B; b += COMBINE*STREAM){
	
		cudaMemcpyAsync(deviceInputMap0, &x.dptr_[(b+COMBINE*1)*C*H*W], C*H*W*COMBINE*sizeof(float), cudaMemcpyHostToDevice, stream[0]);
		cudaMemcpyAsync(deviceInputMap1, &x.dptr_[(b+COMBINE*2)*C*H*W], C*H*W*COMBINE*sizeof(float), cudaMemcpyHostToDevice, stream[1]);
		
		convolution_layer<<<gridDim2, blockDim2, 0, stream[0]>>>(deviceInputMap0, deviceOutputMap0, C, H, W, W_out, M);
		convolution_layer<<<gridDim2, blockDim2, 0, stream[1]>>>(deviceInputMap1, deviceOutputMap1, C, H, W, W_out, M);

		cudaMemcpyAsync(deviceOutputMap0, &y.dptr_[(b+COMBINE*1)*M*H_out*W_out], sizeof(float)*M*H_out*W_out*COMBINE, cudaMemcpyDeviceToHost, stream[0]);
		cudaMemcpyAsync(deviceOutputMap1, &y.dptr_[(b+COMBINE*2)*M*H_out*W_out], sizeof(float)*M*H_out*W_out*COMBINE, cudaMemcpyDeviceToHost, stream[1]);

      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
      //destroy cuda streams
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
