
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32 
#define COMBINE 32    //number of batches to be combined together, set to one means no combination 
#define STREAM 16    //number of streams 

//#define MAX_OUTPUT_CHANNELS 16
//#define MAX_INPUT_CHANNELS  6
//#define MAX_FILTER_WIDTH    5
#define MAX_FILTER_BANK 16*6*5*5

// We define constant memory for the filter data
__constant__ float Kc[MAX_FILTER_BANK];


__global__ void convolution_layer(float* X, float* Y,
    int C, int H_in, int W_in, int W_out, int K, int M){
    int H_out = H_in - K + 1;
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
					acc += X[n*(C*H_in*W_in)+c*(H_in*W_in)+(h+p)*(W_in)+(w+q)] 
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

    cudaStream_t stream[STREAM];
    for (int i = 0; i < STREAM; ++i)
      cudaStreamCreate(&stream[i]);
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];      // Batch size
    const int C = x.shape_[1];      // Input Feature Maps
    const int H = x.shape_[2];      // Height of input maps
    const int W = x.shape_[3];      // Width  of input maps
    const int M = y.shape_[1];      // Output Feature Maps
    const int K = k.shape_[3];      // Filter Dimensions

    const int W_out = (W - K + 1);    // Height of output maps
    const int H_out = (H - K + 1);    // Width  of output maps

    // Grab the pointer to the filter maps
    const float* k_ptr = k.dptr_;
    cudaMemcpyToSymbol(Kc, k_ptr, (M*C*K*K)*sizeof(float));

    // ~~~ Set the kernel dimensions ~~~
	int Z = ceil((float)W_out/TILE_WIDTH)*ceil((float)H_out/TILE_WIDTH);
	dim3 dimGrid(COMBINE, M, Z);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // float* k_ptr = k.dptr_;
    // For each image in the batch
    for (int b = 0; b < B; b+=STREAM*COMBINE)
    {
        // float* x_ptr = &x.dptr_[b*C*H*W];
        // float* y_ptr = &y.dptr_[b*M*W_out*H_out];
        // convolution_layer<<<dimGrid, dimBlock>>>(x_ptr, k_ptr, y_ptr, C, H, W, W_out, K, M);
      for(int i = 0; i<STREAM; i++){
        float* x_ptr = &x.dptr_[(b+i*COMBINE)*C*H*W];
        float* y_ptr = &y.dptr_[(b+i*COMBINE)*M*W_out*H_out];
        convolution_layer<<<dimGrid, dimBlock>>>(x_ptr, y_ptr, C, H, W, W_out, K, M);
        // convolution_layer<<<dimGrid, dimBlock>>>(x_ptr, k_ptr, y_ptr, C, H, W, W_out, K, M);
      }
      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    // destroy cuda streams
    for (int i = 0; i < STREAM; ++i)
      cudaStreamDestroy(stream[i]);
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
