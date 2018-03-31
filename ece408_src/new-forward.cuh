
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void unroll(int B, int C, int H, int W, int K, int float *X, float *X_unroll){
	int c, s, H_out, W_out, h_unroll, w_base, b, p, q;
	int t = blockId.x * 1024 + threadId.x;
	int H_out = H-K+1;
	int W_out = W-K+1;
	int W_unroll = H_out*W_out;

	if(t<C*W_unroll){
		c = t/W_unroll;
		s = t%w_unroll;
		h_out = s/W_out;
		w_out = s%W_out;
		h_unroll = h_out*W_out+w_out;
		w_base = c*K*K;
		for(b=0;b<B;b++){
			for(p=0;p<K;p++){
				for(q=0;q<K;q++){
				w_unroll = w_base+p*K+q;
				X_unroll[b, h_unroll, w_unroll] = X[b, c, h+p, w+q);
				}
			}
		}
	}
	
}

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

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];     // Batch size
    const int C = x.shape_[1];     // Input Feature Maps
    const int H = x.shape_[2];     // Height of input maps
    const int W = x.shape_[3];     // Width  of input maps
    const int M = y.shape_[1];     // Output Feature Maps
    const int H_out = H-K+1;	   // Height of output maps
    const int W_out = W-K+1; 	   // Width  of output maps
    const int K = w.shape_[3];     // Filter Dimensions
    // ...

	// Unroll Kernel Call
	int num_blocks = ceil(C*H_out*W_out)/1024;
	int w_unroll = h*W_out+w;
	int h_unroll = w_base+p*K+q;
	float *X_unrolled = malloc(B*W_unroll*H_unroll*sizeof(float));

	unroll<<<num_blocks,1024>>>(B, C, H, W, K, x.dptr, X_unrolled); 

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
