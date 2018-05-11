
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

// We define constant memory for the filter data
__constant__ float Kc[MAX_FILTER_BANK];

/*********************/
/* Layer 1 Constants */
/*********************/

#define C1   1    // Input Feature Maps
#define HI_1 64   // Height of input maps
#define WI_1 64   // Width  of input maps
#define IN_SIZE1 4096 // Size of input maps

#define M1   6    // Output Feature Maps
#define HO_1 60   // Height of output maps
#define WO_1 60   // Width  of output maps
#define OUT_SIZE1 3600  // Size of output maps

/*********************/
/* Layer 2 Constants */
/*********************/

#define C2   6    // Input Feature Maps
#define HI_2 30   // Height of input maps
#define WI_2 30   // Width  of input maps
#define IN_SIZE2 900 // Size of input maps

#define M2   16   // Output Feature Maps
#define HO_2 26   // Height of output maps
#define WO_2 26   // Width  of output maps
#define OUT_SIZE2 676  // Size of output maps

#define BS2_X 160
#define BS2_Y 6

__global__ void unroll_mm1(const float* X, float* Y)
{
  /* This kernel takes a *single* input image from convolution layer 1 and
   * computes the output data for that image via unrolled indexing.
   */

  // Macro for calculating 1D index from 3D inputs (which are actually 2D in layer 1)
  #define calc2to1(i1, i0) ((i1) * (WI_1) + i0)
  // Macro for accessing the 3 dimensional *input* data (as 2D)
  #define x2i(i1, i0) X[calc2to1(i1, i0)]
  // Macro for accessing the 4 dimensional kernel data (as 3D)
  #define kc3(i2, i1, i0) Kc[(i2) * (FILTER_SIZE) + (i1) * (K) + i0]
  // Macro for accessing the 3 dimensional output data (as 2D)
  #define y2_1(i1, i0) Y[(i1) * (OUT_SIZE1) + i0]

  // Some local values
  int start_inRow, start_inCol, p, q, cur_outMap, cur_outIdx;
  // Grab the thread's index
  int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // Check that the thread is within the appropriate bounds
  if (t < M1*OUT_SIZE1)
  {
    float res = 0.0f; // Sum value initialization

    // Determine which output map this thread corresponds to
    cur_outMap = t / OUT_SIZE1;
    // Determine which linear index this thread corresponds to within a single map
    cur_outIdx = t % OUT_SIZE1;

    // These values are the top left corner of the section we're applying the
    // filter bank to within the input channel
    start_inRow = cur_outIdx / WO_1; // Starting row in input
    start_inCol = cur_outIdx % WO_1; // Starting column in input

    // For each element in the filter bank
    for(p = 0; p < K; p++)
    {
      for(q = 0; q < K; q++)
      {
        res += x2i(start_inRow + p, start_inCol + q) * kc3(cur_outMap, p, q);
      }
    }
    // Finally, set the according output element
    y2_1(cur_outMap, cur_outIdx) = res;
  }
}

__global__ void unroll_mm2(const float* X, float* Y)
{
  /* This kernel takes a *single* input image from convolution layer 2 and
   * computes the output data for that image via unrolled indexing.
   */

  // Macro for calculating 1D index from 3D inputs
  #define calc3to1(i2, i1, i0) ((i2) * (IN_SIZE2) + (i1) * (WI_2) + i0)
  // Macro for accessing the 3 dimensional *input* data
  #define x3i(i2, i1, i0) X[calc3to1(i2, i1, i0)]
  // Macro for accessing the 4 dimensional kernel data
  #define kc4(i3, i2, i1, i0) Kc[(i3) * (C2 * FILTER_SIZE) + (i2) * (FILTER_SIZE) + (i1) * (K) + i0]
  // Macro for accessing the 3 dimensional output data (as 2D)
  #define y2_2(i1, i0) Y[(i1) * (OUT_SIZE2) + i0]

  // Some local values
  int start_inRow, start_inCol, p, q, cur_outMap, cur_outIdx;
  // Grab the thread's linear x index and y index
  int x = blockIdx.x * BS2_X + threadIdx.x;
  int y = threadIdx.y;

  // Check that the thread is within the appropriate bounds
  if (x < M2*OUT_SIZE2)
  {
    float res = 0.0f; // Sum value initialization

    // Determine which output map this thread corresponds to
    cur_outMap = x / OUT_SIZE2;
    // Determine which linear index this thread corresponds to within a single map
    cur_outIdx = x % OUT_SIZE2;

    // These values are the top left corner of the section we're applying the
    // filter bank to within the input channel
    start_inRow = cur_outIdx / WO_2; // Starting row in input
    start_inCol = cur_outIdx % WO_2; // Starting column in input

    // For each element in the filter bank
    for(p = 0; p < K; p++)
    {
      for(q = 0; q < K; q++)
      {
        res += x3i(y, start_inRow + p, start_inCol + q) * kc4(cur_outMap, y, p, q);
      }
    }
    // Finally, add this threads sum *atomically* to the corresponding output map element
    // This has to be done since this kernel is invoked with 6 threads per output element
    atomicAdd( &(y2_2(cur_outMap, cur_outIdx)), res);
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
    // Generate multiple streams
    cudaStream_t stream[STREAM];
    for (int i = 0; i < STREAM; ++i)
      cudaStreamCreate(&stream[i]);

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];      // Batch size
    const int C = x.shape_[1];      // Input Feature Maps
    const int M = y.shape_[1];      // Output Feature Maps

    const int BANK_SIZE = (M * C * FILTER_SIZE);

    // Set the constant memory for the filter banks
    cudaMemcpyToSymbol(Kc, k.dptr_, BANK_SIZE*sizeof(float));

    if(C == 1)
    {
      // Generate the stream with the appropriate MXNet stream
      //cudaStream_t s = y.stream_->stream_;

      // ~~~ Set the kernel dimensions ~~~
      // First we setup for the unroll kernels
      const int num_threads = M1*OUT_SIZE1;
      const int num_blocks = ceil(num_threads / (BLOCK_SIZE * 1.0));
      const dim3 blockDim1(BLOCK_SIZE, 1, 1);
      const dim3 gridDim1(num_blocks, 1, 1);
      // ~~~ Now we begin the bread n butter of all this hard labor ~~~
      // For each image in the batch
      for (int b = 0; b < B; b += STREAM)
      {
        for(int i = 0; i < STREAM; i++)
        {
          float* x_ptr = &x.dptr_[(b+i)*IN_SIZE1];
          float* y_ptr = &y.dptr_[(b+i)*M1*OUT_SIZE1];
          unroll_mm1<<<gridDim1, blockDim1, 0, stream[i]>>>(x_ptr, y_ptr);
        }
      }
    }
    else
    {
      // ~~~ Set the kernel dimensions ~~~
      // First we setup for the unroll kernels
      const int num_threads_x = M2*OUT_SIZE2;
      const int num_blocks_x = ceil(num_threads_x / (BS2_X * 1.0));
      const dim3 blockDim2(BS2_X, BS2_Y, 1);
      const dim3 gridDim2(num_blocks_x, 1, 1);

      // ~~~ Now we begin the bread n butter of all this hard labor ~~~
      // For each image in the batch
      for (int b = 0; b < B; b += STREAM)
      {
        for(int i = 0; i < STREAM; i++)
        {
          float* x_ptr = &x.dptr_[(b+i)*C2*IN_SIZE2];
          float* y_ptr = &y.dptr_[(b+i)*M2*OUT_SIZE2];
          unroll_mm2<<<gridDim2, blockDim2, 0, stream[i]>>>(x_ptr, y_ptr);
        }
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
