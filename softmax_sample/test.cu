#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "float32.h"

#define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
#define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  }

float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DTYPE,
                                          1, 10,
                                          1,
                                          1));

    cudnnTensorDescriptor_t out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DTYPE,
                                          1, 10,
                                          1,
                                          1));

    stype alpha = 1.0f;
    stype beta = 0.0f;
    //GPU data pointers
    dtype *in_data, *out_data;
    //allocate arrays on GPU
    cudaMalloc(&in_data, IN_DATA_BYTES);
    cudaMalloc(&out_data, OUT_DATA_BYTES);
    //copy input data to GPU array
    cudaMemcpy(in_data, input, IN_DATA_BYTES, cudaMemcpyHostToDevice);
    //initize output data on GPU
    cudaMemset(out_data, 0, OUT_DATA_BYTES);

    checkCUDNN(cudnnSoftmaxForward(cudnn,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   in_desc,
                                   in_data,
                                   &beta,
                                   out_desc,
                                   out_data));

    //allocate array on CPU for output tensor data
    dtype *result = (dtype *) malloc(OUT_DATA_BYTES);
    //copy output data from GPU
    cudaMemcpy(result, out_data, OUT_DATA_BYTES, cudaMemcpyDeviceToHost);

    //loop over and check that the forward pass outputs match expected results (exactly)
    int err = 0;
    for (int i = 0; i < OUT_SIZE; i++) {
        float diff = getError(result[i], output[i]);
        if (diff < 0) diff = -diff;
        if (diff > 1e-05) {
            std::cout << "Error! Expected " << output[i] << " got " << result[i] << " for idx " << i
                      << std::endl;
            std::cout << "diff " << diff << std::endl;
            err++;
        }
    }

    std::cout << "Forward finished with " << err << " errors" << std::endl;

    //free CPU arrays
    free(result);

    //free GPU arrays
    cudaFree(in_data);
    cudaFree(out_data);

    //free cuDNN descriptors
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroy(cudnn);

    return 0;
}
