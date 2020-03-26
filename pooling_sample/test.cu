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

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnPoolingDescriptor_t pooling_desc;
    //create descriptor handle
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    std::cout << "cudnnCreatePoolingDescriptor is ok...\n";
    //initialize descriptor
    const int poolDims = 2;
    int windowDimA[poolDims] = {2, 2};
    int paddingA[poolDims] = {0, 0};
    int strideA[poolDims] = {2, 2};
    checkCUDNN(cudnnSetPoolingNdDescriptor(pooling_desc,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_PROPAGATE_NAN,
                                           poolDims,
                                           windowDimA,
                                           paddingA,
                                           strideA));

    std::cout << "cudnnSetPooling2dDescriptor is ok...\n";

    cudnnTensorDescriptor_t in_desc;
    //create input data tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    std::cout << "cudnnCreateTensorDescriptor is ok...\n";
    //initialize input data descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,                  //descriptor handle
                                          CUDNN_TENSOR_NCHW,        //data format
                                          CUDNN_DTYPE,              //data type (precision)
                                          1,                        //number of images
                                          20,                        //number of channels
                                          24,                       //data height
                                          24));                     //data width
    std::cout << "cudnnSetTensor4dDescriptor is ok...\n";
    cudnnTensorDescriptor_t out_desc;
    //create output data tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    std::cout << "cudnnCreateTensorDescriptor is ok...\n";
    //initialize output data descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,                 //descriptor handle
                                          CUDNN_TENSOR_NCHW,        //data format
                                          CUDNN_DTYPE,              //data type (precision)
                                          1,                        //number of images
                                          20,                        //number of channels
                                          12,                        //data height
                                          12));                      //data width

    std::cout << "cudnnSetTensor4dDescriptor is ok...\n";
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

    //Call pooling operator
    checkCUDNN(cudnnPoolingForward(cudnn,         //cuDNN context handle
                                   pooling_desc,  //pooling descriptor handle
                                   &alpha,        //alpha scaling factor
                                   in_desc,       //input tensor descriptor
                                   in_data,       //input data pointer to GPU memory
                                   &beta,         //beta scaling factor
                                   out_desc,      //output tensor descriptor
                                   out_data));    //output data pointer from GPU memory
    std::cout << "cudnnPoolingForward is ok...\n";
    //allocate array on CPU for output tensor data
    dtype *result = (dtype *) malloc(OUT_DATA_BYTES);
    //copy output data from GPU
    cudaMemcpy(result, out_data, OUT_DATA_BYTES, cudaMemcpyDeviceToHost);

    //loop over and check that the forward pass outputs match expected results (exactly)
    int err = 0;
    for (int i = 0; i < OUT_SIZE; i++) {
        if (result[i] != output[i]) {
            std::cout << "Error! Expected " << output[i] << " got " << result[i] << " for idx " << i << std::endl;
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
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroy(cudnn);

    return 0;
}
