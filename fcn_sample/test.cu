#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "gemv.h"
#include "float32.h"

#define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
#define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))
#define WEIGHT_DATA_BYTES (WEIGHT_SIZE*sizeof(dtype))
//#define BIAS_DATA_BYTES (BIAS_SIZE*sizeof(dtype))

float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

int main() {
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    stype alpha = 1.0f;
    stype beta = 0.0f;

    //GPU data pointers
    dtype *in_data, *out_data;
    dtype *weight_data;

    //allocate arrays on GPU
    cudaMalloc(&in_data, IN_DATA_BYTES);
    cudaMalloc(&out_data, OUT_DATA_BYTES);
    cudaMalloc(&weight_data, WEIGHT_DATA_BYTES);
//    cudaMalloc(&bias_data, BIAS_DATA_BYTES);

    //copy input/weight/bias data to GPU array
    cudaMemcpy(in_data, input, IN_DATA_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data, weight, WEIGHT_DATA_BYTES, cudaMemcpyHostToDevice);
//    cudaMemcpy(bias_data, bias, BIAS_DATA_BYTES, cudaMemcpyHostToDevice);

    //initize output data on GPU
    cudaMemset(out_data, 0, OUT_DATA_BYTES);

    int dim_x = 4 * 4 * 5;
    int dim_y = 3;
//    cudaMemcpy(out_data, bias_data, OUT_DATA_BYTES, cudaMemcpyDeviceToDevice);
    //Call fcn operator
    gemv(cublasHandle, dim_x, dim_y, alpha, weight_data, in_data, beta, out_data);

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
    cudaFree(weight_data);
//    cudaFree(bias_data);

    //free cublas descriptors
    cublasDestroy(cublasHandle);

    return 0;
}
