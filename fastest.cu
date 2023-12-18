// op time 1:   6.2ms
// op time 2:   18.3ms
// total:       24.5ms

// loop unrolling + resitrct + utning
// constant memory
// sweeping parameters

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 20

extern __shared__ float shared_mask[];

__constant__ float const_mask[16384];
__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S) {
    /*
    Function paramter definitions:
    output - output,  
    input - input, 
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + (i0)]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + (i0)]
    #define const_mask_4d(i3, i2, i1, i0) const_mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + (i0)]


    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    int W_grid = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;;

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h_index = blockIdx.z / W_grid;  // blockIdx.z is the block index within the grid in the z dimension
    int w_index = blockIdx.z % W_grid;  // W_grid is the number of blocks along the width
    int h_out = h_index * BLOCK_SIZE + threadIdx.y;
    int w_out = w_index * BLOCK_SIZE + threadIdx.x;

    if (w_out < W_out && h_out < H_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            int h_in = h_out * S;
            int w_in = w_out * S;
            if (K == 3){
                acc += in_4d(b, c, h_in + 0, w_in + 0) * const_mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, h_in + 0, w_in + 1) * const_mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, h_in + 0, w_in + 2) * const_mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, h_in + 1, w_in + 0) * const_mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, h_in + 1, w_in + 1) * const_mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, h_in + 1, w_in + 2) * const_mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, h_in + 2, w_in + 0) * const_mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, h_in + 2, w_in + 1) * const_mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, h_in + 2, w_in + 2) * const_mask_4d(m, c, 2, 2);
            } else if (K == 5){
                acc += in_4d(b, c, h_in + 0, w_in + 0) * const_mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, h_in + 0, w_in + 1) * const_mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, h_in + 0, w_in + 2) * const_mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, h_in + 0, w_in + 3) * const_mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, h_in + 0, w_in + 4) * const_mask_4d(m, c, 0, 4);
                acc += in_4d(b, c, h_in + 1, w_in + 0) * const_mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, h_in + 1, w_in + 1) * const_mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, h_in + 1, w_in + 2) * const_mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, h_in + 1, w_in + 3) * const_mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, h_in + 1, w_in + 4) * const_mask_4d(m, c, 1, 4);
                acc += in_4d(b, c, h_in + 2, w_in + 0) * const_mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, h_in + 2, w_in + 1) * const_mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, h_in + 2, w_in + 2) * const_mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, h_in + 2, w_in + 3) * const_mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, h_in + 2, w_in + 4) * const_mask_4d(m, c, 2, 4);
                acc += in_4d(b, c, h_in + 3, w_in + 0) * const_mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, h_in + 3, w_in + 1) * const_mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, h_in + 3, w_in + 2) * const_mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, h_in + 3, w_in + 3) * const_mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, h_in + 3, w_in + 4) * const_mask_4d(m, c, 3, 4);
                acc += in_4d(b, c, h_in + 4, w_in + 0) * const_mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, h_in + 4, w_in + 1) * const_mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, h_in + 4, w_in + 2) * const_mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, h_in + 4, w_in + 3) * const_mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, h_in + 4, w_in + 4) * const_mask_4d(m, c, 4, 4);
            } else if (K == 7){
                acc += in_4d(b, c, h_in + 0, w_in + 0) * const_mask_4d(m, c, 0, 0);
                acc += in_4d(b, c, h_in + 0, w_in + 1) * const_mask_4d(m, c, 0, 1);
                acc += in_4d(b, c, h_in + 0, w_in + 2) * const_mask_4d(m, c, 0, 2);
                acc += in_4d(b, c, h_in + 0, w_in + 3) * const_mask_4d(m, c, 0, 3);
                acc += in_4d(b, c, h_in + 0, w_in + 4) * const_mask_4d(m, c, 0, 4);
                acc += in_4d(b, c, h_in + 0, w_in + 5) * const_mask_4d(m, c, 0, 5);
                acc += in_4d(b, c, h_in + 0, w_in + 6) * const_mask_4d(m, c, 0, 6);
                acc += in_4d(b, c, h_in + 1, w_in + 0) * const_mask_4d(m, c, 1, 0);
                acc += in_4d(b, c, h_in + 1, w_in + 1) * const_mask_4d(m, c, 1, 1);
                acc += in_4d(b, c, h_in + 1, w_in + 2) * const_mask_4d(m, c, 1, 2);
                acc += in_4d(b, c, h_in + 1, w_in + 3) * const_mask_4d(m, c, 1, 3);
                acc += in_4d(b, c, h_in + 1, w_in + 4) * const_mask_4d(m, c, 1, 4);
                acc += in_4d(b, c, h_in + 1, w_in + 5) * const_mask_4d(m, c, 1, 5);
                acc += in_4d(b, c, h_in + 1, w_in + 6) * const_mask_4d(m, c, 1, 6);
                acc += in_4d(b, c, h_in + 2, w_in + 0) * const_mask_4d(m, c, 2, 0);
                acc += in_4d(b, c, h_in + 2, w_in + 1) * const_mask_4d(m, c, 2, 1);
                acc += in_4d(b, c, h_in + 2, w_in + 2) * const_mask_4d(m, c, 2, 2);
                acc += in_4d(b, c, h_in + 2, w_in + 3) * const_mask_4d(m, c, 2, 3);
                acc += in_4d(b, c, h_in + 2, w_in + 4) * const_mask_4d(m, c, 2, 4);
                acc += in_4d(b, c, h_in + 2, w_in + 5) * const_mask_4d(m, c, 2, 5);
                acc += in_4d(b, c, h_in + 2, w_in + 6) * const_mask_4d(m, c, 2, 6);
                acc += in_4d(b, c, h_in + 3, w_in + 0) * const_mask_4d(m, c, 3, 0);
                acc += in_4d(b, c, h_in + 3, w_in + 1) * const_mask_4d(m, c, 3, 1);
                acc += in_4d(b, c, h_in + 3, w_in + 2) * const_mask_4d(m, c, 3, 2);
                acc += in_4d(b, c, h_in + 3, w_in + 3) * const_mask_4d(m, c, 3, 3);
                acc += in_4d(b, c, h_in + 3, w_in + 4) * const_mask_4d(m, c, 3, 4);
                acc += in_4d(b, c, h_in + 3, w_in + 5) * const_mask_4d(m, c, 3, 5);
                acc += in_4d(b, c, h_in + 3, w_in + 6) * const_mask_4d(m, c, 3, 6);
                acc += in_4d(b, c, h_in + 4, w_in + 0) * const_mask_4d(m, c, 4, 0);
                acc += in_4d(b, c, h_in + 4, w_in + 1) * const_mask_4d(m, c, 4, 1);
                acc += in_4d(b, c, h_in + 4, w_in + 2) * const_mask_4d(m, c, 4, 2);
                acc += in_4d(b, c, h_in + 4, w_in + 3) * const_mask_4d(m, c, 4, 3);
                acc += in_4d(b, c, h_in + 4, w_in + 4) * const_mask_4d(m, c, 4, 4);
                acc += in_4d(b, c, h_in + 4, w_in + 5) * const_mask_4d(m, c, 4, 5);
                acc += in_4d(b, c, h_in + 4, w_in + 6) * const_mask_4d(m, c, 4, 6);
                acc += in_4d(b, c, h_in + 5, w_in + 0) * const_mask_4d(m, c, 5, 0);
                acc += in_4d(b, c, h_in + 5, w_in + 1) * const_mask_4d(m, c, 5, 1);
                acc += in_4d(b, c, h_in + 5, w_in + 2) * const_mask_4d(m, c, 5, 2);
                acc += in_4d(b, c, h_in + 5, w_in + 3) * const_mask_4d(m, c, 5, 3);
                acc += in_4d(b, c, h_in + 5, w_in + 4) * const_mask_4d(m, c, 5, 4);
                acc += in_4d(b, c, h_in + 5, w_in + 5) * const_mask_4d(m, c, 5, 5);
                acc += in_4d(b, c, h_in + 5, w_in + 6) * const_mask_4d(m, c, 5, 6);
                acc += in_4d(b, c, h_in + 6, w_in + 0) * const_mask_4d(m, c, 6, 0);
                acc += in_4d(b, c, h_in + 6, w_in + 1) * const_mask_4d(m, c, 6, 1);
                acc += in_4d(b, c, h_in + 6, w_in + 2) * const_mask_4d(m, c, 6, 2);
                acc += in_4d(b, c, h_in + 6, w_in + 3) * const_mask_4d(m, c, 6, 3);
                acc += in_4d(b, c, h_in + 6, w_in + 4) * const_mask_4d(m, c, 6, 4);
                acc += in_4d(b, c, h_in + 6, w_in + 5) * const_mask_4d(m, c, 6, 5);
                acc += in_4d(b, c, h_in + 6, w_in + 6) * const_mask_4d(m, c, 6, 6);
            } else {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        int h_in = h_out * S + p;
                        int w_in = w_out * S + q;
                        if (h_in < H && w_in < W) {
                            acc += in_4d(b, c, h_in, w_in) * const_mask_4d(m, c, p, q);
                        }
                    }
                }
            }
        }
        out_4d(b, m, h_out, w_out) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

    
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    size_t input_size = B * C * H * W * sizeof(float);
    size_t output_size = B * M * H_out * W_out * sizeof(float);
    size_t mask_size = M * C * K * K * sizeof(float);

    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);
    // cudaMalloc(device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mask, host_mask, mask_size);


    cudaMemset(*device_output_ptr, 0, output_size);
}



__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    int W_grid = ceil(W_out * 1.0 / BLOCK_SIZE);
    int H_grid = ceil(H_out * 1.0 / BLOCK_SIZE);

    dim3 gridDim(B, M, W_grid * H_grid);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    // size_t shared_mem_size = K * K * C * sizeof(float);
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();
}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    size_t output_size = B * M * H_out * W_out * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}



__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

