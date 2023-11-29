#include <stdio.h>
#include <sys/time.h>

/*
1，合并访存（Coalesced Memory Access）：
   合并访存是指当线程束（thread warp）中的线程访问连续内存地址时，这些访问可以被合并为一个或多个内存事务。
 线程束是CUDA中的一个基本执行单元，通常包含32个线程（在NVIDIA GPU上的典型架构中）。
2，非合并访存是指线程束中的线程访问不连续的内存地址，导致内存访问无法被合并。
3，性能优化建议：
合并访存：
确保线程束中的线程访问的内存地址是连续的。
尽量使访存模式符合硬件的合并访存规则。
使用一维数组时，尽量使线程束内的线程访问相邻元素。

非合并访存：
考虑重组数据结构以便实现合并访存。
尽量避免非连续的内存访问模式，特别是在计算密集型任务中。
*/


//下面是非合并访存和合并访存的例子，但最终时间都差不多，和NVIDIA视频中讲解不一样。(存疑)

// 非合并访存1
template<int ElementPerBlock = 4096>
__global__ void stride_access(float* src, float* dst) {
    int blockOffset = blockIdx.x * ElementPerBlock;
    int tid = threadIdx.x;

    #pragma unroll
    for(int i=0; i<4; i++) {
        dst[blockOffset + tid * 4 + i] = src[blockOffset + tid *4 + i];
    }

}

// 非合并访存2
template<int ElementPerBlock = 4096>
__global__ void stride_access_float4(float* src, float* dst) {
    int blockOffset = blockIdx.x * ElementPerBlock;
    int tid = threadIdx.x;

    float4* dst_ptr = (float4*)(dst + blockOffset + tid * 4);
    float4* src_ptr = (float4*)(src + blockOffset + tid * 4);

    *dst_ptr = *src_ptr;
}

// 合并访存
template<int ElementPerBlock = 4096>
__global__ void normal_access(float* src, float* dst) {
    int blockOffset = blockIdx.x * ElementPerBlock;
    int tid = threadIdx.x;

    #pragma unroll
    for(int i=0; i<4; i++) {
        dst[blockOffset + tid + i * blockDim.x] = src[blockOffset + tid + i * blockDim.x];
    }

}

int main() {

    struct timeval time_start,  time_stop ;
    double time_pass;

    const long arraySize = 400 * 4096;
    const int blockSize = 1024;
    const long numBlocks = (arraySize + blockSize - 1) / blockSize / 4;

    size_t bytes = arraySize * sizeof(float);
    float *h_src = (float *)malloc(bytes);
    float *h_dst = (float *)malloc(bytes);
    for (long i = 0; i < arraySize; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    float* d_src, *d_dst;
    cudaMalloc((void**)&d_src, bytes);
    cudaMalloc((void**)&d_dst, bytes);

    gettimeofday(&time_start, NULL);

    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    
    stride_access_float4<<<numBlocks, blockSize>>>(d_src, d_dst);
    
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    gettimeofday(&time_stop, NULL);

    time_pass = ((double)time_stop.tv_sec -(double)time_start.tv_sec)*1000000 + (double)time_stop.tv_usec-(double)time_start.tv_usec;
    printf("程序执行时间: %lf us\n", time_pass);

    // 验证结果
    for (long i = 0; i < arraySize; ++i) {
        if (h_dst[i] != h_src[i]) {
            printf("Verification failed at index \n");
            break;
        }
    }

    printf("Verification successful!\n");

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}