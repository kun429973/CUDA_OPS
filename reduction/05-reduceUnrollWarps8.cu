#include <stdio.h>
#include <sys/time.h>


/*
1,只使用1个block处理8个数据块的时间为：GPU程序执行时间: 11.749000 ms
2,当每个block中前半部分+后半部分 < 64,即stride < 32时，就可以全部展开：
              0~32 += tid+32
              0~16 += tid+16
              0~8  += tid+8
              0~4  += tid+4
              0~2  += tid+2
              0~1  += tid+1
GPU程序执行时间: 11.704000 ms
*/

// CUDA核函数，执行并行规约操作
__global__ void reduceUnrollWarps8(int* g_odata, int* g_idata, unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(idx + 7*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int b1 = g_idata[idx + 4*blockDim.x];
        int b2 = g_idata[idx + 5*blockDim.x];
        int b3 = g_idata[idx + 6*blockDim.x];
        int b4 = g_idata[idx + 7*blockDim.x];
    
        g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
    }
    __syncthreads(); //必须加上，不然结果不对。

    for(int stride=blockDim.x / 2; stride > 32; stride >>= 1) {
        if(tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    //unrolling warp
    if(tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size) {
    if(size == 1) return data[0];
    int const stride = size / 2;
    for(int i=0; i<stride; i++) {
        data[i] += data[i + stride];
    }    

    return recursiveReduce(data, stride);
}


int main(int argc, char **argv) {
    
    struct timeval time_start,  time_stop ;
    double time_pass;

    int size = 1<<24;
    int blockSize = 512;
    if(argc > 1) {
        blockSize = atoi(argv[1]);
    }

    dim3 block (blockSize, 1);
    dim3 grid ((size+block.x-1)/block.x, 1);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x/8*sizeof(int));
    int *tmp = (int *)malloc(bytes);

    for(int i=0; i<size; i++) {
        h_idata[i] = (int) (rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, grid.x*sizeof(int));

    //cpu reduction
    gettimeofday(&time_start, NULL);
    int cpu_sum = recursiveReduce(tmp, size);
    gettimeofday(&time_stop, NULL);

    time_pass = ((double)time_stop.tv_sec -(double)time_start.tv_sec)*1000 + ((double)time_stop.tv_usec-(double)time_start.tv_usec)/1000;
    printf("CPU程序执行时间: %lf ms\n", time_pass);

    gettimeofday(&time_start, NULL);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    reduceUnrollWarps8<<<grid.x/8, block.x>>>(d_odata, d_idata, size);
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    gettimeofday(&time_stop, NULL);
    time_pass = ((double)time_stop.tv_sec -(double)time_start.tv_sec)*1000 + ((double)time_stop.tv_usec-(double)time_start.tv_usec)/1000;
    printf("GPU程序执行时间: %lf ms\n", time_pass);

    int gpu_sum = 0;
    for(int i=0; i<grid.x/8; i++) {
        gpu_sum += h_odata[i];
    }

    if(cpu_sum == gpu_sum) {
        printf("CPU&GPU have same result!!! \n");
    } else {
        printf("CPU&GPU result are diffrent!!! \n");
        printf("cpu_sum=%d \n",cpu_sum);
        printf("gpu_sum=%d \n",gpu_sum);

    }

    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);


    return 0;
}
