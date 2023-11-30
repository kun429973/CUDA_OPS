#include <stdio.h>
#include <sys/time.h>


/*
二，针对01_reduceNeighbored.cu 中 if((tid %(2 * stride)) == 0) 线程束分化严重，
优化：使用 int index = 2 * stride * tid;防止线程束的分化，线程0->data[0],线程1->data[2] ...
512个线程，16个线程束，仅使用前8个线程束就可以了。

GPU程序执行时间: 13.580000 ms    
*/

// CUDA核函数，执行并行规约操作
__global__ void reduceNeighboredLess(int* g_odata, int* g_idata, unsigned int n) {

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return;
    for(int stride=1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if(index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
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
    int *h_odata = (int *)malloc(grid.x*sizeof(int));
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
    reduceNeighboredLess<<<grid, block>>>(d_odata, d_idata, size);
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&time_stop, NULL);
    time_pass = ((double)time_stop.tv_sec -(double)time_start.tv_sec)*1000 + ((double)time_stop.tv_usec-(double)time_start.tv_usec)/1000;
    printf("GPU程序执行时间: %lf ms\n", time_pass);

    int gpu_sum = 0;
    for(int i=0; i<grid.x; i++) {
        gpu_sum += h_odata[i];
    }

    if(cpu_sum == gpu_sum) {
        printf("CPU&GPU have same result!!! \n");
    } else {
        printf("CPU&GPU result are diffrent!!! \n");
    }

    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);


    return 0;
}
