#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_baseline(const int* input, int* output, size_t n) {
  int sum = 0;
  for(int i = 0; i < n; i++) {
    sum += input[i];
  }
  *output = sum;
}

// void ReduceBySerial(const float* input, float* output, size_t n) {
//   SerialKernel<<<1, 1>>>(intput, output, n);
// }


bool checkResult(int *out, int groundTruth, int n){
    if (*out != groundTruth){
        return false;
    }
    return true;
}

int main() {
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 1;
    int gridSize = 1;
    
    // 分配内存，此处可以优化为UnifiedMemory
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N*sizeof(int));

    int *out = (int *)malloc(gridSize * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(int));

    // 初始化
    for(int i = 0; i < N; i++){
        a[i] = 1;
    }


    // 预期
    int groundTruth = N * 1;
    
    //拷贝数据到GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 Grid(gridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_baseline<<<gridSize, blockSize>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allocated %d blocks, data counts are %d\n", gridSize, N);

    bool is_right = checkResult(out, groundTruth, gridSize);
    if(is_right) {
        printf("RIGHT!\n");
    } else {
        printf("WRONG!\n");
        for(int i = 0; i < gridSize; i++){
            printf("res per block: %lf", out[i]); 
        }
    }

    return 0;
}