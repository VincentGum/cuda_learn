#include <cstdio>

__global__ void add(int n, float *x, float *y, float *z) {

  // 假如当前线程index是9，就会把每个block中的[9]都运算一一遍
  const int index = threadIdx.x + blockDim.x * blockIdx.x; // 当前线程在block中的index
  const int stride = blockDim.x * gridDim.x; // 一个block中的线程数量

  for(int i = index; i < n; i += stride) {
    z[i] = x[i] + y[i];
  }
}

int main() {
  int N = 1<<20;
  printf("%d\n", N);

  float *x, *y, *z;
  // HOST: allocate and init variables，简单的分配内存
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  z = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f; 
    y[i] = 2.0f;
  }

  // HOST: allocate variables in host，简单的分配内存，只是位置在device上
  float *d_x, *d_y, *d_z;

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, N*sizeof(float));

  // HOST: Copy variables to devices，把host的内存内容复制到device
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // DEVICE: Kernel Called
  int blockSize = 256;                               // 一个block中有256个线程
  int numBlocks = (N + blockSize - 1)/blockSize;     // 算出至少需要的线程block
  
  // 此时，线程的编排如：[0, 1,..., 255], [0, 1,..., 255], ...[0, 1,..., 255],总共有numBlocks个
  add<<<numBlocks, blockSize>>>(N, d_x, d_y, d_z);

  // HOST: Copy result from device to host
  cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

  // HOST: check result
  for (int i = 0; i < N; i++){
    if (z[i] != 3.0f) {
      printf("Error: result verification failed at element %d!\n", i);  
      exit(EXIT_FAILURE);  
    }
  }
  printf("Test PASSED!\n");
  
  // HOST: free device mem
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // HOST: free host mem
  free(x);
  free(y);
  free(z);

  return 0;
}