#include <cstdio>

__global__ void init(int n, float *x, float num) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride) {
    x[i] = num;
  }
}

__global__ void add(int n, float *x, float *y, float *z) {

  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride) {
    z[i] = x[i] + y[i];
  }
}

int main() {
  int deviceId;
  int numberOfSMs;
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("device id: %d\n", deviceId);
  printf("sm num: %d\n", numberOfSMs);

  int N = 1<<20;
  printf("%d\n", N);

  float *x, *y, *z;
  size_t size = N * sizeof(float);
  cudaMallocManaged(&x, size);
  cudaMallocManaged(&y, size);
  cudaMallocManaged(&z, size);

  int blockSize = 1024;
  int numBlocks = 256 * numberOfSMs;
  printf("block num: %d\n", numBlocks);
  printf("thread num: %d\n", blockSize);

  cudaStream_t init_stream_x;  
  cudaStreamCreate(&init_stream_x);  
  init<<<numBlocks, blockSize, 0, init_stream_x>>>(N, x, 1.0f);

  cudaStream_t init_stream_y;
  cudaStreamCreate(&init_stream_y);
  init<<<numBlocks, blockSize, 0, init_stream_y>>>(N, y, 2.0f);

  cudaStream_t init_stream_z;
  cudaStreamCreate(&init_stream_z);
  init<<<numBlocks, blockSize, 0, init_stream_z>>>(N, z, 0.0f);

  add<<<numBlocks, blockSize>>>(N, x, y, z);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++){
    if (z[i] != 3.0f) {
      printf("Error: result verification failed at element %d!\n", i);  
      exit(EXIT_FAILURE);  
    }
  }
  printf("Test PASSED!\n");
  
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  
  return 0;
}