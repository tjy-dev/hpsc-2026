#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int *bucket, const int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    bucket[i] = 0;
  }
}

__global__ void count_bucket(int *bucket, int *key, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void compute_offset_scan(int *bucket, int *offset, const int range) {
  __shared__ int temp[256];
  int i = threadIdx.x;

  if (i < range) {
    temp[i] = bucket[i];
  }
  __syncthreads();

  for (int j = 1; j < range; j<<=1) {
    int temp_ij = 0;
    if (i >= j && i < range) {
      temp_ij = temp[i - j];
    }
    __syncthreads();
    if (i >= j && i < range) temp[i] += temp_ij;
    __syncthreads();
  }

  if (i < range) {
    if (i == 0) offset[i] = 0;
    else offset[i] = temp[i - 1];
  }
}

__global__ void fill_key(int *bucket, int *offset, int *key, const int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    for (int j = 0; j < bucket[i]; j++) {
      key[offset[i] + j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key;
  int *d_bucket;
  int *d_offset;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));
  cudaMalloc(&d_offset, range * sizeof(int));
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks_n = (n + threads - 1) / threads;
  const int blocks_range = (range + threads - 1) / threads;

  init_bucket<<<blocks_range, threads>>>(d_bucket, range);
  count_bucket<<<blocks_n, threads>>>(d_bucket, d_key, n);
  compute_offset_scan<<<1, threads>>>(d_bucket, d_offset, range);
  fill_key<<<blocks_range, threads>>>(d_bucket, d_offset, d_key, range);

  cudaDeviceSynchronize();
  cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_offset);
}
