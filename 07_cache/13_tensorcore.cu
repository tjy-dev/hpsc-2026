#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__device__ __forceinline__ unsigned shared_addr(const void *ptr) {
  unsigned long long addr;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(ptr));
  return static_cast<unsigned>(addr);
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
  unsigned dst_addr = shared_addr(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group 0;");
}

__device__ void async_load_tile(int dim_m, int dim_k, int k,
                                int offset_a_m, int offset_b_n, int tid, int stride,
                                const float *d_a, const float *d_b,
                                float4 stage_a[16][32], float4 stage_b[128][4]) {
  for (int idx = tid; idx < 16 * 32; idx += stride) {
    int j = idx / 32;
    int i4 = idx % 32;
    const float4 *src = reinterpret_cast<const float4 *>(&d_a[(k + j) * dim_m + offset_a_m + i4 * 4]);
    cp_async_16(&stage_a[j][i4], src);
  }
  for (int idx = tid; idx < 128 * 4; idx += stride) {
    int col = idx / 4;
    int j4 = idx % 4;
    const float4 *src = reinterpret_cast<const float4 *>(&d_b[(offset_b_n + col) * dim_k + k + j4 * 4]);
    cp_async_16(&stage_b[col][j4], src);
  }
}

__device__ void convert_tile(int tid,
                             int stride,
                             const float4 stage_a[16][32], const float4 stage_b[128][4],
                             half block_a[16][128], half block_b[128][16]) {
  for (int idx = tid; idx < 16 * 32; idx += stride) {
    int j = idx / 32;
    int i4 = (idx % 32) * 4;
    float4 a4 = stage_a[j][idx % 32];
    *reinterpret_cast<half2 *>(&block_a[j][i4 + 0]) = __floats2half2_rn(a4.x, a4.y);
    *reinterpret_cast<half2 *>(&block_a[j][i4 + 2]) = __floats2half2_rn(a4.z, a4.w);
  }
  for (int idx = tid; idx < 128 * 4; idx += stride) {
    int col = idx / 4;
    int j = (idx % 4) * 4;
    float4 b4 = stage_b[col][idx % 4];
    *reinterpret_cast<half2 *>(&block_b[col][j + 0]) = __floats2half2_rn(b4.x, b4.y);
    *reinterpret_cast<half2 *>(&block_b[col][j + 2]) = __floats2half2_rn(b4.z, b4.w);
  }
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_a_m = 128 * blockIdx.x;
  int offset_b_n = 128 * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ float4 stage_a[2][16][32];
  __shared__ float4 stage_b[2][128][4];
  __shared__ half block_a[2][16][128];
  __shared__ half block_b[2][128][16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][8];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 8; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  async_load_tile(dim_m, dim_k, 0, offset_a_m, offset_b_n, i, blockDim.x,
                  d_a, d_b, stage_a[0], stage_b[0]);
  cp_async_commit();
  cp_async_wait();
  __syncthreads();
  convert_tile(i, blockDim.x, stage_a[0], stage_b[0], block_a[0], block_b[0]);
  __syncthreads();

  for (int k = 0; k < dim_k; k += 16) {
    int stage = (k / 16) & 1;
    int next_stage = stage ^ 1;
    bool has_next = k + 16 < dim_k;
    if (has_next) {
      async_load_tile(dim_m, dim_k, k + 16, offset_a_m, offset_b_n, i, blockDim.x,
                      d_a, d_b, stage_a[next_stage], stage_b[next_stage]);
      cp_async_commit();
    }
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[stage][0][row_tile * 16], 128);
      for (int c = 0; c < 8; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[stage][c * 16][0], 16);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
    if (has_next) {
      cp_async_wait();
      __syncthreads();
      convert_tile(i, blockDim.x, stage_a[next_stage], stage_b[next_stage],
                   block_a[next_stage], block_b[next_stage]);
      __syncthreads();
    }
  }

  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 8; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  dim3 block = dim3(128);
  dim3 grid = dim3((m+128-1)/128, (n+128-1)/128);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
