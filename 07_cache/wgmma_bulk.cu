#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda/ptx>
#include <chrono>
using namespace std;

__device__ __forceinline__ uint64_t insert_bit(uint32_t start_bit, uint64_t target, uint64_t val) {
  return target | (val << start_bit);
}

template <class PointerType>
__device__ uint64_t make_wgmma_desc(PointerType smem_ptr,
                                    int stride_byte_offset,
                                    int leading_byte_offset,
                                    int swizzle_mode = 0) {
  uint64_t desc = 0;
  uint32_t base_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t start_address = base_ptr >> 4;
  desc = insert_bit(62, desc, static_cast<uint64_t>(swizzle_mode));
  desc = insert_bit(49, desc, 0);
  desc = insert_bit(32, desc, static_cast<uint64_t>(stride_byte_offset));
  desc = insert_bit(16, desc, static_cast<uint64_t>(leading_byte_offset));
  desc = insert_bit(0, desc, start_address);
  return desc;
}

__device__ __forceinline__ int swizzle_128b_index(int half_index) {
  int byte_index = half_index * static_cast<int>(sizeof(half));
  byte_index ^= (byte_index & 0x380) >> 3;
  return byte_index / static_cast<int>(sizeof(half));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n128k16(uint64_t desc_a, uint64_t desc_b, float acc[64]) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
      "%64, %65, 1, 1, 1, 0, 0;\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
        "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
        "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]),
        "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
        "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
        "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
        "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
        "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
        "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
        "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
        "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
        "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
        "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
        "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
        "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
        "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
      : "l"(desc_a), "l"(desc_b)
      : "memory");
}

__global__ void prepack_a_tiles(int dim_m, int dim_k, const float *d_a, half *packed_a) {
  int tile_m = blockIdx.x;
  int tile_k = blockIdx.y;
  int tid = threadIdx.x;
  int offset_a_m = tile_m * 128;
  int kk = tile_k * 16;
  half *tile = packed_a + (tile_m * gridDim.y + tile_k) * 8192;

  for (int idx = tid; idx < 128 * 16; idx += blockDim.x) {
    int row = idx / 16;
    int k_local = idx % 16;
    int k_in = k_local % 8;
    int k_core = k_local / 8;
    int row_in = row % 8;
    int row_group = row / 8;
    int smem_idx = swizzle_128b_index(row_group * 512 + row_in * 64 + k_core * 8 + k_in);
    int global_row = offset_a_m + row;
    tile[smem_idx] = (global_row < dim_m) ? __float2half(d_a[(kk + k_local) * dim_m + global_row]) : __float2half(0.0f);
  }
}

__global__ void prepack_b_tiles(int dim_n, int dim_k, const float *d_b, half *packed_b) {
  int tile_n = blockIdx.x;
  int tile_k = blockIdx.y;
  int tid = threadIdx.x;
  int offset_b_n = tile_n * 128;
  int kk = tile_k * 16;
  half *tile = packed_b + (tile_n * gridDim.y + tile_k) * 8192;

  for (int idx = tid; idx < 16 * 128; idx += blockDim.x) {
    int k_local = idx / 128;
    int col = idx % 128;
    int k_in = k_local % 8;
    int k_core = k_local / 8;
    int col_in = col % 8;
    int col_group = col / 8;
    int smem_idx = swizzle_128b_index(col_group * 512 + col_in * 64 + k_core * 8 + k_in);
    int global_col = offset_b_n + col;
    tile[smem_idx] = (global_col < dim_n) ? __float2half(d_b[global_col * dim_k + kk + k_local]) : __float2half(0.0f);
  }
}

__device__ void bulk_copy_packed_tiles(const half *src_a, half *dst_a,
                                       const half *src_b, half *dst_b,
                                       uint64_t *bar) {
  namespace ptx = cuda::ptx;
  constexpr uint32_t tile_bytes = 8192 * sizeof(half);
  if (threadIdx.x == 0)
    ptx::mbarrier_init(bar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(ptx::space_cluster, ptx::space_global,
                       dst_a, src_a, tile_bytes, bar);
    ptx::cp_async_bulk(ptx::space_cluster, ptx::space_global,
                       dst_b, src_b, tile_bytes, bar);
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                   ptx::space_shared, bar, 2 * tile_bytes);
  }
  __syncthreads();
  while (!ptx::mbarrier_try_wait_parity(bar, 0))
    ;
  __syncthreads();
}

__global__ void wgmma_kernel(int dim_m, int dim_n, int dim_k,
                            const half *packed_a, const half *packed_b, float *d_c) {
  int offset_a_m = 128 * blockIdx.x;
  int offset_b_n = 128 * blockIdx.y;
  int ktile_count = (dim_k + 15) / 16;

  __shared__ __align__(1024) half block_a[2][8192];
  __shared__ __align__(1024) half block_b[2][8192];
  __shared__ uint64_t bulk_bar;

  float acc[2][64];
  for (int r = 0; r < 2; r++)
    for (int x = 0; x < 64; x++)
      acc[r][x] = 0.0f;

  uint64_t desc_a0[2] = {
      make_wgmma_desc(block_a[0], 64, 1, 1),
      make_wgmma_desc(block_a[1], 64, 1, 1)};
  uint64_t desc_a1[2] = {
      make_wgmma_desc(&block_a[0][4096], 64, 1, 1),
      make_wgmma_desc(&block_a[1][4096], 64, 1, 1)};
  uint64_t desc_b[2] = {
      make_wgmma_desc(block_b[0], 64, 1, 1),
      make_wgmma_desc(block_b[1], 64, 1, 1)};

  wgmma_fence();
  int tile_a0 = blockIdx.x * ktile_count;
  int tile_b0 = blockIdx.y * ktile_count;
  const half *src_a0 = packed_a + tile_a0 * 8192;
  const half *src_b0 = packed_b + tile_b0 * 8192;
  bulk_copy_packed_tiles(src_a0, block_a[0], src_b0, block_b[0], &bulk_bar);

  for (int ktile = 0; ktile < ktile_count; ktile++) {
    int stage = ktile & 1;
    int next_stage = stage ^ 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    wgmma_fence();
    wgmma_m64n128k16(desc_a0[stage], desc_b[stage], acc[0]);
    wgmma_m64n128k16(desc_a1[stage], desc_b[stage], acc[1]);
    wgmma_commit();
    if (ktile + 1 < ktile_count) {
      int tile_a = blockIdx.x * ktile_count + ktile + 1;
      int tile_b = blockIdx.y * ktile_count + ktile + 1;
      const half *src_a = packed_a + tile_a * 8192;
      const half *src_b = packed_b + tile_b * 8192;
      bulk_copy_packed_tiles(src_a, block_a[next_stage], src_b, block_b[next_stage], &bulk_bar);
    }
    wgmma_wait();
    __syncthreads();
  }

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32;
  int row0 = warp * 16 + lane / 4;
  int row1 = row0 + 8;
  int col_pair = 2 * (lane % 4);

  for (int ngrp = 0; ngrp < 16; ngrp++) {
    int col = ngrp * 8 + col_pair;
    for (int rb = 0; rb < 2; rb++) {
      int row_base = rb * 64;
      if (offset_a_m + row_base + row0 < dim_m && offset_b_n + col + 1 < dim_n) {
        d_c[(offset_b_n + col + 0) * dim_m + offset_a_m + row_base + row0] = acc[rb][ngrp * 4 + 0];
        d_c[(offset_b_n + col + 1) * dim_m + offset_a_m + row_base + row0] = acc[rb][ngrp * 4 + 1];
      }
      if (offset_a_m + row_base + row1 < dim_m && offset_b_n + col + 1 < dim_n) {
        d_c[(offset_b_n + col + 0) * dim_m + offset_a_m + row_base + row1] = acc[rb][ngrp * 4 + 2];
        d_c[(offset_b_n + col + 1) * dim_m + offset_a_m + row_base + row1] = acc[rb][ngrp * 4 + 3];
      }
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
  half *Ap, *Bp;
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

  int mtile_count = (m + 127) / 128;
  int ntile_count = (n + 127) / 128;
  int ktile_count = (k + 15) / 16;
  cudaMalloc(&Ap, int64_t(mtile_count) * ktile_count * 8192 * sizeof(half));
  cudaMalloc(&Bp, int64_t(ntile_count) * ktile_count * 8192 * sizeof(half));
  tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    prepack_a_tiles<<<dim3(mtile_count, ktile_count), 256>>>(m, k, A, Ap);
    prepack_b_tiles<<<dim3(ntile_count, ktile_count), 256>>>(n, k, B, Bp);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tprepack = chrono::duration<double>(toc - tic).count() / Nt;

  dim3 block = dim3(128);
  dim3 grid = dim3(mtile_count, ntile_count);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    wgmma_kernel<<< grid, block >>>(m, n, k, Ap, Bp, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double twgmma = chrono::duration<double>(toc - tic).count() / Nt;
  double wgmma_flops = double(num_flops) / twgmma / 1.0e9;
  double total_flops = double(num_flops) / (tprepack + twgmma) / 1.0e9;
  printf("CUBLAS: %.2f Gflops, WGMMA: %.2f Gflops, total: %.2f Gflops\n",
         cublas_flops, wgmma_flops, total_flops);
  printf("prepack: %.6f s, wgmma: %.6f s\n", tprepack, twgmma);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(Ap);
  cudaFree(Bp);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
