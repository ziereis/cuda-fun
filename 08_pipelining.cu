#include "utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32


template<int M, int N, int K, int LDA, int LDB, int NUM_THREADS>
__device__ void mma(half *__restrict A_s, half *__restrict  B_s, uint32_t *__restrict C_frag) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int C_M = M / (MMA_M * NUM_WARPS);
  constexpr int C_N = N / MMA_N;
  for (int k = 0; k < K; k += MMA_K) {
    for (int m = 0; m < C_M; m++) {
      uint32_t A_frag[4];
      {
        int local_row = (tidx % 8) + (tidx / 16) * 8;
        int row = (m * NUM_WARPS + widx) * MMA_M + local_row;
        int col = k + ((tidx / 8) % 2) * 8;
        half *A_row = A_s + (row * LDA) + col;
        uint32_t A_addr = __cvta_generic_to_shared(A_row);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
                     : "r"(A_addr));
      }
      for (int n = 0; n < C_N; n++) {
        uint32_t B_frag[2];
        {
          int row = k + (tidx % 16);
          int col = n * MMA_N;
          half *B_row = B_s + (row * LDB) + col;
          uint32_t B_addr = __cvta_generic_to_shared(B_row);
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                       : "=r"(B_frag[0]), "=r"(B_frag[1])
                       : "r"(B_addr));
        }
        int c0 = m * (C_N * 4) + n * 4;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "     // D
                     "{%4, %5, %6, %7}, "     // A
                     "{%8, %9}, "             // B
                     "{%0, %1, %2, %3};\n"    // C
                     : "+r"(C_frag[c0]), "+r"(C_frag[c0 + 1]), 
                       "+r"(C_frag[c0 + 2]), "+r"(C_frag[c0 + 3])
                     : "r"(A_frag[0]), "r"(A_frag[2]), "r"(A_frag[1]), "r"(A_frag[3]), // Permuted A
                       "r"(B_frag[0]), "r"(B_frag[1]));
      }
    }
  }
}

template<int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS, int NUM_STAGES>
__global__ void matmul_kernel(half *A, half *B, float *C, int M, int N, int K) {
  constexpr int GROUP_SIZE = 8;
  constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int VEC_SIZE_HALF = 8;  // int4 = 8 halves (16 bytes)
  
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int C_M = TILE_M / (MMA_M * NUM_WARPS);
  constexpr int C_N = TILE_N / MMA_N;
  const int LDA = TILE_K + 8;
  const int LDB = TILE_N + 8;
  
  constexpr int A_size = TILE_M * LDA * 2;  
  constexpr int B_size = TILE_K * LDB * 2;  
  constexpr int smem_size = (A_size + B_size) * NUM_STAGES;
  __shared__ __align__(16) uint8_t smem[smem_size];
  half *A_s = reinterpret_cast<half*>(smem);
  half *B_s = reinterpret_cast<half*>(smem + A_size * NUM_STAGES);
  uint32_t C_frag[C_M][C_N][4] = {0};
  
  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;

  auto load_stage = [&](int k, int stage_idx) {
    half* A_dst = A_s + (stage_idx * TILE_M * LDA);
    half* B_dst = B_s + (stage_idx * TILE_K * LDB);

    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE * VEC_SIZE_HALF) { 
        int local_row = mm + widx * NUM_GROUPS + gidx;
        int local_col = kk + lidx * VEC_SIZE_HALF;
        int global_row = m + local_row;
        int global_col = k + local_col;
        uint32_t smem_addr = __cvta_generic_to_shared(
            A_dst + local_row * LDA + local_col);
        const half* gmem_addr = A  + global_row * K + global_col;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
      }
    }
    
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * VEC_SIZE_HALF) {
        int local_row = kk + widx * NUM_GROUPS + gidx;
        int local_col = nn + lidx * VEC_SIZE_HALF;
        int global_row = k + local_row;
        int global_col = n + local_col;
        uint32_t smem_addr = __cvta_generic_to_shared(
            B_dst + local_row * LDB + local_col);
        const half* gmem_addr = B  + global_row * N + global_col;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
      }
    }
  };
  
  for (int s = 0; s < NUM_STAGES - 1; ++s) {
    load_stage(s * TILE_K, s);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int k = 0; k < K; k += TILE_K) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
    __syncthreads();
    int next_k = k + (NUM_STAGES - 1) * TILE_K;
    int write_stage = (next_k / TILE_K) % NUM_STAGES;

    if (next_k < K) {
      load_stage(next_k, write_stage);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    int read_stage = (k / TILE_K) % NUM_STAGES;
    mma<TILE_M, TILE_N, TILE_K, LDA, LDB, NUM_THREADS>(
              A_s + (read_stage * TILE_M * LDA), 
              B_s + (read_stage * TILE_K * LDB),  
              reinterpret_cast<uint32_t*>(C_frag));
    __syncthreads();
  }
  
  int32_t* C_int = reinterpret_cast<int32_t*>(C);
  for (int mm = 0; mm < C_M; mm++) {
    for (int nn = 0; nn < C_N; nn++) {
      int local_row0 = tidx / 4;
      int row0 = widx * MMA_M + local_row0;
      int row1 = row0 + 8;
      int col0 = (tidx % 4) * 2;
      int col1 = col0 + 1;
      int global_row_base = m + mm * MMA_M * NUM_WARPS;
      int global_col_base = n + nn * MMA_N;
      C_int[(global_row_base + row0) * N + global_col_base + col0] = C_frag[mm][nn][0];
      C_int[(global_row_base + row0) * N + global_col_base + col1] = C_frag[mm][nn][1];
      C_int[(global_row_base + row1) * N + global_col_base + col0] = C_frag[mm][nn][2];
      C_int[(global_row_base + row1) * N + global_col_base + col1] = C_frag[mm][nn][3];
    }
  }
}


template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS, int NUM_STAGES>
void matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);
  matmul_kernel<TILE_M, TILE_N, TILE_K, NUM_THREADS, NUM_STAGES><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

// with NUM_STAGES=2 TILE_N=128
// ========== Benchmark Results ==========
// Matrix dimensions: M=4096, N=4096, K=4096
// Iterations: 100
//
// Custom Kernel:  3.9595 ms  (34.71 TFLOPS)
// cuBLAS:         3.1815 ms  (43.20 TFLOPS)
// Efficiency:     80.35%
// Speedup:        0.80x (slower)
// =======================================
// ========== Benchmark Results ==========
// Matrix dimensions: M=1024, N=1024, K=1024
// Iterations: 100
//
// Custom Kernel:  0.0845 ms  (25.42 TFLOPS)
// cuBLAS:         0.0816 ms  (26.31 TFLOPS)
// Efficiency:     96.62%
// Speedup:        0.97x (slower)
// =======================================
int main() {
  constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 64;
  constexpr int NUM_THREADS = 256;
  const int M = 4*1024, N = 4*1024, K = 4*1024;
  // validate_matmul(matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS, 3>, M, N, K, 1e-2);
  benchmark_matmul(matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS, 2>, M, N, K);
}
