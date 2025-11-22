#include "utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32


template<int M, int N, int K, int LDA, int LDB, int LDC, int NUM_THREADS>
__device__ void mma(half *__restrict A_s, half *__restrict  B_s, float *__restrict C_s) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  for (int m = 0; m < M; m+= MMA_M * NUM_WARPS) {
    for (int n = 0; n < N; n+= MMA_N) {
      uint32_t C_frag[4] = {0};
      for (int k = 0; k < K; k += MMA_K) {
        uint32_t A_frag[4];
        {
          int local_row = (tidx % 8) + (tidx / 16) * 8;
          int row = m + widx * MMA_M + local_row;
          int col = k + ((tidx / 8) % 2) * 8;
          half *A_row = A_s + (row * LDA) + col;
          uint32_t A_addr = __cvta_generic_to_shared(A_row);
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
                       : "r"(A_addr));
        }
        uint32_t B_frag[2];
        {
          int row = k + (tidx % 16);
          int col = n;
          half *B_row = B_s + (row * LDB) + col;

          uint32_t B_addr = __cvta_generic_to_shared(B_row);
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                       : "=r"(B_frag[0]), "=r"(B_frag[1])
                       : "r"(B_addr));
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "     // D
                     "{%4, %5, %6, %7}, "     // A
                     "{%8, %9}, "             // B
                     "{%0, %1, %2, %3};\n"    // C
                     : "+r"(C_frag[0]), "+r"(C_frag[1]), 
                       "+r"(C_frag[2]), "+r"(C_frag[3])
                     : "r"(A_frag[0]), "r"(A_frag[2]), "r"(A_frag[1]), "r"(A_frag[3]), // Permuted A
                       "r"(B_frag[0]), "r"(B_frag[1]));
      }
    {
      int local_row0 = tidx / 4;
      int row0 = m + widx * MMA_M + local_row0;
      int row1 = row0 + 8;
      int col0 = n + (tidx % 4) * 2;
      int col1 = col0 + 1;
      C_s[row0 * LDC + col0] += reinterpret_cast<float*>(C_frag)[0];
      C_s[row0 * LDC + col1] += reinterpret_cast<float*>(C_frag)[1];
      C_s[row1 * LDC + col0] += reinterpret_cast<float*>(C_frag)[2];
      C_s[row1 * LDC + col1] += reinterpret_cast<float*>(C_frag)[3];
    }
    }
  }
}

template<int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
__global__ void matmul_kernel(half *A, half *B, float *C, int M, int N, int K) {
  constexpr int GROUP_SIZE = 8;
  constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int VEC_SIZE_HALF = 8;  // int4 = 8 halves (16 bytes)
  constexpr int VEC_SIZE_FLOAT = 4; // int4 = 4 floats (16 bytes)
  
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE;
  const int LDA = TILE_K + 8;
  const int LDB = TILE_N + 8;
  const int LDC = TILE_N + 4;
  
  __shared__ half A_s[TILE_M * LDA];
  __shared__ half B_s[TILE_K * LDB];
  __shared__ float C_s[TILE_M * LDC];
  
  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;
  
  int4 zero = make_int4(0, 0, 0, 0);
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * VEC_SIZE_FLOAT) { 
      int local_row = mm + widx * NUM_GROUPS + gidx;
      int local_col = nn + lidx * VEC_SIZE_FLOAT;
      *reinterpret_cast<int4*>(&C_s[local_row * LDC + local_col]) = zero;
    }
  }
  __syncthreads();
  
  for (int k = 0; k < K; k += TILE_K) {
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE * VEC_SIZE_HALF) { 
        int local_row = mm + widx * NUM_GROUPS + gidx;
        int local_col = kk + lidx * VEC_SIZE_HALF;
        int global_row = m + local_row;
        int global_col = k + local_col;
        
        *reinterpret_cast<int4*>(&A_s[local_row * LDA + local_col]) =
            *reinterpret_cast<const int4*>(&A[global_row * K + global_col]);
      }
    }
    
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * VEC_SIZE_HALF) {
        int local_row = kk + widx * NUM_GROUPS + gidx;
        int local_col = nn + lidx * VEC_SIZE_HALF;
        int global_row = k + local_row;
        int global_col = n + local_col;
        
        *reinterpret_cast<int4*>(&B_s[local_row * LDB + local_col]) =
            *reinterpret_cast<const int4*>(&B[global_row * N + global_col]);
      }
    }
    __syncthreads();
    
    mma<TILE_M, TILE_N, TILE_K, LDA, LDB, LDC, NUM_THREADS>(A_s, B_s, C_s);
    __syncthreads();
  }
  
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * VEC_SIZE_FLOAT) {
      int local_row = mm + widx * NUM_GROUPS + gidx;
      int local_col = nn + lidx * VEC_SIZE_FLOAT;
      int global_row = m + local_row;
      int global_col = n + local_col;
      
      *reinterpret_cast<int4*>(&C[global_row * N + global_col]) =
          *reinterpret_cast<const int4*>(&C_s[local_row * LDC + local_col]);
    }
  }
}


template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);
  matmul_kernel<TILE_M, TILE_N, TILE_K, NUM_THREADS><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

// bank conflics before:
//   Section: Command line profiler metrics
//   -------------------------------------------------------- ----------- ------------
//   Metric Name                                              Metric Unit Metric Value
//   -------------------------------------------------------- ----------- ------------
//   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum               25.698.261
//   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    1.048.576
//   -------------------------------------------------------- ----------- ------------
//
// bank conflicts after:
// Section: Command line profiler metrics
// -------------------------------------------------------- ----------- ------------
// Metric Name                                              Metric Unit Metric Value
// -------------------------------------------------------- ----------- ------------
// l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  524.288
// l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    1.572.864
// -------------------------------------------------------- ----------- ------------
// cublas has 0 bank conflicts !
// ampere_s16816gemm_fp16_256x128_ldg8_stages_32x3_nn (4, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
//   Section: Command line profiler metrics
//   -------------------------------------------------------- ----------- ------------
//   Metric Name                                              Metric Unit Metric Value
//   -------------------------------------------------------- ----------- ------------
//   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
//   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      786.432
//   -------------------------------------------------------- ----------- ------------

// Results match!
//
// ========== Benchmark Results ==========
// Matrix dimensions: M=1024, N=1024, K=1024
// Iterations: 100
//
// Custom Kernel:  0.1207 ms  (17.79 TFLOPS)
// cuBLAS:         0.0816 ms  (26.32 TFLOPS)
// Efficiency:     67.59%
// Speedup:        0.68x (slower)
// =======================================
int main() {
  constexpr int TILE_M = 128, TILE_N = 64, TILE_K = 64;
  constexpr int NUM_THREADS = 256;
  const int M = 1024, N = 1024, K = 1024;
  validate_matmul(matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>, M, N, K);
  benchmark_matmul(matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>, M, N, K);
}
