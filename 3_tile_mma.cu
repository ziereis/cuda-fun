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
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE;
  const int LDA = TILE_K;
  const int LDB = TILE_N;
  const int LDC = TILE_N;
  __shared__ half A_s[TILE_M * LDA];
  __shared__ half B_s[TILE_K * LDB];
  __shared__ float C_s[TILE_M * LDC];

  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;

  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE) { 
      int local_row = mm + widx * NUM_GROUPS + gidx;
      int local_col = nn + lidx;
      C_s[local_row * LDC + local_col] = 0.0;
    }
  }

  __syncthreads();

  for (int k = 0; k < K; k += TILE_K) {
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE) { 
        int local_row = mm + widx * NUM_GROUPS + gidx;
        int local_col = kk + lidx;
        A_s[local_row * LDA + local_col] =
            A[(m + local_row) * K + k + local_col];
      }
    }
    // same applies here
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE) {
        int local_row = kk + widx * NUM_GROUPS + gidx;
        int local_col = nn + lidx;
        B_s[local_row * LDB + local_col] =
            B[(k + local_row) * N + n + local_col];
      }
    }
    __syncthreads();
    mma<TILE_M, TILE_N, TILE_K, LDA, LDB, LDC, NUM_THREADS>(A_s, B_s, C_s);
    __syncthreads();
  }
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE) {
      int local_row = mm + widx * NUM_GROUPS + gidx;
      int local_col = nn + lidx;
      C[(m + local_row) * N + n + local_col] =
          C_s[local_row * LDC + local_col];
    }
  }
  
}


template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);
  matmul_kernel<TILE_M, TILE_N, TILE_K, NUM_THREADS><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}


int main() {
  validate_matmul(matmul<32, 32, 32, 64>, 64, 64, 64);
}
