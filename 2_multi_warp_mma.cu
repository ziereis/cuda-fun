#include "utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>


#define WARP_SIZE 32

__device__ void mma(half *__restrict A_s, half *__restrict  B_s, float *__restrict C_s) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int LDA = MMA_K;
  constexpr int LDB = MMA_N;
  constexpr int LDC = MMA_N;
  uint32_t A_frag[4];
  {
    int local_row = (tidx % 8) + (tidx / 16) * 8;
    int row = widx * MMA_M + local_row;
    int col = ((tidx / 8) % 2) * 8;
    half *A_row = A_s + (row * LDA) + col;
    uint32_t A_addr = __cvta_generic_to_shared(A_row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
                 : "r"(A_addr));
  }
  uint32_t B_frag[2];
  {
    int row = tidx % 16;
    int col = 0;
    half *B_row = B_s + (row * LDB) + col;

    uint32_t B_addr = __cvta_generic_to_shared(B_row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(B_frag[0]), "=r"(B_frag[1])
                 : "r"(B_addr));
  }
  uint32_t C_frag[4] = {0};
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "     // D
               "{%4, %5, %6, %7}, "     // A
               "{%8, %9}, "             // B
               "{%0, %1, %2, %3};\n"    // C
               : "+r"(C_frag[0]), "+r"(C_frag[1]), 
                 "+r"(C_frag[2]), "+r"(C_frag[3])
               : "r"(A_frag[0]), "r"(A_frag[2]), "r"(A_frag[1]), "r"(A_frag[3]), // Permuted A
                 "r"(B_frag[0]), "r"(B_frag[1]));
  {
    int local_row0 = tidx / 4;
    int row0 = widx * MMA_M + local_row0;
    int row1 = row0 + 8;
    int col0 = (tidx % 4) * 2;
    int col1 = col0 + 1;
    *reinterpret_cast<uint32_t*>(C_s + row0 * LDC + col0) = C_frag[0];
    *reinterpret_cast<uint32_t*>(C_s + row0 * LDC + col1) = C_frag[1];
    *reinterpret_cast<uint32_t*>(C_s + row1 * LDC + col0) = C_frag[2];
    *reinterpret_cast<uint32_t*>(C_s + row1 * LDC + col1) = C_frag[3];
  }
}


template<int NUM_THREADS>
__global__ void matmul_kernel(half *A, half *B, float *C) {
  const int tidx = threadIdx.x;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int M = 16 * NUM_WARPS;
  constexpr int N = 8;
  constexpr int K = 16;
  __shared__ half A_s[M * K];
  __shared__ half B_s[K * N];
  __shared__ float C_s[M * N];
  
  for (int i = 0; i < M * K; i+= NUM_THREADS){
    A_s[i + tidx] = A[i + tidx];
  }
  for (int i = 0; i < K * N; i+= NUM_THREADS){
    B_s[i + tidx] = B[i + tidx];
  }
  mma(A_s,B_s, C_s);
  for (int i = 0; i < M * N; i+= NUM_THREADS){
    C[i + tidx] = C_s[i + tidx];
  }
}


void matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(1);
  dim3 block_dim(2* WARP_SIZE);
  matmul_kernel<2 * WARP_SIZE><<<grid_dim, block_dim>>>(A, B, C);
}


int main() {
  validate_matmul(matmul, 32, 8, 16);
}
