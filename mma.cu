#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <sys/types.h>

#define WARP_SIZE 32

__global__ void mma_ldmatrix(half *A, half *B, half *C) {
  const int tidx = threadIdx.x;
  __shared__ half A_s[16 * 16];
  __shared__ half B_s[16 * 8];
  for (int i = 0; i < 16 * 16; i += WARP_SIZE) {
    A_s[i + tidx] = A[i + tidx];
  }
  for (int i = 0; i < 16 * 8; i += WARP_SIZE) {
    B_s[i + tidx] = B[i + tidx];
  }

  __syncthreads();

  half *A_row =
      &A_s[(tidx % 8) * 16 + (tidx / 16) * 16 * 8 + ((tidx / 8) % 2) * 8];
  uint32_t A_addr = __cvta_generic_to_shared(A_row);
  uint32_t A_frag[4];
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
      : "r"(A_addr));

  printf("tidx%d loads B at %d\n", tidx, tidx * 8);
  half *B_row = &B_s[tidx * 8];
  uint32_t B_addr = __cvta_generic_to_shared(B_row);
  uint32_t B_frag[2];
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(B_frag[0]), "=r"(B_frag[1])
      : "r"(B_addr));

  uint32_t C_frag[2] = {0, 0};

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "                       // D (output)
               "{%2, %3, %4, %5}, "               // A
               "{%6, %7}, "                       // B
               "{%0, %1};\n"                      // C (same regs as D)
               : "+r"(C_frag[0]), "+r"(C_frag[1]) // read-write: C in, D out
               : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
                 "r"(B_frag[0]), "r"(B_frag[1]));
  // C is 16x8
  // t0 writes to 8*0+0,8*0+1, and 16*8,16*8+1
  // t1 writes to 8*0 2,3  and 16*8+2,16*8+3
  // t4 writes to 8*1+0,8*1+1 and 16*9+0, 16*9+1

  int row_id = tidx / 4;
  int col_id = tidx % 4;
  // printf("tidx%d writes to %d\n", tidx, row_id * 8 + col_id * 2);
  // printf("tidx%d writes to %d\n", tidx, 8 * (8 + row_id) + col_id * 2);
  *((uint32_t *)(&C[row_id * 8 + col_id * 2])) = C_frag[0];
  *((uint32_t *)(&C[8 * (8 + row_id) + col_id * 2])) = C_frag[1];
}

__global__ void mma(half *A, half *B, half *C) {
  const int tidx = threadIdx.x;
  __shared__ half A_s[16 * 16];
  __shared__ half B_s[16 * 8];
  for (int i = 0; i < 16 * 16; i += WARP_SIZE) {
    A_s[i + tidx] = A[i + tidx];
  }
  for (int i = 0; i < 16 * 8; i += WARP_SIZE) {
    B_s[i + tidx] = B[i + tidx];
  }

  __syncthreads();

  uint32_t A_frag[4];

  // printf("tidx%d load A at: %d\n", tidx, (tidx / 4) * 16 + (tidx % 4) * 2);
  // printf("tidx%d load A at: %d\n", tidx, (tidx / 4) * 16 + 8 + (tidx % 4) *
  // 2); printf("tidx%d load A at: %d\n", tidx,
  //       (tidx / 4) * 16 + 16 * 8 + (tidx % 4) * 2);
  // printf("tidx%d load A at: %d\n", tidx,
  //       (tidx / 4) * 16 + 16 * 8 + 8 + (tidx % 4) * 2);
  A_frag[0] = *(uint32_t *)(&A_s[(tidx / 4) * 16 + (tidx % 4) * 2]);
  A_frag[2] = *(uint32_t *)(&A_s[(tidx / 4) * 16 + 8 + (tidx % 4) * 2]);
  A_frag[1] = *(uint32_t *)(&A_s[(tidx / 4) * 16 + 16 * 8 + (tidx % 4) * 2]);
  A_frag[3] =
      *(uint32_t *)(&A_s[(tidx / 4) * 16 + 16 * 8 + 8 + (tidx % 4) * 2]);

  printf("tidx%d loads B at %d\n", tidx, tidx * 8);
  half *B_row = &B_s[tidx * 8];
  uint32_t B_addr = __cvta_generic_to_shared(B_row);
  uint32_t B_frag[2];
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(B_frag[0]), "=r"(B_frag[1])
      : "r"(B_addr));

  uint32_t C_frag[2] = {0, 0};

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "                       // D (output)
               "{%2, %3, %4, %5}, "               // A
               "{%6, %7}, "                       // B
               "{%0, %1};\n"                      // C (same regs as D)
               : "+r"(C_frag[0]), "+r"(C_frag[1]) // read-write: C in, D out
               : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
                 "r"(B_frag[0]), "r"(B_frag[1]));
  // C is 16x8
  // t0 writes to 8*0+0,8*0+1, and 16*8,16*8+1
  // t1 writes to 8*0 2,3  and 16*8+2,16*8+3
  // t4 writes to 8*1+0,8*1+1 and 16*9+0, 16*9+1

  int row_id = tidx / 4;
  int col_id = tidx % 4;
  // printf("tidx%d writes to %d\n", tidx, row_id * 8 + col_id * 2);
  // printf("tidx%d writes to %d\n", tidx, 8 * (8 + row_id) + col_id * 2);
  *((uint32_t *)(&C[row_id * 8 + col_id * 2])) = C_frag[0];
  *((uint32_t *)(&C[8 * (8 + row_id) + col_id * 2])) = C_frag[1];
}

template <int M, int N>
__device__ void _mma_smem_smem_smem(half *A, half *B, half *C, int K) {
  const int tidx = threadIdx.x;
  constexpr int mma_m = 16;
  constexpr int mma_n = 8;
  constexpr int mma_k = 16;
#pragma unroll
  for (int m = 0; m < M; m += mma_m) {
#pragma unroll
    for (int n = 0; n < N; n += mma_n) {
      // mma_n / 2 because we are allocating 32 bit registers
      uint32_t C_tile[mma_m * (mma_n / 2)] = {0};
      for (int k = 0; k < K; k += mma_k) {
        half *A_row = &A[K + ((tidx % 8) * K + (tidx / 16) * K * 8 +
                              ((tidx / 8) % 2) * 8)];
        uint32_t A_addr = __cvta_generic_to_shared(A_row);
        uint32_t A_frag[4];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
            : "r"(A_addr));

        printf("tidx%d loads B at %d\n", tidx, tidx * 8);
        half *B_row = &B[K + (tidx * K)];
        uint32_t B_addr = __cvta_generic_to_shared(B_row);
        uint32_t B_frag[2];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(B_frag[0]), "=r"(B_frag[1])
            : "r"(B_addr));
        uint32_t C_frag[2] = {0, 0};

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0, %1}, "         // D (output)
                     "{%2, %3, %4, %5}, " // A
                     "{%6, %7}, "         // B
                     "{%0, %1};\n"        // C (same regs as D)
                     : "+r"(C_frag[0]),
                       "+r"(C_frag[1]) // read-write: C in, D out
                     : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]),
                       "r"(A_frag[3]), "r"(B_frag[0]), "r"(B_frag[1]));
        // C is 16x8
        // t0 writes to 8*0+0,8*0+1, and 16*8,16*8+1
        // t1 writes to 8*0 2,3  and 16*8+2,16*8+3
        // t4 writes to 8*1+0,8*1+1 and 16*9+0, 16*9+1

        int row_id = tidx / 4;
        int col_id = tidx % 4;
        // printf("tidx%d writes to %d\n", tidx, row_id * 8 + col_id * 2);
        // printf("tidx%d writes to %d\n", tidx, 8 * (8 + row_id) + col_id * 2);
        C_tile[(row_id * 8 + col_id * 2)] += C_frag[0];
        C_tile[8 * (8 + row_id) + col_id * 2] = C_frag[1];
      }
      for (int mm = 0; mm < mma_m; mm++) {
        for (int nn = 0; nn < mma_n / 2; nn++) {
          ((uint32_t *)C)[(m + mm) * M + n + nn] = C_tile[mm * mma_m + n];
        }
      }
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  __shared__ half A_s[TILE_M * TILE_K];
  __shared__ half B_s[TILE_N * TILE_K];
  __shared__ half C_s[TILE_M * TILE_N];
  for (int m = 0; m < TILE_M; m++) {
    for (int k = 0; k < TILE_K; k += WARP_SIZE) {
      A_s[m * TILE_K + k + tidx] =
          A[(blockIdx.x + m) * K + blockIdx.y + k + tidx];
    }
  }
  for (int k = 0; k < TILE_K; k++) {
    for (int n = 0; n < TILE_N; n += WARP_SIZE) {
      B_s[k * TILE_N + n + tidx] =
          B[(blockIdx.x + k) * N + blockIdx.y + n + tidx];
    }
  }
  __syncthreads();
  for (int k = 0; k < TILE_K; k += TILE_K) {
    _mma_smem_smem_smem<TILE_M, TILE_N>(half * A, half * B, half * C, int K)
  }
}

int main() {
  // Simple test: Identity matrix * ones vector
  const int M = 16, K = 16, N = 8;

  half *h_A = new half[M * K];
  half *h_B = new half[K * N];
  half *h_C = new half[M * N];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      float v = (i == j) ? float(i + 1) : 0.0f;
      h_A[i * K + j] = __float2half(v);
    }
  }

  // B: all ones (like you already have)
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = __float2half(1.0f);
  }

  // Initialize C to zero
  for (int i = 0; i < M * N; i++) {
    h_C[i] = __float2half(0.0f);
  }

  // Allocate device memory
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));

  // Copy to device
  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice);

  // Launch kernel with 1 warp (32 threads)
  mma<<<1, 32>>>(d_A, d_B, d_C);

  // Copy result back
  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // Print result (should be all 1s since I * ones = ones)
  printf("Result C (16x8):\n");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%.1f ", __half2float(h_C[i * N + j]));
    }
    printf("\n");
  }

  // Cleanup
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
