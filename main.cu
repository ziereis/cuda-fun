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

__global__ void ldmatrix(half *A, half *B, uint64_t *latency) {
  const int tidx = threadIdx.x;
  __shared__ half A_s[16 * 16];
  for (int i = 0; i < 16 * 16; i += WARP_SIZE) {
    A_s[i + tidx] = A[i + tidx];
  }
  __syncthreads();
  uint64_t start = clock64();
  // assume non-negative tidx
  int lane8 = tidx & 7;
  int group16 = tidx >> 4;
  int pair8 = (tidx >> 3) & 1;

  int A_idx = (lane8 << 4) + (group16 << 7) + (pair8 << 3);

  half *A_row = &A_s[A_idx];
  uint32_t A_addr = __cvta_generic_to_shared(A_row);
  uint32_t A_frag[4];
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
      : "r"(A_addr));
  uint64_t stop = clock64();
  latency[tidx] = stop - start;
  *((uint32_t *)(&B[tidx * 8])) = A_frag[0];
  *((uint32_t *)(&B[tidx * 8 + 2])) = A_frag[1];
  *((uint32_t *)(&B[tidx * 8 + 4])) = A_frag[2];
  *((uint32_t *)(&B[tidx * 8 + 6])) = A_frag[3];
}

__global__ void load(half *A, half *B, uint64_t *latency) {
  const int tidx = threadIdx.x;
  __shared__ half A_s[16 * 16];
  for (int i = 0; i < 16 * 16; i += WARP_SIZE) {
    A_s[i + tidx] = A[i + tidx];
  }
  __syncthreads();
  uint64_t start = clock64();
  uint32_t A_frag[4];
  int q = tidx >> 2;
  int r = tidx & 3;
  int base = (q << 4) + (r << 1);
  A_frag[0] = *(uint32_t *)(&A_s[base]);
  A_frag[1] = *(uint32_t *)(&A_s[base + 8]);
  A_frag[2] = *(uint32_t *)(&A_s[base + 16 * 8]);
  A_frag[3] = *(uint32_t *)(&A_s[base + 16 * 8 + 8]);
  uint64_t stop = clock64();
  latency[tidx] = stop - start;
  *((uint32_t *)(&B[tidx * 8])) = A_frag[0];
  *((uint32_t *)(&B[tidx * 8 + 2])) = A_frag[1];
  *((uint32_t *)(&B[tidx * 8 + 4])) = A_frag[2];
  *((uint32_t *)(&B[tidx * 8 + 6])) = A_frag[3];
}

int main() {
  // Simple test: Identity matrix * ones vector
  const int M = 16;

  half *h_A = new half[M * M];
  half *h_B = new half[M * M];
  half *h_B0 = new half[M * M];
  uint64_t *time_ldmatrix_h = new uint64_t[32];
  uint64_t *time_load_h = new uint64_t[32];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      h_A[i * M + j] = __float2half(i * M + j);
    }
  }

  // Initialize C to zero
  for (int i = 0; i < M * M; i++) {
    h_B[i] = __float2half(0.0f);
  }
  for (int i = 0; i < M * M; i++) {
    h_B0[i] = __float2half(0.0f);
  }

  // Allocate device memory
  half *d_A, *d_B, *d_B0;
  uint64_t *time_ldmatrix, *time_load;
  cudaMalloc(&d_A, M * M * sizeof(half));
  cudaMalloc(&d_B, M * M * sizeof(half));
  cudaMalloc(&d_B0, M * M * sizeof(half));
  cudaMalloc(&time_ldmatrix, 32 * sizeof(uint64_t));
  cudaMalloc(&time_load, 32 * sizeof(uint64_t));

  // Copy to device
  cudaMemcpy(d_A, h_A, M * M * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, M * M * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B0, h_B0, M * M * sizeof(half), cudaMemcpyHostToDevice);

  ldmatrix<<<1, 32>>>(d_A, d_B, time_ldmatrix);
  load<<<1, 32>>>(d_A, d_B0, time_load);

  // Copy result back
  cudaMemcpy(h_B, d_B, M * M * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B0, d_B0, M * M * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(time_ldmatrix_h, time_ldmatrix, 32 * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(time_load_h, time_load, 32 * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  double sum_load = 0.0;
  double sum_ldmatrix = 0.0;
  for (int i = 0; i < 32; i++) {
    sum_load += (double)time_load_h[i];
    sum_ldmatrix += (double)time_ldmatrix_h[i];
  }

  printf("Time ldmatrix: %f\n", sum_ldmatrix / 32.0);
  printf("Time load: %f\n", sum_load / 32.0);
  printf("load/ldmatrix: %f\n\n", (sum_load / 32) / (sum_ldmatrix / 32.0));

  // Print result (should be all 1s since I * ones = ones)
  printf("Result ldmatrix (16x8):\n");
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 8; j++) {
      printf("%.1f ", __half2float(h_B[i * 8 + j]));
    }
    printf("\n");
  }
  printf("Result load (16x8):\n");
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 8; j++) {
      printf("%.1f ", __half2float(h_B0[i * 8 + j]));
    }
    printf("\n");
  }
}

int foo() {
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
  mma_ldmatrix<<<1, 32>>>(d_A, d_B, d_C);

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
