#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <random>
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


template <int M, int N, int K, int NUM_THREADS>
__device__ void _mma_sync_smem_smem_smem(half *__restrict A, half *__restrict B,
                                         half *__restrict C) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int mma_m = 16;
  constexpr int mma_n = 8;
  constexpr int mma_k = 16;
#pragma unroll
  for (int m = 0; m < M; m += mma_m) {
#pragma unroll
    for (int n = 0; n < N; n += mma_n * NUM_WARPS) {
      // mma_n / 2 because we are allocating 32 bit registers
      uint32_t C_frag[2] = {0};
      for (int k = 0; k < K; k += mma_k) {

        int idx_A = m * K + (tidx % 8) * K + (tidx / 16) * K * 8 +
                    ((tidx / 8) % 2) * 8 + k;
        // printf("tidx%d widx%d loads A at %d\n", tidx, widx, idx_A);
        half *A_row = &A[idx_A];
        uint32_t A_addr = __cvta_generic_to_shared(A_row);
        uint32_t A_frag[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
                     "%3}, [%4];\n"
                     : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]),
                       "=r"(A_frag[3])
                     : "r"(A_addr));

        // printf("tidx%d widx%d loads B at %d\n", tidx, widx,
        // (k + tidx) * N + mma_n * widx);
        half *B_row = &B[(k + (tidx % 16)) * N + n + mma_n * widx];
        uint32_t B_addr = __cvta_generic_to_shared(B_row);
        uint32_t B_frag[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                     "{%0, %1}, [%2];\n"
                     : "=r"(B_frag[0]), "=r"(B_frag[1])
                     : "r"(B_addr));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0, %1}, "         // D (output)
                     "{%2, %3, %4, %5}, " // A
                     "{%6, %7}, "         // B
                     "{%0, %1};\n"        // C (same regs as D)
                     : "+r"(C_frag[0]),
                       "+r"(C_frag[1]) // read-write: C in, D out
                     : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]),
                       "r"(A_frag[3]), "r"(B_frag[0]), "r"(B_frag[1]));
      }
      int row_id = tidx / 4;
      int col_id = tidx % 4;
      int c0 = m * N + n + widx * mma_n;
      int idx = c0 + row_id * N + col_id * 2;
      // accumulate into C
      __half2* dst = reinterpret_cast<__half2*>(&C[idx]);
      *dst = __hadd2(*dst, *reinterpret_cast<__half2*>(&C_frag[0]));  
      idx = c0 + (8 + row_id) * N + col_id * 2; 
      dst = reinterpret_cast<__half2*>(&C[idx]);
      *dst = __hadd2(*dst, *reinterpret_cast<__half2*>(&C_frag[1]));  
    }
  }
}

template <int M, int N, int K, int NUM_THREADS>
__device__ void _mma_smem_smem_smem(half *__restrict A, half *__restrict B,
                                    half *__restrict C) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
#pragma unroll
  for (int m = 0; m < M; m += NUM_WARPS) {
#pragma unroll
    for (int n = 0; n < N; n += WARP_SIZE) {
      half C_tile[NUM_WARPS * WARP_SIZE] = {0.0};
      for (int k = 0; k < K; k++) {
        C_tile[widx * WARP_SIZE + tidx] +=
            A[(m + widx) * K + k] * B[k * N + n + tidx];
      }
      C[(m + widx) * N + n + tidx] += C_tile[widx * WARP_SIZE + tidx];
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
__global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  __shared__ __align__(16) half A_s[TILE_M * TILE_K];
  __shared__ __align__(16) half B_s[TILE_K * TILE_N];
  __shared__ __align__(16) half C_s[TILE_M * TILE_N];

  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;
  
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS) {
    for (int nn = 0; nn < TILE_N; nn += WARP_SIZE) {
      C_s[(mm + widx) * TILE_N + nn + tidx] = 0.0;
    }
  }

  for (int k = 0; k < K; k += TILE_K) {
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS) {
      for (int kk = 0; kk < TILE_K; kk += WARP_SIZE) {
        A_s[(mm + widx) * TILE_K + kk + tidx] =
            A[(m + mm + widx) * K + k + kk + tidx];
      }
    }
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS) {
      for (int nn = 0; nn < TILE_N; nn += WARP_SIZE) {
        B_s[(kk + widx) * TILE_N + nn + tidx] =
            B[(k + kk + widx) * N + n + nn + tidx];
      }
    }

    __syncthreads();

    _mma_smem_smem_smem<TILE_M, TILE_N, TILE_K, NUM_THREADS>(A_s, B_s,
                                                                  C_s);
    __syncthreads();
  }
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS) {
    for (int nn = 0; nn < TILE_N; nn += WARP_SIZE) {
      C[(m + mm + widx) * N + n + nn + tidx] =
          C_s[(mm + widx) * TILE_N + nn + tidx];
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void cuda_matmul(half *A, half *B, half *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);

  matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

void ref_matmul(half* A,half* B, half* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      half c_val = __float2half(0.0f);
      for (int k = 0; k < K; ++k) {
        c_val = __hadd(c_val, __hmul(A[m * K + k], B[k * N + n]));
      }
      C[m * N + n] = c_val;
    }
  }
}


void print_matrix(half *mat, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%.2f ", __half2float(mat[i * cols + j]));
    }
    printf("\n");
  }
}

void check_results(half *h_C_ref, half *h_C_gpu, int M, int N, float epsilon = 1e-2) {
  bool ok = true;
  for (int i = 0; i < M * N; ++i) {
    float ref_val = __half2float(h_C_ref[i]);
    float gpu_val = __half2float(h_C_gpu[i]);
    if (fabs(ref_val - gpu_val) > epsilon) {
      ok = false;
      printf("Error at index %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
      break;
    }
  }
  if (ok) {
    printf("Results match!\n");
  } else {
    printf("Results mismatch.\n");
    printf("Reference:");
    print_matrix(h_C_ref, M, N);
    printf("GPU result:\n");
    print_matrix(h_C_gpu, M, N);
  }
}


template<int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void validate_matmul(int M, int N, int K) {
  // randomly generate A, B and K matrices on host
  half *h_A_rand = new half[M * K];
  half *h_B_rand = new half[K * N];
  half *h_C_ref = new half[M * N];
  half *h_C_gpu = new half[M * N];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0.0, 1.0);

  for (int i = 0; i < M * K; ++i) {
    h_A_rand[i] = __float2half(distrib(gen));
  }
  for (int i = 0; i < K * N; ++i) {
    h_B_rand[i] = __float2half(distrib(gen));
  }
  for (int i = 0; i < M * N; ++i) {
    h_C_ref[i] = __float2half(0.0f);
    h_C_gpu[i] = __float2half(0.0f);
  }

  // Calculate reference result on CPU
  ref_matmul(h_A_rand, h_B_rand, h_C_ref, M, N, K);

  // Allocate device memory and copy data
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A_rand, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_rand, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C_gpu, M * N * sizeof(half), cudaMemcpyHostToDevice); // Initialize d_C with zeros

  // Execute kernel
  cuda_matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>(d_A, d_B, d_C, M, N, K);

  // Copy result back to host
  cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // Check results
  check_results(h_C_ref, h_C_gpu, M, N);

  // Cleanup
  delete[] h_A_rand;
  delete[] h_B_rand;
  delete[] h_C_ref;
  delete[] h_C_gpu;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}

int main() {
  // Simple test: Identity matrix * ones vector
  const int M = 32, K = 64, N = 64;
  constexpr int TILE_M = 16, TILE_K = 32, TILE_N = 32;
  constexpr int num_threads = 128;
  validate_matmul<TILE_M, TILE_N, TILE_K, num_threads>(M, N, K);


  half *h_A = new half[M * K];
  half *h_B = new half[K * N];
  half *h_C = new half[M * N];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      float v = (i == j) ? 1.0 : 0.0f;
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


  assert(M % TILE_M == 0);
  assert(N % TILE_N == 0);
  assert(K % TILE_K == 0);

  dim3 grid(M / TILE_M, N / TILE_N);
  matmul<TILE_M, TILE_K, TILE_N, num_threads><<<grid, num_threads>>>(d_A, d_B, d_C, M, N, K);

  // Copy result back
  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // Print result (should be all 1s since I * ones = ones)
  printf("Result C (%dx%d):\n", M, N);
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
