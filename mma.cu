#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <random>
#include <stdio.h>
#include <sys/types.h>
#include <cublas_v2.h>

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
__device__ void _mma_smem_smem_smem(half *__restrict A, half *__restrict B,
                                    float *__restrict C) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
#pragma unroll
  for (int m = 0; m < M; m += NUM_WARPS) {
#pragma unroll
    for (int n = 0; n < N; n += WARP_SIZE) {
      float C_tile[NUM_WARPS * WARP_SIZE] = {0.0};
      for (int k = 0; k < K; k++) {
        C_tile[widx * WARP_SIZE + tidx] +=
            __half2float(A[(m + widx) * K + k]) * __half2float(B[k * N + n + tidx]);
      }
      C[(m + widx) * N + n + tidx] += C_tile[widx * WARP_SIZE + tidx];
    }
  }
}

template <int M, int N, int K, int NUM_THREADS>
__device__ void _mma_sync_smem_smem_smem(half *__restrict A, half *__restrict B,
                                         float *__restrict C) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int mma_m = 16;
  constexpr int mma_n = 8;
  constexpr int mma_k = 16;
  for (int m = 0; m < M; m += mma_m) {
    for (int n = 0; n < N; n += mma_n * NUM_WARPS) {
      // mma_n / 2 because we are allocating 32 bit registers
      uint32_t C_frag[4] = {0};
      for (int k = 0; k < K; k += mma_k) {
        
        int a0 = m * (K + 8) + k;
        int idx_A = a0 + (tidx % 8) * (K + 8) + (tidx / 16) * (K+8) * 8 +
                    ((tidx / 8) % 2) * 8;
        // printf("tidx%d widx%d loads A at %d\n", tidx, widx, idx_A);
        half *A_row = &A[idx_A];
        uint32_t A_addr = __cvta_generic_to_shared(A_row);
        uint32_t A_frag[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
                     "%3}, [%4];\n"
                     : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]),
                       "=r"(A_frag[3])
                     : "r"(A_addr));

        // each thread in a warp loads 4 consecutive bytes (8 f16 elements)
        // starting at indices 
        // N is always some multiple of 32
        // we can calculate the bank of an address like this:
        // bank = (address / 4) % 32
        // assume N -> 64 f16s -> 128 bytes
        // 0 * N: bank = (0 / 4) % 32 -> 0
        // 1 * N: bank = (128 / 4) % 32 -> 0
        // 2 * N: bank = (256 / 4) % 32 -> 0
        // ..
        // 15 * N: bank = (1920 / 4) % 32 -> 0
        // here all 16 threads would try to access the same memory bank and we get bank conflicts
        // now if we were to pad N by 2 elements (or 4 bytes)
        // assume N -> 66 f16s -> 132 bytes
        // 0 * N: bank = (0 / 4) % 32 -> 0
        // 1 * N: bank = (132 / 4) % 32 -> 1
        // 2 * N: bank = (264 / 4) % 32 -> 2
        // ..
        // 15 * N: bank = (1980 / 4) % 32 -> 15
        // now if we were to pad N by 8 elements (or 16 bytes)
        // assume N -> 74 f16s -> 144 bytes
        // 0 * N: bank = (0 / 4) % 32 -> 0
        // 1 * N: bank = (144 / 4) % 32 -> 4
        // 2 * N: bank = (264 / 4) % 32 -> 2
        // ..
        // 15 * N: bank = (1980 / 4) % 32 -> 15
        half *B_row = &B[(k + (tidx % 16)) * (N + 8) + n + mma_n * widx];
        uint32_t B_addr = __cvta_generic_to_shared(B_row);
        uint32_t B_frag[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                     "{%0, %1}, [%2];\n"
                     : "=r"(B_frag[0]), "=r"(B_frag[1])
                     : "r"(B_addr));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "         // D (output)
                     "{%4, %5, %6, %7}, " // A
                     "{%8, %9}, "         // B
                     "{%0, %1, %2, %3};\n"        // C (same regs as D)
                     : "+r"(C_frag[0]), "+r"(C_frag[1]), "+r"(C_frag[2]) ,"+r"(C_frag[3]) 
                     : "r"(A_frag[0]), "r"(A_frag[2]), "r"(A_frag[1]), // permute A here it needs to b passed col major
                       "r"(A_frag[3]), "r"(B_frag[0]), "r"(B_frag[1]));
      }

      // to which banks do the threads store to?
      // every thread stores 8 bytes
      // assuming N is 64 = 256 bytes
      // bank = (idx / 4) % 32
      // t0: idx = 0 * (256 + 4) + 0 * 2; banks(0,1)
      // t0: idx = (8+0) * (256 + 4) + 0 * 2; banks(8,9)
      // t1: idx = 0 * (256 + 4) + 1 * 2; banks(0,1)
      // t1: idx = (8+0) * (256 + 4) + 1 * 2; banks(8,9)
      int row_id = tidx / 4;
      int col_id = tidx % 4;
      int c0 = m * (N+4) + n + widx * mma_n;
      int idx = c0 + row_id * (N+4) + col_id * 2;
      float* C_fragf32 = reinterpret_cast<float*>(C_frag);
      C[idx] += C_fragf32[0];
      C[idx+1] += C_fragf32[1];
      idx = c0 + (8 + row_id) * (N+4) + col_id * 2; 
      C[idx] += C_fragf32[2];
      C[idx+1] += C_fragf32[3];
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS, int PIPELINE_STAGES>
__global__ void matmul(half *A, half *B, float *C, int M, int N, int K) {
  constexpr int GROUP_SIZE = 8;
  constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE;
  
  // N-stage buffered shared memory
  __shared__ __align__(16) half A_s[PIPELINE_STAGES][TILE_M * (TILE_K + 8)];
  __shared__ __align__(16) half B_s[PIPELINE_STAGES][TILE_K * (TILE_N + 8)];
  __shared__ __align__(16) float C_s[TILE_M * (TILE_N + 4)];
  
  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;
  const int num_tiles = (K + TILE_K - 1) / TILE_K;
  
  // Initialize C_s
  float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * 4) {
      *reinterpret_cast<float4*>(&C_s[(mm + widx * NUM_GROUPS + gidx) * (TILE_N + 4) + nn + lidx * 4]) = zero;
    }
  }
  
  // Helper lambda for loading a tile
  auto load_tile = [&](int tile_idx, int buffer_idx) {
    int k = tile_idx * TILE_K;
    if (k >= K) return;
    
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE * 8) {
        if (k + kk + lidx * 8 < K) {
          uint32_t smem_addr = __cvta_generic_to_shared(
              &A_s[buffer_idx][(mm + widx * NUM_GROUPS + gidx) * (TILE_K + 8) + kk + lidx * 8]);
          const half* gmem_addr = &A[(m + mm + widx * NUM_GROUPS + gidx) * K + k + kk + lidx * 8];
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
        }
      }
    }
    
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * 8) {
        if (k + kk + widx * NUM_GROUPS + gidx < K) {
          uint32_t smem_addr = __cvta_generic_to_shared(
              &B_s[buffer_idx][(kk + widx * NUM_GROUPS + gidx) * (TILE_N + 8) + nn + lidx * 8]);
          const half* gmem_addr = &B[(k + kk + widx * NUM_GROUPS + gidx) * N + n + nn + lidx * 8];
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  
  // Prefetch first PIPELINE_STAGES-1 tiles
  #pragma unroll
  for (int stage = 0; stage < PIPELINE_STAGES - 1; ++stage) {
    load_tile(stage, stage);
  }
  
  // Main loop with N-stage software pipelining
  for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int read_buffer = tile_idx % PIPELINE_STAGES;
    int write_buffer = (tile_idx + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
    
    // Issue load for tile (tile_idx + PIPELINE_STAGES - 1)
    if (tile_idx + PIPELINE_STAGES - 1 < num_tiles) {
      load_tile(tile_idx + PIPELINE_STAGES - 1, write_buffer);
    }
    
    // Wait until current tile is ready (keep PIPELINE_STAGES-1 groups in flight)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(PIPELINE_STAGES - 1));
    __syncthreads();
    
    // Compute current tile
    _mma_sync_smem_smem_smem<TILE_M, TILE_N, TILE_K, NUM_THREADS>(
        A_s[read_buffer], B_s[read_buffer], C_s);
    
    __syncthreads();
  }
  
  // Wait for all remaining async copies
  asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
  __syncthreads();
  
  // Store results
  for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
    for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * 4) {
      *reinterpret_cast<int4*>(&C[(m + mm + widx * NUM_GROUPS + gidx) * N + n + nn + lidx * 4]) =
          *reinterpret_cast<int4*>(&C_s[(mm + widx * NUM_GROUPS + gidx) * (TILE_N + 4) + nn + lidx * 4]);
    }
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void cuda_matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);

  matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS, 2><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

void ref_matmul(half* A,half* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float c_val = 0.0f;
      for (int k = 0; k < K; ++k) {
        c_val += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
      }
      C[m * N + n] = c_val;
    }
  }
}


void print_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%.2f ", mat[i * cols + j]);
    }
    printf("\n");
  }
}

void check_results(float *h_C_ref, float *h_C_gpu, int M, int N, float epsilon = 1e-3) {
  bool ok = true;
  for (int i = 0; i < M * N; ++i) {
    float ref_val = h_C_ref[i];
    float gpu_val = h_C_gpu[i];
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
    // printf("Reference:");
    // print_matrix(h_C_ref, M, N);
    // printf("GPU result:\n");
    // print_matrix(h_C_gpu, M, N);
  }
}

template<int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void benchmark_matmul(int M, int N, int K, int num_iterations = 100) {
  // Allocate and initialize host memory
  half *h_A = new half[M * K];
  half *h_B = new half[K * N];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0.0, 1.0);
  
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = __float2half(distrib(gen));
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = __float2half(distrib(gen));
  }

  // Allocate device memory
  half *d_A, *d_B;
  float *d_C_custom, *d_C_cublas;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C_custom, M * N * sizeof(float));
  cudaMalloc(&d_C_cublas, M * N * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Warmup runs
  for (int i = 0; i < 10; ++i) {
    cuda_matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>(d_A, d_B, d_C_custom, M, N, K);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C_cublas, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaDeviceSynchronize();

  // Benchmark custom kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < num_iterations; ++i) {
    cuda_matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>(d_A, d_B, d_C_custom, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float custom_time_ms;
  cudaEventElapsedTime(&custom_time_ms, start, stop);
  float custom_time_avg = custom_time_ms / num_iterations;

  // Benchmark cuBLAS
  cudaEventRecord(start);
  for (int i = 0; i < num_iterations; ++i) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C_cublas, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float cublas_time_ms;
  cudaEventElapsedTime(&cublas_time_ms, start, stop);
  float cublas_time_avg = cublas_time_ms / num_iterations;

  // Calculate performance metrics
  double flops = 2.0 * M * N * K;
  double custom_tflops = (flops / (custom_time_avg / 1000.0)) / 1e12;
  double cublas_tflops = (flops / (cublas_time_avg / 1000.0)) / 1e12;
  double efficiency = (custom_tflops / cublas_tflops) * 100.0;

  // Print results
  printf("\n========== Benchmark Results ==========\n");
  printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Tile config: TILE_M=%d, TILE_N=%d, TILE_K=%d, NUM_THREADS=%d\n", 
         TILE_M, TILE_N, TILE_K, NUM_THREADS);
  printf("Iterations: %d\n\n", num_iterations);
  printf("Custom Kernel:  %.4f ms  (%.2f TFLOPS)\n", custom_time_avg, custom_tflops);
  printf("cuBLAS:         %.4f ms  (%.2f TFLOPS)\n", cublas_time_avg, cublas_tflops);
  printf("Efficiency:     %.2f%%\n", efficiency);
  printf("Speedup:        %.2fx %s\n", 
         custom_time_avg < cublas_time_avg ? cublas_time_avg / custom_time_avg : cublas_time_avg / custom_time_avg,
         custom_time_avg < cublas_time_avg ? "(faster)" : "(slower)");
  printf("=======================================\n\n");

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  delete[] h_A;
  delete[] h_B;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C_custom);
  cudaFree(d_C_cublas);
}

template<int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void validate_matmul(int M, int N, int K) {
  // randomly generate A, B and K matrices on host
  half *h_A_rand = new half[M * K];
  half *h_B_rand = new half[K * N];
  float *h_C_ref = new float[M * N];
  float *h_C_gpu = new float[M * N];
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
    h_C_ref[i] = 0.0f;
    h_C_gpu[i] = 0.0f;
  }

  // Calculate reference result on CPU
  ref_matmul(h_A_rand, h_B_rand, h_C_ref, M, N, K);

  // Allocate device memory and copy data
  half *d_A, *d_B;
  float *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A_rand, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_rand, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C_gpu, M * N * sizeof(float), cudaMemcpyHostToDevice); // Initialize d_C with zeros

  // Execute kernel
  cuda_matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS>(d_A, d_B, d_C, M, N, K);

  // Copy result back to host
  cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
  // after vectorzing all loads and stores 128bit 
  const int M = 1024, K = 1024, N = 1024;
  constexpr int TILE_M = 64, TILE_K = 64, TILE_N = 128;
  constexpr int num_threads = 256;
  validate_matmul<TILE_M, TILE_N, TILE_K, num_threads>(M, N, K);
  benchmark_matmul<TILE_M,TILE_N, TILE_K, num_threads>(M, N, K);


  // half *h_A = new half[M * K];
  // half *h_B = new half[K * N];
  // float *h_C = new float[M * N];
  //
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < K; ++j) {
  //     float v = i;
  //     h_A[i * K + j] = __float2half(v);
  //   }
  // }
  //
  // // B: all ones (like you already have)
  // for (int i = 0; i < K * N; ++i) {
  //   h_B[i] = __float2half(1.0f);
  // }
  //
  // // Initialize C to zero
  // for (int i = 0; i < M * N; i++) {
  //   h_C[i] = 0.0f;
  // }
  //
  // // Allocate device memory
  // half *d_A, *d_B;
  // float* d_C;
  // cudaMalloc(&d_A, M * K * sizeof(half));
  // cudaMalloc(&d_B, K * N * sizeof(half));
  // cudaMalloc(&d_C, M * N * sizeof(float));
  //
  // // Copy to device
  // cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
  //
  //
  // assert(M % TILE_M == 0);
  // assert(N % TILE_N == 0);
  // assert(K % TILE_K == 0);
  //
  // dim3 grid(M / TILE_M, N / TILE_N);
  // matmul<TILE_M, TILE_N, TILE_K, num_threads><<<grid, num_threads>>>(d_A, d_B, d_C, M, N, K);
  //
  // // Copy result back
  // cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  //
  // // Print result (should be all 1s since I * ones = ones)
  // printf("Result C (%dx%d):\n", M, N);
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //     printf("%.1f ", h_C[i * N + j]);
  //   }
  //   printf("\n");
  // }
  //
  // // Cleanup
  // delete[] h_A;
  // delete[] h_B;
  // delete[] h_C;
  // cudaFree(d_A);
  // cudaFree(d_B);
  // cudaFree(d_C);

  return 0;
}
