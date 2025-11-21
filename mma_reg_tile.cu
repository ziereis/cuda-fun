#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <random>
#include <stdio.h>
#include <sys/types.h>
#include <cublas_v2.h>

#define WARP_SIZE 32


template<int M, int N, int K, int LDA, int LDB, int NUM_THREADS>
__device__ void _mma_sync_smem_smem_reg_n_dist(half *__restrict A_s, half *__restrict  B_s,
                                        uint32_t *__restrict C_frag){
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int C_M = M / MMA_M;
  constexpr int C_N = N / MMA_N;
  for (int k = 0; k < K; k += MMA_K) {
    for (int m = 0; m < C_M; m++) {
      int a0 = m * MMA_M * LDA + k;
      int idx_A = (tidx % 8) * LDA + (tidx / 16) * LDA * 8 +
                        ((tidx / 8) % 2) * 8;
      half *A_row = A_s + a0 + idx_A;
      uint32_t A_addr = __cvta_generic_to_shared(A_row);
      uint32_t A_frag[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
                   "%3}, [%4];\n"
                   : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]),
                     "=r"(A_frag[3])
                   : "r"(A_addr));
      for (int n = 0; n < C_N; n += NUM_WARPS) {
        int b0 = k * LDB  + (n * MMA_N) + MMA_N * widx;
        int idx_B = (tidx % 16) * LDB;
        half *B_row = B_s + b0 + idx_B;
        uint32_t B_addr = __cvta_generic_to_shared(B_row);
        uint32_t B_frag[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                     "{%0, %1}, [%2];\n"
                     : "=r"(B_frag[0]), "=r"(B_frag[1])
                     : "r"(B_addr));
        int c0 = m * (C_N * 4) + (n + widx) * 4;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "         // D (output)
                     "{%4, %5, %6, %7}, " // A
                     "{%8, %9}, "         // B
                     "{%0, %1, %2, %3};\n"        // C (same regs as D)
                     : "+r"(C_frag[c0]), "+r"(C_frag[c0 + 1]), "+r"(C_frag[c0 + 2]) ,"+r"(C_frag[c0 + 3]) 
                     : "r"(A_frag[0]), "r"(A_frag[2]), "r"(A_frag[1]), // permute A here it needs to b passed col major
                       "r"(A_frag[3]), "r"(B_frag[0]), "r"(B_frag[1]));
      }
    }
  }
}

template<int M, int N, int K, int LDA, int LDB, int NUM_THREADS>
__device__ void _mma_sync_smem_smem_reg_m_dist(half *__restrict A_s, half *__restrict  B_s,
                                               uint32_t *__restrict C_frag) {
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  
  constexpr int C_M = M / (MMA_M * NUM_WARPS);
  constexpr int C_N = N / MMA_N;
  constexpr int swizzle_bits = 3;
  constexpr int swizzle_mask = (1 << swizzle_bits) - 1;

  for (int k = 0; k < K; k += MMA_K) {
    #pragma unroll
    for (int m = 0; m < C_M; m++) {
      int local_row_a = (tidx % 8) + (tidx / 16) * 8;
      int row_idx = (m * NUM_WARPS + widx) * MMA_M + local_row_a;
      int logical_col = k + ((tidx / 8) % 2) * 8;
      // int modifier = (row_idx & swizzle_mask) << 3;
      // int swizzled_col = logical_col ^ modifier;
      half *A_row = A_s + (row_idx * LDA) + logical_col;
      
      uint32_t A_addr = __cvta_generic_to_shared(A_row);
      uint32_t A_frag[4];
      
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                   : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]), "=r"(A_frag[3])
                   : "r"(A_addr));
      #pragma unroll
      for (int n = 0; n < C_N; n++) {
        int local_row_b = tidx % 16;
        int row_idx_b = k + local_row_b;
        int logical_col_b = n * MMA_N;
        int modifier_b = (row_idx_b & swizzle_mask) << 3;
        int swizzled_col_b = logical_col_b ^ modifier_b;
        half *B_row = B_s + (row_idx_b * LDB) + swizzled_col_b;
        
        uint32_t B_addr = __cvta_generic_to_shared(B_row);
        uint32_t B_frag[2];
        
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(B_frag[0]), "=r"(B_frag[1])
                     : "r"(B_addr));

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

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS, int NUM_STAGES = 4>
__global__ void matmul(half *A, half *B, float *C, int M, int N, int K) {
  constexpr int GROUP_SIZE = 8;
  constexpr int GROUP_SIZE_A = 4;
  constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  constexpr int NUM_GROUPS_A = WARP_SIZE / GROUP_SIZE_A;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int C_M = TILE_M / (MMA_M * NUM_WARPS);
  constexpr int C_N = TILE_N / MMA_N;
  constexpr int LDA = (TILE_K + 8);
  constexpr int LDB = TILE_N;
  constexpr int LDC = (TILE_N + 4);
  // we have to further divide the thread ids so we can also distribute them across M
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE; 
  const int gidx_a = tidx / GROUP_SIZE_A;
  const int lidx_a = tidx % GROUP_SIZE_A; 
  constexpr int A_size = TILE_M * LDA * 2;  
  constexpr int B_size = TILE_K * LDB * 2;  
  constexpr int C_size = MMA_M * NUM_WARPS * LDC * 4; // we only have to keep one chunk of M in shmem at once  
  constexpr int AB_size = (A_size + B_size) * NUM_STAGES;
  constexpr int smem_size = (AB_size > C_size) ? AB_size : C_size;
  __shared__ __align__(16) uint8_t smem[smem_size];
  half *A_s = reinterpret_cast<half*>(smem);
  half *B_s = reinterpret_cast<half*>(smem + A_size * NUM_STAGES);
  uint32_t C_frag[C_M][C_N][4] = {0};


  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;
  constexpr int swizzle_bits = 3;
  constexpr int mask = (1 << swizzle_bits) - 1;

  auto load_stage = [&](int k, int stage_idx) {
    half* A_dst = A_s + (stage_idx * TILE_M * LDA);
    half* B_dst = B_s + (stage_idx * TILE_K * LDB);

    // Load A
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS_A) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE_A * 8) {
        int local_row = mm + widx * NUM_GROUPS_A + gidx_a;
        // int modifier = (local_row & mask) << 3;
        int logical_col = kk + lidx_a * 8;
        // int swizzled_col = logical_col ^ modifier;
        uint32_t smem_addr = __cvta_generic_to_shared(
            &A_dst[local_row * LDA + logical_col]);
        const half* gmem_addr = &A[(m + local_row) * K + k + logical_col];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
      }
    }
    // Load B
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * 8) {
        int local_row = kk + widx * NUM_GROUPS + gidx;
        int modifier = (local_row & mask) << 3;
        int logical_col = nn + lidx * 8;
        int swizzled_col = logical_col ^ modifier;
        uint32_t smem_addr = __cvta_generic_to_shared(
            &B_dst[local_row * LDB + swizzled_col]);
        const half* gmem_addr = &B[(k + local_row) * N + n + logical_col];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
      }
    }
  };

  // start loading of the first NUM_STAGES - 1 tiles
#pragma unroll
  for (int s = 0; s < NUM_STAGES - 1; ++s) {
    load_stage(s * TILE_K, s);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  for (int k = 0; k < K; k += TILE_K) {
    int next_k = k + (NUM_STAGES - 1) * TILE_K;
    int write_stage = (next_k / TILE_K) % NUM_STAGES;

    if (next_k < K) {
      load_stage(next_k, write_stage);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
    __syncthreads();

    int read_stage = (k / TILE_K) % NUM_STAGES;
    _mma_sync_smem_smem_reg_m_dist<TILE_M, TILE_N, TILE_K, LDA, LDB, NUM_THREADS>(
              A_s + (read_stage * TILE_M * LDA), 
              B_s + (read_stage * TILE_K * LDB), 
              reinterpret_cast<uint32_t*>(C_frag)
          );

    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;\n" ::);

  uint32_t *C_s = reinterpret_cast<uint32_t*>(smem);
  for (int mm = 0; mm < C_M; mm++) {
      for (int nn = 0; nn < C_N; nn++) {
        int row_id = tidx / 4;
        int col_id = tidx % 4;
        int c0 = widx * MMA_M * LDC + (nn * MMA_N); 
        int idx = c0 + row_id * LDC + col_id * 2;
        C_s[idx] = C_frag[mm][nn][0];
        C_s[idx+1] = C_frag[mm][nn][1];
        idx = c0 + (8 + row_id) * LDC + col_id * 2;
        C_s[idx] = C_frag[mm][nn][2];
        C_s[idx+1] = C_frag[mm][nn][3];
      }

      // __syncthreads();
      // need a min N of 128 for this to work but its fine for now, could also distribute like above
      for (int row = 0; row < MMA_M; row++) {
        for (int nn = 0; nn < TILE_N; nn += WARP_SIZE * 4) {
          *reinterpret_cast<int4*>(&C[(m + (mm * MMA_M * NUM_WARPS) + widx * MMA_M + row) * N + n + nn + tidx * 4]) =
              *reinterpret_cast<int4*>(&C_s[(widx * MMA_M + row) * LDC + nn + tidx * 4]);
        }
      }
   }
}

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
void cuda_matmul(half *A, half *B, float *C, int M, int N, int K) {
  dim3 grid_dim(M / TILE_M, N / TILE_N);
  dim3 block_dim(NUM_THREADS);

  matmul<TILE_M, TILE_N, TILE_K, NUM_THREADS><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
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

void check_results(float *h_C_ref, float *h_C_gpu, int M, int N, float epsilon = 1e-2) {
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
  const int M = 1024, K = 1024, N = 1024;
  constexpr int TILE_M = 128, TILE_K = 32, TILE_N = 128;
  constexpr int num_threads = 256;
  validate_matmul<TILE_M, TILE_N, TILE_K, num_threads>(M, N, K);
  benchmark_matmul<TILE_M,TILE_N, TILE_K, num_threads>(M, N, K);


  // half *h_A = new half[M * K];
  // half *h_B = new half[K * N];
  // float *h_C = new float[M * N];
  //
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < K; ++j) {
  //     float v = i == j ? 1.0 : 0.0;
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
  //
  // return 0;
}
