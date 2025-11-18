#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <random>
#include <stdio.h>
#include <sys/types.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

template<int LDA, int LDB, int NUM_THREADS>
__device__ void mma_m16n8k16(half* __restrict A_s, half* __restrict B_s, uint32_t* __restrict C_frag) {
  const int tidx = threadIdx.x % WARP_SIZE;
  int idx_A = (tidx % 8) * LDA + (tidx / 16) * LDA * 8 +
                    ((tidx / 8) % 2) * 8;
  half *A_row = A_s + idx_A;
  uint32_t A_addr = __cvta_generic_to_shared(A_row);
  uint32_t A_frag[4];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
               "%3}, [%4];\n"
               : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]),
                 "=r"(A_frag[3])
               : "r"(A_addr));
  int idx_B =(tidx % 16) * LDB;
  half *B_row = B_s + idx_B;
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

template <int TILE_M, int TILE_N, int TILE_K, int NUM_THREADS>
__global__ void matmul(half *A, half *B, float *C, int M, int N, int K) {
  constexpr int GROUP_SIZE = 8;
  constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int tidx = threadIdx.x % WARP_SIZE;
  const int widx = threadIdx.x / WARP_SIZE;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int C_M = TILE_M / MMA_M;
  constexpr int C_N = TILE_N / MMA_N;
  constexpr int LDA = (TILE_K + 8);
  constexpr int LDB = (TILE_N + 8);
  // we have to further divide the thread ids so we can also distribute them across M
  const int gidx = tidx / GROUP_SIZE;
  const int lidx = tidx % GROUP_SIZE;
  __shared__ __align__(16) half A_s[TILE_M * LDA];
  __shared__ __align__(16) half B_s[TILE_K * LDB];
  uint32_t C_frag[C_M][C_N][4] = {0};


  const int m = blockIdx.x * TILE_M;
  const int n = blockIdx.y * TILE_N;


  for (int k = 0; k < K; k += TILE_K) {
    for (int mm = 0; mm < TILE_M; mm += NUM_WARPS * NUM_GROUPS) {
      for (int kk = 0; kk < TILE_K; kk += GROUP_SIZE * 8) { // 8 * f16 = 128
        *reinterpret_cast<int4*>(&A_s[(mm + widx * NUM_GROUPS + gidx) * LDA + kk + lidx * 8]) =
            *reinterpret_cast<int4*>(&A[(m + mm + widx * NUM_GROUPS + gidx) * K + k + kk + lidx * 8]);
      }
    }
    // same applies here
    for (int kk = 0; kk < TILE_K; kk += NUM_WARPS * NUM_GROUPS) {
      for (int nn = 0; nn < TILE_N; nn += GROUP_SIZE * 8) {
        *reinterpret_cast<int4*>(&B_s[(kk + widx *NUM_GROUPS + gidx) * LDB + nn + lidx * 8]) =
            *reinterpret_cast<int4*>(&B[(k + kk + widx * NUM_GROUPS + gidx) * N + n + nn + lidx * 8]);
      }
    }

    __syncthreads();

    for (int kk = 0; kk < TILE_K; kk += MMA_M) {
      for (int mm = 0; mm < C_M; mm++) {
        for (int nn = 0; nn < C_N; nn += NUM_WARPS) {
          int a0 = mm * MMA_M * LDA + kk;
          int b0 = kk * LDB + (nn * MMA_N) + MMA_N * widx;
          mma_m16n8k16<LDA, LDB, NUM_THREADS>(A_s + a0, B_s + b0, C_frag[mm][nn + widx]);
        }
      }

    }

    __syncthreads();
    for (int mm = 0; mm < C_M; mm++) {
      for (int nn = 0; nn < C_N; nn += NUM_WARPS) {
        int row_id = tidx / 4;
        int col_id = tidx % 4;
        int c0 = (m + mm * MMA_M) * N + n + (nn * MMA_N) + widx * MMA_N; 
        int idx = c0 + row_id * N + col_id * 2;
        *reinterpret_cast<uint32_t*>(&C[idx]) = C_frag[mm][nn + widx][0];
        *reinterpret_cast<uint32_t*>(&C[idx+1]) = C_frag[mm][nn + widx][1];
        idx = c0 + (8 + row_id) * N + col_id * 2;
        *reinterpret_cast<uint32_t*>(&C[idx]) = C_frag[mm][nn + widx][2];
        *reinterpret_cast<uint32_t*>(&C[idx+1]) = C_frag[mm][nn + widx][3];
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
  const int M = 16, K = 64, N = 64;
  constexpr int TILE_M = 16, TILE_K = 64, TILE_N = 64;
  constexpr int num_threads = 128;
  validate_matmul<TILE_M, TILE_N, TILE_K, num_threads>(M, N, K);
  // benchmark_matmul<TILE_M,TILE_N, TILE_K, num_threads>(M, N, K);


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
