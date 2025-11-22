#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cassert>
#include <cstdint>
#include <random>
#include <stdio.h>


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
  }
}

template<typename Fn>
void validate_matmul(Fn matmul_kernel, int M, int N, int K) {
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

  ref_matmul(h_A_rand, h_B_rand, h_C_ref, M, N, K);

  half *d_A, *d_B;
  float *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A_rand, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_rand, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C_gpu, M * N * sizeof(float), cudaMemcpyHostToDevice);

  matmul_kernel(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  check_results(h_C_ref, h_C_gpu, M, N);

  delete[] h_A_rand;
  delete[] h_B_rand;
  delete[] h_C_ref;
  delete[] h_C_gpu;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

template<typename Fn>
void benchmark_matmul(Fn matmulFunc, int M, int N, int K, int num_warmup_iters= 10, int num_bench_iters = 100) {
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

  half *d_A, *d_B;
  float *d_C_custom, *d_C_cublas;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C_custom, M * N * sizeof(float));
  cudaMalloc(&d_C_cublas, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  const float alpha = 1.0f;
  const float beta = 0.0f;


  //warmup
  for (int i = 0; i < num_warmup_iters; ++i) {
    matmulFunc(d_A, d_B, d_C_custom, M, N, K);
  }
  cudaDeviceSynchronize();

  // Benchmark custom kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < num_bench_iters; ++i) {
    matmulFunc(d_A, d_B, d_C_custom, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float custom_time_ms;
  cudaEventElapsedTime(&custom_time_ms, start, stop);
  float custom_time_avg = custom_time_ms / num_bench_iters;


  //warmup
  for (int i = 0; i < num_warmup_iters; ++i) {
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

  cudaEventRecord(start);
  for (int i = 0; i < num_bench_iters; ++i) {
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
  float cublas_time_avg = cublas_time_ms / num_bench_iters;

  double flops = 2.0 * M * N * K;
  double custom_tflops = (flops / (custom_time_avg / 1000.0)) / 1e12;
  double cublas_tflops = (flops / (cublas_time_avg / 1000.0)) / 1e12;
  double efficiency = (custom_tflops / cublas_tflops) * 100.0;

  printf("\n========== Benchmark Results ==========\n");
  printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Iterations: %d\n\n", num_bench_iters);
  printf("Custom Kernel:  %.4f ms  (%.2f TFLOPS)\n", custom_time_avg, custom_tflops);
  printf("cuBLAS:         %.4f ms  (%.2f TFLOPS)\n", cublas_time_avg, cublas_tflops);
  printf("Efficiency:     %.2f%%\n", efficiency);
  printf("Speedup:        %.2fx %s\n", 
         custom_time_avg < cublas_time_avg ? cublas_time_avg / custom_time_avg : cublas_time_avg / custom_time_avg,
         custom_time_avg < cublas_time_avg ? "(faster)" : "(slower)");
  printf("=======================================\n\n");

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
