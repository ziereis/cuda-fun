#include <cuda_fp16.h>
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
  matmul_kernel(d_A, d_B, d_C, M, N, K);

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
