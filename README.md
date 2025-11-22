# How to match cublas on single precision (M_N_K_f16_f16_f32) matrix multiplication?

This repo contains educational cuda kernels that build up from a single warp calculating a single 16x8x16 tile using a PTX mma instruction up to outperforming cublas by 20% for a M=N=K=1024 f16xfp16xf32 matrix multiplication.

NOTE: all benchmarks are performed on RTX 3070 which is sm_86 (while it is ampere it is different from a 4090 (sm_89) and has less shared memory)

## [Single warp MMA][01_warp_mma.cu]
This example shows how to correctly load/store the A B and C matrix to use the `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction.
```
nvcc  -lcublas -arch=sm_86 -O3 01_warp_mma.cu -o main && ./main  
```

## [Multi warp MMA][02_multi_warp_mma.cu]
This example shows how to utilize multiple warps to compute a larger tile
```
nvcc  -lcublas -arch=sm_86 -O3 02_multi_warp_mma.cu -o main && ./main
```

## [Full tiled matrix multiplication][03_tile_mma.cu]
This example shows how to go from computing a single tile with multip warps to integrating the the inner kernel into a full tiled matrix multiplication.
```
 nvcc  -lcublas -arch=sm_86 -O3 03_tile_mma.cu -o main && ./main  
```
12 % of cublas performance

## [128bit vectorized loads/stores][04_vectorized_mem_access.cu]
This example shows how to perform 128bit loads and stores on every thread.
```
nvcc  -lcublas -arch=sm_86 -O3 04_vectorized_mem_access.cu -o main && ./main 
```
13 % of cublas performance

## [Reducing bank conflicts by LD padding][05_ld_padding.cu]
This example shows how to reduce shared memory bank conflicts by padding the leading dimension of the A, B and C matrices.
```
nvcc  -lcublas -arch=sm_86 -O3 05_ld_padding.cu -o main && ./main
```
68 % of cublas performance


## [Reducing bank conflicts by shared memory swizzling][06_smem_swizzling.cu]
This example shows how to reduce shared memory bank conflicts by XOR swizzling shared memory accesses.
```
nvcc  -lcublas -arch=sm_86 -O3 06_smem_swizzling.cu -o main && ./main   
```
58 % of cublas performance

## [Increase register usage by promoting C to registers][07_c_reg_tile.cu]
This example shows how to decrease shared memory usage and increase register usage by keeping the entire C tile in registers.
```
nvcc  -lcublas -arch=sm_86 -O3 07_c_reg_tile.cu -o main && ./main 
```
103 % of cublas performance

## [Software pipelining][08_pipelining.cu]
This example shows how to do software pipelining with async shared memory load/stores to overlap with compute. 
```
nvcc  -lcublas -arch=sm_86 -O3 08_pipelining.cu -o main && ./main 
```
97 % of cublas performance

## [C_frag -> C shared memory staging][09_c_smem_staging.cu]
This examples shows how to get coalesced global memory stores to C by staging C_frag through shared memory
```
nvcc  -lcublas -arch=sm_86 -O3 09_c_smem_staging.cu -o main && ./main 
```
104 % of cublas performance

## [Software pipelining and shared memory swizzling][10_pipelining_with_swizzling.cu]
This examples shows how to combine software pipelining (a technique that increases shared memory usage) and swizzling (a techniques that reduces shared memory usage in comparison to LD padding) to make software pipeling profitable.

```
nvcc  -lcublas -arch=sm_86 -O3 10_pipelining_with_swizzling.cu -o main && ./main   
```
105 % of cublas performance

## [Software pipelining/shared memory swizzling/C shared memory staging][11_pipeling_swizzling_c_smem_staging.cu]
`11_pipelining_with_swizzling.cu`
Software pipeling + shared memory swizzling + C shared memory staging. This is the kernel with the highest performance.

```
nvcc  -lcublas -arch=sm_86 -O3 11_pipelining_with_swizzling.cu -o main && ./main   
```
120 % of cublas performance

## [BONUS: how much performance do we lose by not using the ldmatrix PTX instructions?][12_fastest_without_ldmatrix.cu]
`12_fastest_without_ldmatrix.cu` 
Takes fastest implementation and replaces usage of the ldmatrix PTX instruction by manually reading the matrix fragments from shared memory.

```
nvcc  -lcublas -arch=sm_86 -O3 12_fastest_without_ldmatrix.cu -o main && ./main 
```
100 % of cublas performance
