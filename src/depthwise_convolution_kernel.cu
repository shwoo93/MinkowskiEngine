/* 
Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

#ifndef GPU_DEPTHWISE_CONVOLUTION
#define GPU_DEPTHWISE_CONVOLUTION

#include <iostream>

#include "allocators.cuh"
#include "convolution_kernel.cuh"
#include "math_functions.cuh"
#include <stdio.h>

#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>

namespace minkowski {

namespace detail {

template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmulDwconv(const Dtype *__restrict__ A, const int wA, const int hA, //
             const Dtype *__restrict__ B, const int wB, const int hB, //
             Dtype *__restrict__ C,                                   //
             const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;
  
  if (y < hA && x < wB)
    atomicAdd(&C[wB * out_row + x], A[wA * in_row + x] * B[x]);
}
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
 * wA is A's width and wB is B's width
 *
 *                +---+
 *                |B^T|
 *            +-------+
 *            |   |   |
 *            | A | C |
 *            |   |   |
 *            |   |   |
 * +------------------+
 * |    D^T   | E |
 * +----------+---+
 *
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmulDwconv2(const Dtype *__restrict__ A, const int wA, const int hA, //
              const Dtype *__restrict__ B, const int wB, const int hB, //
              const Dtype *__restrict__ D, const int wD, const int hD, //
              Dtype *__restrict__ C, Dtype *__restrict__ E,
              const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
    // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Coordinate. y is for rows, x is for columns.
    const int x = BLOCK_SIZE * bx + tx;
    const int y = BLOCK_SIZE * by + ty;

    const Itype in_row = y < hA ? in_map[y] : 0;
    const Itype out_row = y < hA ? out_map[y] : 0;

    Dtype Esub = 0;
    __shared__ Dtype Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ Dtype Dsub[BLOCK_SIZE][BLOCK_SIZE];

    Asub[ty][tx] = (y < hA && x < wA) ? A[wA * out_row + x] : 0;
    Dsub[ty][tx] = (y < hD && x < wD) ? D[wD * in_row + x] : 0;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        Esub += Asub[i][tx] * Dsub[i][ty];
    }
    __syncthreads();

    if (tx == ty && x < wB)
        atomicAdd(&E[x], Esub);
    
    if (y < hA && x < wA)
        atomicAdd(&C[wA * in_row + x], A[wA * out_row + x] * B[x]);
#ifdef DEBUG
    // printf("Blk: (%d,%d), Thread: (%d,%d), -> Row/Col = (%d,%d)\n",
    //         bx, by,
    //         tx, ty,
    //         x, y);
    printf("grad_out: (%d), kernel: (%d), input: (%d)",
           wA * out_row + x,
           x,
           wD * in_row + x);
#endif
}
} // namespace detail

template <typename Dtype, typename Itype, typename ByteAllocator>
void DepthwiseConvolutionForwardKernelGPU(
    Dtype const *d_in_feat,                      //
    default_types::size_type const in_nchannel,  //
    Dtype *d_out_feat,                           //
    default_types::size_type const out_nchannel, //
    Dtype *d_kernel, gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    default_types::size_type const in_nrows,      //
    default_types::size_type const out_nrows,     //
    ByteAllocator &allocator,                     //
    MinkowskiAlgorithm::Mode const algo_index,    //
    ConvolutionMode::Type const convolution_mode, //
    cublasHandle_t cuhandle, cudaStream_t stream) {

    size_t n_active_in_volume, shared_mem_size = -1;
    // Define the shared memory size
    if ((in_nchannel > 16 && out_nchannel > 16 &&
         in_nchannel * out_nchannel >= 512) ||
        (in_nchannel > 24 && out_nchannel > 24))
      shared_mem_size = 32;
    else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
      shared_mem_size = 24;
    else if ((in_nchannel > 8 && out_nchannel > 8) ||
             (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
      shared_mem_size = 16;
    else
      shared_mem_size = 8;

    dim3 threads(shared_mem_size, shared_mem_size);

    // Iterate through each spatial kernel and get indices for in_map and
    // out_map
    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
        auto const k = it->first;
        n_active_in_volume = kernel_map.size(k);
        if (n_active_in_volume == 0)
            continue;

        size_t const num_grid =
            (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
        size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
        size_t const step = (n_active_in_volume + num_div - 1) / num_div;

        for (size_t s = 0; s < num_div; s++) {
            size_t const offset = step * s;
            size_t const remainder = n_active_in_volume - offset;
            size_t const curr_num_active = remainder < step ? remainder : step;
            dim3 const grid((out_nchannel + threads.x - 1) / threads.x,
                            (curr_num_active + threads.y - 1) / threads.y);

            switch (shared_mem_size) {
            case 32:
                detail::matmulDwconv<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
                    d_in_feat, in_nchannel, curr_num_active,
                    &d_kernel[k * in_nchannel], out_nchannel,
                    1, d_out_feat, kernel_map.in_maps.begin(k) + offset,
                    kernel_map.out_maps.begin(k) + offset);
                break;
            case 24:
                detail::matmulDwconv<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
                    d_in_feat, in_nchannel, curr_num_active,
                    &d_kernel[k * in_nchannel], out_nchannel,
                    1, d_out_feat, kernel_map.in_maps.begin(k) + offset,
                    kernel_map.out_maps.begin(k) + offset);
                break;
            case 16:
                detail::matmulDwconv<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
                    d_in_feat, in_nchannel, curr_num_active,
                    &d_kernel[k * in_nchannel], out_nchannel,
                    1, d_out_feat, kernel_map.in_maps.begin(k) + offset,
                    kernel_map.out_maps.begin(k) + offset);
                break;
            case 8:
                detail::matmulDwconv<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
                    d_in_feat, in_nchannel, curr_num_active,
                    &d_kernel[k * in_nchannel], out_nchannel,
                    1, d_out_feat, kernel_map.in_maps.begin(k) + offset,
                    kernel_map.out_maps.begin(k) + offset);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

// default_allocator
template void
DepthwiseConvolutionForwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    float const *d_in_feat, default_types::size_type const in_nchannel,
    float *d_out_feat, default_types::size_type const out_nchannel,
    float *d_kernel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
DepthwiseConvolutionForwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    double const *d_in_feat, default_types::size_type const in_nchannel,
    double *d_out_feat, default_types::size_type const out_nchannel,
    double *d_kernel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// c10_allocator
template void
DepthwiseConvolutionForwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    float const *d_in_feat, default_types::size_type const in_nchannel,
    float *d_out_feat, default_types::size_type const out_nchannel,
    float *d_kernel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::c10_allocator<char> &allocator,    //
    MinkowskiAlgorithm::Mode const algo_index, //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
DepthwiseConvolutionForwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    double const *d_in_feat, default_types::size_type const in_nchannel,
    double *d_out_feat, default_types::size_type const out_nchannel,
    double *d_kernel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::c10_allocator<char> &allocator,    //
    MinkowskiAlgorithm::Mode const algo_index, //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// Backward
template <typename Dtype, typename Itype, typename ByteAllocator>
void DepthwiseConvolutionBackwardKernelGPU(
    Dtype const *d_in_feat,                                            //
    Dtype *d_grad_in_feat, default_types::size_type const in_nchannel, //
    Dtype const *d_grad_out_feat,                                      //
    default_types::size_type const out_nchannel,                       //
    Dtype const *d_kernel, Dtype *d_grad_kernel,                       //
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,            //
    default_types::size_type const in_nrows,                           //
    default_types::size_type const out_nrows,                          //
    ByteAllocator &allocator,                                          //
    MinkowskiAlgorithm::Mode const algo_index,                         //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream) {

#ifdef DEBUG
  CUDA_CHECK_ARGS(cudaDeviceSynchronize(),
                  "Error triggered from a previous kernel call.");
#endif

    size_t n_active_in_volume, shared_mem_size = -1;
    // Define the shared memory size
    if ((in_nchannel > 16 && out_nchannel > 16 &&
        in_nchannel * out_nchannel >= 512) ||
        (in_nchannel % 32 == 0 && out_nchannel % 32 == 0))
        shared_mem_size = 32;
    else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
        shared_mem_size = 24;
    else if ((in_nchannel > 8 && out_nchannel > 8) ||
            (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
        shared_mem_size = 16;
    else
        shared_mem_size = 8;

    dim3 threads(shared_mem_size, shared_mem_size);

    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
        auto const k = it->first;
        n_active_in_volume = kernel_map.size(k);
        if (n_active_in_volume == 0)
            continue;

        size_t const num_grid =
            (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
        size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
        size_t const step = (n_active_in_volume + num_div - 1) / num_div;

        for (int s = 0; s < num_div; s++) {
            size_t const offset = step * s;
            size_t const remainder = n_active_in_volume - offset;
            size_t const curr_num_active = remainder < step ? remainder : step;
            dim3 const grid((in_nchannel + threads.x - 1) / threads.x,
                            (curr_num_active + threads.y - 1) / threads.y);

#ifdef DEBUG
        // /*
        // size_t map_size = curr_num_active;
        // Itype *p_kernel_map = (Itype *)std::malloc(map_size * 3 *
        // sizeof(Itype)); CUDA_CHECK(cudaMemcpy(p_kernel_map,
        // kernel_map.kernels.begin(k), map_size * sizeof(Itype),
        //                       cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(p_kernel_map + 1 * map_size,
        //                       kernel_map.in_maps.begin(k),
        //                       map_size * sizeof(Itype),
        // cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(p_kernel_map + 2 *
        // map_size, kernel_map.out_maps.begin(k), map_size * sizeof(Itype),
        // cudaMemcpyDeviceToHost));

        // for (size_t i = curr_num_active - 20; i < curr_num_active; ++i) {
        //   std::cout << p_kernel_map[i + 0 * map_size] << ":"
        //             << p_kernel_map[i + 1 * map_size] << "->"
        //             << p_kernel_map[i + 2 * map_size] << "\n";
        // }

        // CUDA_CHECK(cudaDeviceSynchronize());
        // std::free(p_kernel_map);
        // */
#endif

            switch (shared_mem_size) {
                case 32:
                    detail::matmulDwconv2<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
                        d_grad_out_feat, out_nchannel, curr_num_active, // A, wA, hA
                        &d_kernel[k * in_nchannel], out_nchannel, 1,    // B, wB, hB
                        d_in_feat, in_nchannel, curr_num_active,        // D, wD, hD
                        d_grad_in_feat,                                 // C
                        &d_grad_kernel[k * in_nchannel],                // E
                        kernel_map.in_maps.begin(k) + offset,
                        kernel_map.out_maps.begin(k) + offset);
                    break;
                case 24:
                    detail::matmulDwconv2<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
                        d_grad_out_feat, out_nchannel, curr_num_active, // A, wA, hA
                        &d_kernel[k * in_nchannel], out_nchannel, 1,    // B, wB, hB
                        d_in_feat, in_nchannel, curr_num_active,        // D, wD, hD
                        d_grad_in_feat,                                 // C
                        &d_grad_kernel[k * in_nchannel],                // E
                        kernel_map.in_maps.begin(k) + offset,
                        kernel_map.out_maps.begin(k) + offset);
                    break;
                case 16:
                    detail::matmulDwconv2<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
                        d_grad_out_feat, out_nchannel, curr_num_active, // A, wA, hA
                        &d_kernel[k * in_nchannel], out_nchannel, 1,    // B, wB, hB
                        d_in_feat, in_nchannel, curr_num_active,        // D, wD, hD
                        d_grad_in_feat,                                 // C
                        &d_grad_kernel[k * in_nchannel],                // E
                        kernel_map.in_maps.begin(k) + offset,
                        kernel_map.out_maps.begin(k) + offset);
                    break;
                case 8:
                    detail::matmulDwconv2<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
                        d_grad_out_feat, out_nchannel, curr_num_active, // A, wA, hA
                        &d_kernel[k * in_nchannel], out_nchannel, 1,    // B, wB, hB
                        d_in_feat, in_nchannel, curr_num_active,        // D, wD, hD
                        d_grad_in_feat,                                 // C
                        &d_grad_kernel[k * in_nchannel],                // E
                        kernel_map.in_maps.begin(k) + offset,
                        kernel_map.out_maps.begin(k) + offset);
                    break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// default_allocator
template void
DepthwiseConvolutionBackwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    float const *d_in_feat, float *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    float const *d_grad_out_feat,
    default_types::size_type const out_nchannel, //
    float const *d_kernel, float *p_grad_kernel, //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map,                            //
    default_types::size_type const in_nrows,    //
    default_types::size_type const out_nrows,   //
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
DepthwiseConvolutionBackwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    double const *d_in_feat, double *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    double const *d_grad_out_feat,
    default_types::size_type const out_nchannel,   //
    double const *d_kernel, double *p_grad_kernel, //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map,                            //
    default_types::size_type const in_nrows,    //
    default_types::size_type const out_nrows,   //
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// c10_allocator
template void
DepthwiseConvolutionBackwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    float const *d_in_feat, float *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    float const *d_grad_out_feat,
    default_types::size_type const out_nchannel,                             //
    float const *d_kernel, float *p_grad_kernel,                             //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    default_types::size_type const in_nrows,                                 //
    default_types::size_type const out_nrows,                                //
    detail::c10_allocator<char> &allocator,                                  //
    MinkowskiAlgorithm::Mode const algo_index,                               //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
DepthwiseConvolutionBackwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    double const *d_in_feat, double *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    double const *d_grad_out_feat,
    default_types::size_type const out_nchannel,                             //
    double const *d_kernel, double *p_grad_kernel,                           //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    default_types::size_type const in_nrows,                                 //
    default_types::size_type const out_nrows,                                //
    detail::c10_allocator<char> &allocator,                                  //
    MinkowskiAlgorithm::Mode const algo_index,                               //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

} // namespace minkowski
#endif // end GPU_DEPTHWISE_CONVOLUTION