/* 
Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

#include <iostream>
#include "coordinate_map.hpp"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "depthwise_convolution_kernel.cuh"
#include "kernel_map.cuh"

// #include <ATen/ATen.h>
#include <ATen/cuda/CUDAUtils.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor DepthwiseConvolutionForwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

    ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
    ASSERT(kernel.is_contiguous(), "kernel must be contiguous");

    ASSERT(in_feat.is_cuda(), "in_feat must be CUDA");
    ASSERT(kernel.is_cuda(), "kernel must be CUDA");
    ASSERT(at::cuda::check_device({in_feat, kernel}),
            "in_feat and kernel must be on the same device");

    ASSERT(in_feat.scalar_type() == kernel.scalar_type(), "type mismatch");

    ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
    ASSERT(kernel.dim() == 2, "kernel.dim():", kernel.dim());

    ASSERT(in_feat.size(1) == kernel.size(1),
            "Input feature size and kernel size mismatch");
    
    coordinate_map_key_type in_key = p_in_map_key->get_key();
    ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

    ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
            in_feat.size(0), "!=", p_map_manager->size(in_key));

    if (!p_out_map_key->is_key_set()) {
        coordinate_map_key_type out_key =
                std::get<0>(p_map_manager->stride(in_key, kernel_stride));
        p_out_map_key->set_key(out_key);
    }

    auto const &in_out = p_map_manager->kernel_map(
      p_in_map_key,    //
      p_out_map_key,   //
      kernel_size,     //
      kernel_stride,   //
      kernel_dilation, //
      region_type,     //
      offset, false /* is_transpose */, false /* is_pool */);
    
    auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
    at::Tensor out_feat =
        torch::zeros({out_nrows, kernel.size(1)}, in_feat.options());
    LOG_DEBUG("Allocated", out_nrows, "x", kernel.size(1), "out_features.");

    LOG_DEBUG("DepthwiseConvolution on", out_nrows, "x", kernel.size(1));
    if (out_nrows > 0) {
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cublasSetStream(handle, stream);

        AT_DISPATCH_FLOATING_TYPES(
            in_feat.scalar_type(), "depthwise_convolution_forward_gpu", [&] {
                LOG_DEBUG("DepthwiseConvolutionForwardKernelGPU with",
                            std::is_same<float, scalar_t>::value ? "float" : "double");
                TemplatedAllocator<char> byte_allocator;
                DepthwiseConvolutionForwardKernelGPU<scalar_t, default_types::index_type,
                                            TemplatedAllocator<char>>(
                    in_feat.template data_ptr<scalar_t>(),  //
                    in_feat.size(1),                        //
                    out_feat.template data_ptr<scalar_t>(), //
                    out_feat.size(1),                       //
                    kernel.template data_ptr<scalar_t>(),   //
                    in_out,                                 //
                    in_feat.size(0),                        //
                    out_nrows,                              //
                    byte_allocator,                         //
                    p_map_manager->algorithm(),             //
                    convolution_mode,
                    handle, stream);
            }
        );
    }
    return  out_feat;
}

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> DepthwiseConvolutionBackwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor &grad_out_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  // ASSERT(grad_out_feat.is_contiguous(), "grad_out_feata must be contiguous");
  grad_out_feat = grad_out_feat.contiguous();
  ASSERT(kernel.is_contiguous(), "kernel must be contiguous");

  ASSERT(in_feat.is_cuda(), "in_feat must be CUDA");
  ASSERT(grad_out_feat.is_cuda(), "in_feat must be CUDA");
  ASSERT(kernel.is_cuda(), "kernel must be CUDA");
  ASSERT(at::cuda::check_device({in_feat, grad_out_feat, kernel}),
         "in_feat, grad_out_feat, kernel must be on the same device");

  ASSERT(in_feat.scalar_type() == kernel.scalar_type(), "type mismatch");
  ASSERT(in_feat.scalar_type() == grad_out_feat.scalar_type(), "type mismatch");

  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
  ASSERT(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());
  ASSERT(kernel.dim() == 2, "kernel.dim():", kernel.dim());

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  coordinate_map_key_type out_key = p_out_map_key->get_key();
  ASSERT(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

  auto const &in_out = p_map_manager->kernel_map(p_in_map_key,    //
                                                 p_out_map_key,   //
                                                 kernel_size,     //
                                                 kernel_stride,   //
                                                 kernel_dilation, //
                                                 region_type,     //
                                                 offset, false, false);

  at::Tensor grad_in_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());
  at::Tensor grad_kernel = torch::zeros(
      {kernel.size(0), kernel.size(1)}, kernel.options());

  if (in_feat.size(0) > 0) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "depthwise_convolution_backward_gpu", [&] {
          TemplatedAllocator<char> byte_allocator;
          DepthwiseConvolutionBackwardKernelGPU<scalar_t, default_types::index_type,
                                       TemplatedAllocator<char>>(
              in_feat.template data_ptr<scalar_t>(),       //
              grad_in_feat.template data_ptr<scalar_t>(),  //
              in_feat.size(1),                             //
              grad_out_feat.template data_ptr<scalar_t>(), //
              grad_out_feat.size(1),                       //
              kernel.template data_ptr<scalar_t>(),        //
              grad_kernel.template data_ptr<scalar_t>(),   //
              in_out,                                      //
              in_feat.size(0),                             //
              grad_out_feat.size(0),                       //
              byte_allocator,                              //
              p_map_manager->algorithm(),                  //
              convolution_mode,                            //
              handle, stream);
        });
  }

  return std::make_pair(grad_in_feat, grad_kernel);
}

// Forward
// default_allocator
template at::Tensor DepthwiseConvolutionForwardGPU<default_types::dcoordinate_type,
                                          detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10_allocator
template at::Tensor DepthwiseConvolutionForwardGPU<default_types::dcoordinate_type,
                                          detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

// Backward
// default_allocator
template std::pair<at::Tensor, at::Tensor>
DepthwiseConvolutionBackwardGPU<default_types::dcoordinate_type,
                       detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor &grad_out_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10_allocator
template std::pair<at::Tensor, at::Tensor>
DepthwiseConvolutionBackwardGPU<default_types::dcoordinate_type,
                       detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor &grad_out_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

} // end namespace minkowski