# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch
from torch.autograd import Function
from torch.nn import Parameter

import MinkowskiEngineBackend._C as _C
from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType, ConvolutionMode
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiCommon import MinkowskiModuleBase
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiKernelGenerator import KernelGenerator

class MinkowskiDepthwiseConvolutionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        kernel_weights: torch.Tensor,
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if not input_features.is_cuda:
            raise NotImplementedError("Not implemented on the CPU")
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )

        input_features = input_features.contiguous()

        ctx.input_features = input_features
        ctx.kernel_weights = kernel_weights
        ctx.misc = [
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager
        ]
        return _C.DepthwiseConvolutionForwardGPU(
            ctx.input_features,
            kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        if not grad_out_feat.is_cuda:
            raise NotImplementedError("Not implemented on the CPU")
        grad_out_feat = grad_out_feat.contiguous()
        (
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        ) = ctx.misc

        grad_in_feat, grad_kernel =  _C.DepthwiseConvolutionBackwardGPU(
            ctx.input_features,
            grad_out_feat,
            ctx.kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )
        return (
            grad_in_feat,
            grad_kernel,
            None,
            None,
            None,
            None,
            None,
        )

class MinkowskiDepthwiseConvolution(MinkowskiModuleBase):

    def __init__(
        self,
        in_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=-1,
        use_cuda_kernel=True
    ):

        super(MinkowskiDepthwiseConvolution, self).__init__()
        assert (
            dimension > 0
        ), f"Invalid dimension, Please provide a valid dimension argument, dimension={dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
            )
        
        self.in_channels = in_channels
        
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_cuda_kernel = use_cuda_kernel

        Tensor = torch.FloatTensor
        kernel_shape = (kernel_generator.kernel_volume, self.in_channels)
        self.kernel = Parameter(Tensor(*kernel_shape))
        self.bias = Parameter(Tensor(1, in_channels)) if bias else None
        self.convolution_mode = convolution_mode
        self.conv = MinkowskiDepthwiseConvolutionFunction()
        self.reset_parameters()
    
    def reset_parameters(self, is_transpose=False):
        with torch.no_grad():
            n = (
                self.out_channels if is_transpose else self.in_channels
            ) * self.kernel_generator.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = "(in={}, region_type={}, ".format(
            self.in_channels, self.kernel_generator.region_type
        )
        if self.kernel_generator.region_type in [RegionType.CUSTOM]:
            s += "kernel_volume={}, ".format(self.kernel_generator.kernel_volume)
        else:
            s += "kernel_size={}, ".format(self.kernel_generator.kernel_size)
        s += "stride={}, dilation={})".format(
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_dilation,
        )
        return self.__class__.__name__ + s
    
    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.IntTensor, CoordinateMapKey, SparseTensor] = None,
    ):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        assert (
            self.in_channels == input.shape[1]
        ), f"Channel size mismatch {self.in_channels} != {input.shape[1]}"
        
        cm = input._manager
        out_key = _get_coordinate_map_key(
            input, coordinates, None
        )
        outfeat = self.conv.apply(
            input.F,
            self.kernel,
            self.kernel_generator,
            self.convolution_mode,
            input.coordinate_map_key,
            out_key,
            cm,
        )
        if self.bias is not None:
            outfeat += self.bias
        return SparseTensor(
            outfeat, 
            coordinate_map_key=out_key, 
            coordinate_manager=cm)

if __name__ == "__main__":
    in_channels, out_channels, D = 2, 2, 1
    coords = torch.IntTensor([[0, 0], [0, 1], [0, 2]]).to(0)
    feats = torch.FloatTensor([[0, 1], [1, 0], [1, 1]]).to(0)
    input = SparseTensor(feats, coordinates=coords)

    me_dwconv = MinkowskiDepthwiseConvolution(
        in_channels, kernel_size=2, stride=1, bias=False, dimension=D
    ).to(0)
    with torch.no_grad():
        me_dwconv.kernel[:] = torch.FloatTensor([[[1, 2], [2, 1]]]).to(0)
    # forward analytic test
    output = me_dwconv(input)
    print(input)
    print(output)
    """
    input
    01
    10
    11
    kernel
    12
    21
    output
    22
    31
    12
    """
    # backward analytic test
    in_key = input.coordinate_map_key
    cm = input.coordinate_manager
    out_key = output.coordinate_map_key
    kernel_generator = KernelGenerator(
        kernel_size=2,
        stride=1,
        dilation=1,
        dimension=D,
    )
    out_feat_grad = torch.ones_like(feats).to(0)
    in_feat_grad, kernel_grad = _C.DepthwiseConvolutionBackwardGPU(
        feats,
        out_feat_grad,
        me_dwconv.kernel,
        kernel_generator.kernel_size,
        kernel_generator.kernel_stride,
        kernel_generator.kernel_dilation,
        kernel_generator.region_type,
        kernel_generator.region_offsets,
        ConvolutionMode.DEFAULT,
        in_key,
        out_key,
        cm._manager,
    )
    print(in_feat_grad)
    print(kernel_grad)
    """
    in_grad
    1 2
    3 3
    3 3
    kernel_grad
    2 2
    2 1
    """