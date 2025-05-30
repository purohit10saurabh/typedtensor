from __future__ import annotations

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped

from .typed import ImageTensor, TypedTensor


@jaxtyped(typechecker=beartype)
def conv2d(
    x: ImageTensor,
    weight: Float[torch.Tensor, "out_channels in_channels kernel_h kernel_w"],
    bias: Float[torch.Tensor, "out_channels"] | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> ImageTensor:
    return ImageTensor(F.conv2d(x, weight, bias, stride, padding, dilation, groups))


@jaxtyped(typechecker=beartype)
def conv_transpose2d(
    x: ImageTensor,
    weight: Float[torch.Tensor, "in_channels out_channels kernel_h kernel_w"],
    bias: Float[torch.Tensor, "out_channels"] | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1
) -> ImageTensor:
    return ImageTensor(F.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation))


@jaxtyped(typechecker=beartype)
def conv1d(
    x: Float[torch.Tensor, "batch channels length"],
    weight: Float[torch.Tensor, "out_channels in_channels kernel_size"],
    bias: Float[torch.Tensor, "out_channels"] | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> Float[torch.Tensor, "batch out_channels out_length"]:
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)


@jaxtyped(typechecker=beartype)
def conv3d(
    x: Float[torch.Tensor, "batch channels depth height width"],
    weight: Float[torch.Tensor, "out_channels in_channels kernel_d kernel_h kernel_w"],
    bias: Float[torch.Tensor, "out_channels"] | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> Float[torch.Tensor, "batch out_channels out_depth out_height out_width"]:
    return F.conv3d(x, weight, bias, stride, padding, dilation, groups)


@jaxtyped(typechecker=beartype)
def conv2d_typed(
    x: TypedTensor,
    weight: Float[torch.Tensor, "out_channels in_channels kernel_h kernel_w"],
    bias: Float[torch.Tensor, "out_channels"] | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> TypedTensor:
    result = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    return TypedTensor(result, tensor_type=x.get_tensor_type()) 