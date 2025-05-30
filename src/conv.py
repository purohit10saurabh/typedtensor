from __future__ import annotations

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import jaxtyped

from .typed import ImageTensor, Sequence1DTensor, TypedTensor, VolumeTensor


@jaxtyped(typechecker=beartype)
def conv2d(
    x: ImageTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> ImageTensor:
    return ImageTensor(F.conv2d(x, weight, bias, stride, padding, dilation, groups))


@jaxtyped(typechecker=beartype)
def conv_transpose2d(
    x: ImageTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1
) -> ImageTensor:
    return ImageTensor(F.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation))


@jaxtyped(typechecker=beartype)
def conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)


@jaxtyped(typechecker=beartype)
def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    return F.conv3d(x, weight, bias, stride, padding, dilation, groups)


@jaxtyped(typechecker=beartype)
def conv2d_typed(
    x: TypedTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> TypedTensor:
    result = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    return TypedTensor(result, tensor_type=x.get_tensor_type())


@jaxtyped(typechecker=beartype)
def conv1d_typed(
    x: Sequence1DTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> Sequence1DTensor:
    result = F.conv1d(x, weight, bias, stride, padding, dilation, groups)
    return Sequence1DTensor(result)


@jaxtyped(typechecker=beartype)
def conv3d_typed(
    x: VolumeTensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> VolumeTensor:
    result = F.conv3d(x, weight, bias, stride, padding, dilation, groups)
    return VolumeTensor(result) 