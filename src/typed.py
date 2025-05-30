from __future__ import annotations

from typing import Generic, Optional, Tuple, TypeVar, Union

import torch
from beartype import beartype
from jaxtyping import Float, Integer, jaxtyped

# Type variables for shape dimensions
B = TypeVar('B')  # Batch size
C = TypeVar('C')  # Channels
H = TypeVar('H')  # Height
W = TypeVar('W')  # Width
T = TypeVar('T')  # Time/sequence length
D = TypeVar('D')  # Feature dimensions


class TypedTensor(torch.Tensor):
    """
    A subclass of torch.Tensor that allows for shape-aware type annotations.
    This class can be used to create tensors with specific shapes for static type checking.
    """
    _shape_info: Optional[str] = None
    _tensor_type: Optional[str] = None
    
    def __new__(cls, data: Union[torch.Tensor, list], tensor_type: str = "generic"):
        # Create tensor from data
        tensor = torch.as_tensor(data)
        # Create instance
        instance = super().__new__(cls, tensor)
        instance._tensor_type = tensor_type
        instance._shape_info = str(tensor.shape)
        return instance

    def __init__(self, data: Union[torch.Tensor, list], tensor_type: str = "generic"):
        super().__init__()
        
    def get_shape(self) -> torch.Size:
        """Returns the shape of the tensor."""
        return self.shape
    
    def get_tensor_type(self) -> str:
        """Returns the tensor type (image, sequence, scalar, etc.)."""
        return self._tensor_type or "generic"
    
    def __repr__(self) -> str:
        return f"TypedTensor(shape={self.shape}, type={self._tensor_type}, dtype={self.dtype})"
        
    def clone(self) -> 'TypedTensor':
        """Override clone to preserve type information."""
        new_tensor = super().clone()
        return TypedTensor(new_tensor, tensor_type=self._tensor_type or "generic")


class ImageTensor(TypedTensor):
    def __new__(cls, data: Union[torch.Tensor, list]):
        tensor = torch.as_tensor(data)
        if len(tensor.shape) != 4:
            raise ValueError(f"ImageTensor requires 4D tensor (B, C, H, W), got {tensor.shape}")
        instance = super().__new__(cls, tensor, "image")
        return instance
    
    @property
    def batch_size(self) -> int:
        return self.shape[0]
    
    @property
    def channels(self) -> int:
        return self.shape[1]
    
    @property
    def height(self) -> int:
        return self.shape[2]
    
    @property
    def width(self) -> int:
        return self.shape[3]


class SequenceTensor(TypedTensor):
    def __new__(cls, data: Union[torch.Tensor, list]):
        tensor = torch.as_tensor(data)
        if len(tensor.shape) != 3:
            raise ValueError(f"SequenceTensor requires 3D tensor (B, T, D), got {tensor.shape}")
        instance = super().__new__(cls, tensor, "sequence")
        return instance
    
    @property
    def batch_size(self) -> int:
        return self.shape[0]
    
    @property
    def sequence_length(self) -> int:
        return self.shape[1]
    
    @property
    def feature_dim(self) -> int:
        return self.shape[2]


class ScalarTensor(TypedTensor):
    def __new__(cls, data: Union[torch.Tensor, list]):
        tensor = torch.as_tensor(data)
        if len(tensor.shape) != 2 or tensor.shape[1] != 1:
            raise ValueError(f"ScalarTensor requires 2D tensor (B, 1), got {tensor.shape}")
        instance = super().__new__(cls, tensor, "scalar")
        return instance
    
    @property
    def batch_size(self) -> int:
        return self.shape[0]



@jaxtyped(typechecker=beartype)
def create_image_tensor(data: Float[torch.Tensor, "batch channels height width"]) -> ImageTensor:
    return ImageTensor(data)


@jaxtyped(typechecker=beartype)
def create_sequence_tensor(data: Float[torch.Tensor, "batch time features"]) -> SequenceTensor:
    return SequenceTensor(data)


@jaxtyped(typechecker=beartype)
def create_scalar_tensor(data: Float[torch.Tensor, "batch 1"]) -> ScalarTensor:
    return ScalarTensor(data)