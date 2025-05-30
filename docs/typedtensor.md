# TypedTensor Project

## Overview
A Python library that extends `torch.Tensor` with shape-aware type annotations for static type checking. Enables catching tensor shape mismatches at edit-time rather than runtime.

## Features
- **TypedTensor**: Base class with shape-aware type annotations
- **ImageTensor**: For image data (batch, channels, height, width)
- **SequenceTensor**: For sequence data (batch, time, features)  
- **ScalarTensor**: For scalar data (batch, 1)
- **Static type checking** with `jaxtyping` and `beartype`
- **Runtime shape validation** 
- **`get_shape()` method** for accessing tensor dimensions

## Usage Examples

### Basic Usage
```python
from typed import ImageTensor, SequenceTensor, ScalarTensor

# Image tensor (batch=2, channels=3, height=32, width=32)
img = ImageTensor(torch.randn(2, 3, 32, 32))
print(f"Shape: {img.get_shape()}")
print(f"Channels: {img.channels}")

# Sequence tensor (batch=2, time=10, features=64)
seq = SequenceTensor(torch.randn(2, 10, 64))
print(f"Sequence length: {seq.sequence_length}")

# Scalar tensor (batch=2, values=1)
scalar = ScalarTensor(torch.randn(2, 1))
print(f"Batch size: {scalar.batch_size}")
```

### Static Type Checking
```python
from jaxtyping import Float, jaxtyped
from beartype import beartype

@jaxtyped(typechecker=beartype)
def conv2d_operation(
    x: Float[torch.Tensor, "batch channels height width"],
    weight: Float[torch.Tensor, "out_channels channels kernel_h kernel_w"]
) -> Float[torch.Tensor, "batch out_channels height width"]:
    return F.conv2d(x, weight, padding=1)

# Type-safe tensor creation
img_data: ImageFloat = torch.randn(1, 3, 32, 32)
img_tensor = create_image_tensor(img_data)  # Static type checking
```

## Implementation Status ✅
- [x] `TypedTensor` class extending `torch.Tensor`
- [x] Constructor with tensor and type parameters  
- [x] Different shape types (images, sequences, scalars)
- [x] Static type checking integration
- [x] `get_shape()` method implementation
- [x] Runtime shape validation
- [x] Comprehensive testing

## Dependencies
- `torch`
- `jaxtyping` 
- `beartype`
- `typing`
