# Conv2D Operations with Typed Tensors

This module provides type-safe 2D convolution operations that work with typed tensors, particularly `ImageTensor`. All functions use static type checking with `jaxtyping` and `beartype`, and return typed tensor objects that preserve type information throughout the computation pipeline.

## Features

- **Typed tensor inputs/outputs**: All main functions accept and return `ImageTensor` objects
- **Type preservation**: Convolution results maintain tensor type information
- **Multiple convolution variants**: Standard, depthwise, pointwise, separable, grouped, dilated, and transposed convolutions
- **Convenience functions**: High-level operations for common use cases (preserve size, downsample, upsample)
- **Raw tensor compatibility**: Fallback functions for working with raw tensors
- **Runtime type checking**: Automatic validation via `beartype`

## Main Functions (ImageTensor → ImageTensor)

### Basic Convolutions

#### `conv2d(x: ImageTensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> ImageTensor`
Standard 2D convolution operation with ImageTensor input and output.

**Parameters:**
- `x`: Input ImageTensor `[batch, in_channels, height, width]`
- `weight`: Convolution kernel `[out_channels, in_channels, kernel_h, kernel_w]`
- `bias`: Optional bias tensor `[out_channels]`
- `stride`, `padding`, `dilation`, `groups`: Standard conv2d parameters

**Returns:** `ImageTensor` with shape `[batch, out_channels, out_height, out_width]`

#### `conv2d_same_padding(x: ImageTensor, weight, bias=None, stride=1, dilation=1, groups=1) -> ImageTensor`
Convolution with automatic "same" padding to preserve spatial dimensions.

**Returns:** `ImageTensor` with same spatial size as input

### Specialized Convolutions

#### `depthwise_conv2d(x: ImageTensor, weight, bias=None, stride=1, padding=0, dilation=1) -> ImageTensor`
Depthwise convolution where each input channel is convolved separately.

#### `pointwise_conv2d(x: ImageTensor, weight, bias=None) -> ImageTensor`
1×1 convolution for channel mixing without spatial processing.

#### `separable_conv2d(x: ImageTensor, depthwise_weight, pointwise_weight, ...) -> ImageTensor`
Efficient separable convolution combining depthwise and pointwise operations.

#### `grouped_conv2d(x: ImageTensor, weight, bias=None, groups=1, ...) -> ImageTensor`
Grouped convolution for efficient processing with channel grouping.

### Advanced Convolutions

#### `dilated_conv2d(x: ImageTensor, weight, bias=None, stride=1, padding=0, dilation=2) -> ImageTensor`
Dilated (atrous) convolution for expanded receptive fields.

#### `conv2d_transpose(x: ImageTensor, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1) -> ImageTensor`
Transposed convolution (deconvolution) for upsampling.

## Convenience Functions

#### `conv2d_preserve_size(img: ImageTensor, weight, bias=None) -> ImageTensor`
Convolution that automatically preserves spatial dimensions (alias for `conv2d_same_padding`).

#### `conv2d_downsample(img: ImageTensor, weight, bias=None, factor=2) -> ImageTensor`
Convolution that downsamples by a given factor.

#### `conv2d_upsample(img: ImageTensor, weight, bias=None, factor=2) -> ImageTensor`
Transposed convolution that upsamples by a given factor.

## Flexible Functions

#### `conv2d_typed_tensor(x: TypedTensor, weight, ...) -> TypedTensor`
Convolution with generic TypedTensor (preserves original tensor type).

#### `conv2d_raw(x: torch.Tensor, weight, ...) -> torch.Tensor`
Raw tensor convolution for backward compatibility.

## Usage Examples

### Basic Usage with ImageTensor
```python
import torch
from conv2d import conv2d, conv2d_same_padding
from typed import ImageTensor

# Create ImageTensor
img = ImageTensor(torch.randn(2, 3, 32, 32))
weight = torch.randn(16, 3, 3, 3)

# Standard convolution - returns ImageTensor
result = conv2d(img, weight, padding=1)
print(f"Output channels: {result.channels}")  # 16
print(f"Output size: {result.height}x{result.width}")  # 32x32

# Preserve spatial dimensions
result_same = conv2d_same_padding(img, weight)
print(f"Preserved size: {result_same.height}x{result_same.width}")  # 32x32
```

### Convenience Functions
```python
from conv2d import conv2d_preserve_size, conv2d_downsample, conv2d_upsample

img = ImageTensor(torch.randn(1, 3, 64, 64))
weight = torch.randn(16, 3, 3, 3)

# Preserve size (same as conv2d_same_padding)
preserved = conv2d_preserve_size(img, weight)

# Downsample by factor of 2
downsampled = conv2d_downsample(img, weight, factor=2)
print(f"Downsampled: {downsampled.height}x{downsampled.width}")  # 32x32

# Upsample by factor of 2
upsample_weight = torch.randn(3, 16, 4, 4)
upsampled = conv2d_upsample(img, upsample_weight, factor=2)
print(f"Upsampled: {upsampled.height}x{upsampled.width}")  # ~128x128
```

### Separable Convolution
```python
from conv2d import separable_conv2d

img = ImageTensor(torch.randn(1, 32, 64, 64))
depthwise_weight = torch.randn(32, 1, 3, 3)
pointwise_weight = torch.randn(64, 32, 1, 1)

result = separable_conv2d(img, depthwise_weight, pointwise_weight, padding=1)
print(f"Separable result: {result.channels} channels")  # 64
```

### Working with Generic TypedTensors
```python
from conv2d import conv2d_typed_tensor
from typed import TypedTensor

# Create a feature map tensor
feature_map = TypedTensor(torch.randn(1, 8, 16, 16), tensor_type="feature_map")
weight = torch.randn(32, 8, 3, 3)

result = conv2d_typed_tensor(feature_map, weight, padding=1)
print(f"Result type: {result.get_tensor_type()}")  # "feature_map"
```

### Raw Tensor Compatibility
```python
from conv2d import conv2d_raw

# For backward compatibility or when you need raw tensors
x = torch.randn(2, 3, 32, 32)
weight = torch.randn(16, 3, 3, 3)

result = conv2d_raw(x, weight, padding=1)  # Returns torch.Tensor
```

## Type Safety Benefits

The typed tensor approach provides several advantages:

1. **Type Preservation**: Results maintain tensor type information
2. **Property Access**: Easy access to tensor properties like `channels`, `height`, `width`
3. **Pipeline Consistency**: All operations return compatible typed tensors
4. **Runtime Validation**: Automatic shape and type checking

```python
img = ImageTensor(torch.randn(1, 3, 224, 224))
print(f"Input: {img.channels} channels, {img.height}x{img.width}")

result = conv2d(img, torch.randn(64, 3, 7, 7), stride=2, padding=3)
print(f"Output: {result.channels} channels, {result.height}x{result.width}")

# Chain operations naturally
result2 = conv2d(result, torch.randn(128, 64, 3, 3), padding=1)
print(f"Chained: {result2.channels} channels")
```

## Performance Considerations

- **Separable convolutions** are more efficient than standard convolutions for large channel counts
- **Grouped convolutions** reduce computational complexity by processing channels in groups
- **Dilated convolutions** expand receptive fields without increasing parameter count
- **Convenience functions** automatically calculate appropriate parameters

## Testing

Run the test suite to verify all operations:

```bash
python test_conv2d.py
```

The tests cover:
- Basic convolution operations with ImageTensor
- Specialized convolution variants
- Convenience functions
- Type preservation and safety
- Raw tensor compatibility
- Performance comparisons 