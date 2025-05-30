# TypedTensor

Type-safe PyTorch tensor operations with static type checking.

## Structure

```
typedtensor/
├── src/                    # Core library
│   ├── typed.py           # TypedTensor classes
│   ├── conv2d.py          # Convolution operations
│   └── __init__.py        # Package exports
├── tests/                 # Test files
│   ├── test_conv2d.py     # Convolution tests
│   ├── test_typed_tensor.py # TypedTensor tests
│   └── test_import.py     # Import tests
├── docs/                  # Documentation
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Usage

```python
from src import ImageTensor, conv2d
import torch

img = ImageTensor(torch.randn(1, 3, 224, 224))
weight = torch.randn(64, 3, 7, 7)
result = conv2d(img, weight, stride=2, padding=3)
```

## Testing

```bash
python -m pytest tests/
``` 