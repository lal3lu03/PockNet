import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import torch
    print("PyTorch is available, version:", torch.__version__)
except ImportError:
    print("PyTorch not found")

try:
    import numpy
    print("NumPy is available, version:", numpy.__version__)
except ImportError:
    print("NumPy not found")

try:
    import pandas
    print("Pandas is available, version:", pandas.__version__)
except ImportError:
    print("Pandas not found")
