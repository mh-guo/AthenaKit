# Installation

## Requirements

- Python 3.8+
- NumPy, h5py, Matplotlib, SciPy, packaging

Optional (for GPU acceleration):
- [CuPy](https://cupy.dev/) — CUDA-backed array library (drop-in NumPy replacement on GPU)

Optional (for distributed analysis):
- `mpi4py`

## Install from source

```bash
git clone https://github.com/mh-guo/AthenaKit.git
cd AthenaKit
pip install -e .
```

## Enabling GPU support

Install CuPy for your CUDA version, then AthenaKit will automatically detect and use it:

```bash
pip install cupy-cuda12x   # adjust for your CUDA version
```

Check that GPU is detected:

```python
import athenakit
from athenakit import global_vars
print("GPU enabled:", global_vars.cupy_enabled)
```

## Enabling MPI support

```bash
pip install mpi4py
```

Run a script across 4 MPI ranks:

```bash
mpirun -n 4 python my_analysis.py
```
