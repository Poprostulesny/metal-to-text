# Installing the CUDA 12.6 training stack

This project uses two separate Python environments:

1. `.venv-data` for downloading songs, lyrics cleanup and audio preprocessing.
2. `.venv-train` for NeMo / Parakeet training.

Do not merge them. The data stack and the training stack want different versions of core packages, especially `numpy`, `torch` and CUDA-related wheels.

## Quick install

From the project root:

```bash
python3.11 -m venv .venv-data
.venv-data/bin/python -m pip install -U pip setuptools wheel
.venv-data/bin/python -m pip install -r requirements-data.txt
```

For training, use the scripted three-stage install:

```bash
scripts/install_train_cuda126.sh
```

Equivalent manual commands:

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install -U pip setuptools wheel
.venv-train/bin/python -m pip install pybind11 ninja

.venv-train/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.7.1" "torchaudio==2.7.1"

# The script also discovers .venv-train's pip-installed NVIDIA headers/libs and
# exports CFLAGS, CXXFLAGS and LD_LIBRARY_PATH before native wheels are built.
.venv-train/bin/python -m pip install --no-build-isolation -r requirements-train.txt
.venv-train/bin/python -m pip install --no-build-isolation -r requirements-train-transformer-engine.txt
```

Do not install `requirements-train-transformer-engine.txt` into a fresh environment before `torch` is importable.

## Why this is split

NeMo's current speech installation guide recommends installing PyTorch first so CUDA wheels match the driver/runtime before NeMo resolves optional GPU packages.

Transformer Engine is even stricter. `transformer-engine` is a meta package; for PyTorch it pulls `transformer_engine_torch`, and that package often builds a local PyTorch extension. With `--no-build-isolation`, its metadata/build step imports `torch`. If `torch` is only listed earlier in the same requirements file, pip has not installed it yet when metadata is generated, so installation can fail with:

```text
RuntimeError: This package needs Torch to build.
```

The fix is not merely reordering lines in one requirements file. Install PyTorch first, install the NeMo ASR stack second, and install Transformer Engine third.

Some NeMo ASR dependencies, notably `texterrors`, also build native wheels. Their build scripts import `pybind11` while compiling. `pybind11` is still listed in `requirements-train.txt` for reproducibility, but it must be installed before the main `requirements-train.txt` command when using `--no-build-isolation`.

The install script also exposes pip-installed CUDA headers and libraries from `.venv-train/lib*/python*/site-packages`, including both `nvidia/*/lib` and package-specific paths such as `cusparselt/lib`. That helps local builds find cuDNN, NCCL, cuSPARSELt, CUDA runtime and related CUDA libraries without accidentally linking against a mismatched system path. The default native build parallelism is `10`; override it with:

```bash
BUILD_JOBS=4 scripts/install_train_cuda126.sh
```

## CUDA 12.6 and Numba RNNT loss

The Parakeet RNNT model uses NeMo's RNNT loss path, which may run Numba CUDA kernels. On CUDA 12.x, Numba JIT linking is sensitive to the driver/toolkit/linker combination.

For CUDA 12.6, keep these in the environment before Python imports NeMo/Numba:

```bash
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export STRICT_NUMBA_COMPAT_CHECK=0
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
```

`requirements-train.txt` includes:

```text
pynvjitlink-cu12
cuda-python==12.6.2.post1
```

`pynvjitlink-cu12` provides the CUDA 12 JIT linker package used by Numba minor-version compatibility. `cuda-python==12.6.2.post1` provides NVIDIA's CUDA Python bindings for `NUMBA_CUDA_USE_NVIDIA_BINDING=1`.

Do not let pip upgrade `cuda-python` to `13.x` while using `numba==0.61.x`. That Numba version imports the old `cuda.cuda` module, while CUDA Python 13 exposes the newer `cuda.bindings` namespace.

`parakeet_train.py` sets the three environment variables at process startup as a guardrail, but setting them in the shell is still useful for notebooks and one-off probes.

## Smoke tests

After installation:

```bash
.venv-train/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

import nemo.collections.asr as nemo_asr
print("nemo asr import ok")

import transformer_engine
print("transformer_engine", transformer_engine.__version__)
PY
```

Then check Numba's CUDA path:

```bash
NUMBA_CUDA_USE_NVIDIA_BINDING=1 \
STRICT_NUMBA_COMPAT_CHECK=0 \
NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 \
.venv-train/bin/python - <<'PY'
from nemo.core.utils import numba_utils

print(numba_utils.numba_cuda_is_supported(numba_utils.__NUMBA_MINIMUM_VERSION_FP16_SUPPORTED__))
print(numba_utils.is_numba_cuda_fp16_supported())
PY
```

If the Numba probe prints `False`, fix the CUDA/Numba linkage before starting RNNT training.

## If Transformer Engine still builds locally

A local build may still happen when NVIDIA has no matching prebuilt `transformer_engine_torch` wheel for the exact Python, PyTorch, CUDA and C++ ABI combination. For CUDA 12.6, make sure a normal CUDA toolkit with `nvcc` is visible:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
```

If your system compiler is too new for CUDA 12.6, install a supported GCC/G++ alongside the system compiler and set `CC`, `CXX` and `CUDAHOSTCXX` only for the Transformer Engine pip command. Do not replace the system compiler globally.

## Notes from the current workspace

In this sandbox, `nvcc --version` reports CUDA `12.6`, but `nvidia-smi` cannot communicate with the NVIDIA driver. That means install commands can be prepared here, but GPU availability and RNNT runtime probes must be verified in the real shell session where the driver is visible.
