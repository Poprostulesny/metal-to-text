#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv-train}"
PYTHON="${PYTHON:-python3.11}"
BUILD_JOBS="${BUILD_JOBS:-10}"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip setuptools wheel
"$VENV/bin/python" -m pip install pybind11 ninja
"$VENV/bin/python" -m pip install "cuda-python==12.6.2.post1"

# Put torch in the environment first. NeMo recommends this order, and
# Transformer Engine's PyTorch extension requires torch during metadata/build.
"$VENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.7.1" "torchaudio==2.7.1"

mapfile -t SITE_PACKAGES < <("$VENV/bin/python" - <<'PY'
import site
import sysconfig

paths = []
for key in ("purelib", "platlib"):
    path = sysconfig.get_paths().get(key)
    if path:
        paths.append(path)
paths.extend(site.getsitepackages())

seen = set()
for path in paths:
    if path not in seen:
        print(path)
        seen.add(path)
PY
)

shopt -s nullglob
CUDA_INCLUDE_FLAGS=()
CUDA_LIB_PATHS=()
for site_dir in "${SITE_PACKAGES[@]}"; do
  [ -d "$site_dir" ] || continue

  if [ -d "$site_dir/nvidia/cudnn" ]; then
    export CUDNN_PATH="$site_dir/nvidia/cudnn"
    export CUDNN_HOME="$CUDNN_PATH"
  fi

  for include_dir in "$site_dir"/nvidia/*/include "$site_dir"/*/include; do
    CUDA_INCLUDE_FLAGS+=("-I$include_dir")
  done
  for lib_dir in "$site_dir"/nvidia/*/lib "$site_dir"/*/lib; do
    CUDA_LIB_PATHS+=("$lib_dir")
  done
done
shopt -u nullglob

if [ "${#CUDA_INCLUDE_FLAGS[@]}" -gt 0 ]; then
  export CFLAGS="${CUDA_INCLUDE_FLAGS[*]} ${CFLAGS:-}"
  export CXXFLAGS="${CUDA_INCLUDE_FLAGS[*]} ${CXXFLAGS:-}"
fi

if [ "${#CUDA_LIB_PATHS[@]}" -gt 0 ]; then
  CUDA_LD_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}")"
  export LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi

export MAX_JOBS="$BUILD_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export NINJA_NUM_JOBS="$BUILD_JOBS"
export Ninja_NUM_JOBS="$BUILD_JOBS"

"$VENV/bin/python" -m pip install --no-build-isolation -r requirements-train.txt
"$VENV/bin/python" -m pip install --no-build-isolation -r requirements-train-transformer-engine.txt

"$VENV/bin/python" - <<'PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

import nemo.collections.asr as nemo_asr
print("nemo asr import ok", nemo_asr.__name__)

import transformer_engine
print("transformer_engine", transformer_engine.__version__)
PY
