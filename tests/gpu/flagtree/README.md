# FlagTree kernel compatibility lane

This lane proves that the shared sources under `flagquantum/kernels/triton/`
compile and execute when the `triton` Python namespace is owned by the
`flagtree` distribution. It is a compiler-substitution gate, not a second
implementation catalog.

The pinned image supplies Python 3.12, PyTorch 2.13.0 with CUDA 12.9, and GLIBC
2.39. The build removes stock Triton before installing the NVIDIA FlagTree
wheel. Current FlagTree NVIDIA wheels require GLIBC 2.38 or newer, so installing
them directly on the Ubuntu 22.04 A800 hosts is not supported by this lane.

From the repository root, build and run the lane with:

```bash
docker build \
  --file tests/gpu/flagtree/Dockerfile \
  --tag flagquantum-flagtree-kernels:0.7.0 \
  tests/gpu/flagtree

docker run --rm --gpus all \
  --volume "$PWD:/workspace:ro" \
  --workdir /workspace \
  --env PYTHONPATH=/workspace \
  flagquantum-flagtree-kernels:0.7.0 \
  /usr/bin/python3 -m pytest @tests/gpu/flagtree/suite.txt
```

The identity tests fail closed unless package metadata reports `flagtree` as
the sole owner of the `triton` namespace and the active compiler backend is
CUDA. The remaining suite executes every maintained in-repository Triton kernel
family against its existing correctness, gradient, and fallback contracts.

Record the image digest, FlagTree version, FlagQuantum commit, GPU, driver,
PyTorch version, command, and complete pytest result for each hardware run.
