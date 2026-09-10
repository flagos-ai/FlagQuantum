# Jiuding runtime image

A compact CUDA/PyTorch environment for FlagQuantum jobs and resident workspace
execution. It includes Jiuding's SSH requirements and the toolchain needed by
Triton's first-use driver bootstrap.

## Build and check

From a clean integration checkout with the build prerequisites installed:

```bash
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
docker build --file docker/runtime/Dockerfile --tag flagquantum-runtime:local .
docker run --rm --gpus all flagquantum-runtime:local \
  python -c 'import flagquantum as fq, torch; print(fq.__version__, torch.cuda.get_device_name(0))'
```

Use a validated base-image digest and check the actual GPU path. Keep secrets
out of image layers and preserve platform-provided device and rank settings.
A runtime image alone is not benchmark or scalability evidence.

[Build and deployment guide](BUILD_GUIDE.md) contains Jiuding setup, dependency
requirements, versioned wheel hashes, and recorded validation. Those historical
build records remain unchanged.
