# FlagQuantum Jiuding runtime image

This image is the small, reproducible environment for NVIDIA GPU jobs. It adds
only FlagQuantum and Jiuding's required SSH service to an existing CUDA-enabled
PyTorch runtime. Development tools, JAX, test dependencies, compilers, and
benchmark artifacts are intentionally excluded.

## Build the wheel

Build from a clean integration commit so unfinished working-tree files do not
enter a remote job:

```bash
python -m pip wheel . \
  --no-build-isolation \
  --no-deps \
  --wheel-dir dist
```

The expected artifact for the current package version is:

```text
dist/flagquantum-0.2.0-py3-none-any.whl
```

## Build locally

```bash
docker build \
  --file docker/runtime/Dockerfile \
  --tag flagquantum-runtime:0.2.0 .

docker run --rm --gpus all flagquantum-runtime:0.2.0 \
  python -c 'import flagquantum as fq, torch; print(fq.__version__, torch.cuda.get_device_name(0))'
```

The default base is
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. Pin a validated digest for a
reproducible production image.

## Build on Jiuding

1. Upload `flagquantum-0.2.0-py3-none-any.whl` to the personal storage root.
2. In **AI asset management -> Image management -> Add image**, select
   **Build from Dockerfile**. The current form expects a complete Dockerfile,
   including `FROM`.
3. Set image name to `flagquantum-runtime` and tag to `0.2.0-cu128-a100`.
4. Select GPU, PyTorch, Beijing Daxing, and availability zone A.
5. Paste the complete contents of `Dockerfile.jiuding` into the Dockerfile
   field. It uses a CUDA-enabled PyTorch *runtime* base compatible with Python
   3.10--3.12 and deliberately avoids a larger `devel` image.
6. Build the image, then run this bounded smoke command on one A100:

```bash
python -c "import json,flagquantum as fq,torch; assert torch.cuda.is_available(); print(json.dumps({'flagquantum':fq.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"
```

Jiuding requires SSH support even for job images, so `openssh-server` is the
only operating-system package added here. The Dockerfile does not replace the
base image's entrypoint or modify platform-provided `NVIDIA_*`, `CUDA_*`,
`NCCL_*`, `RANK`, `MASTER_*`, or `AIRS_*` environment variables.

This runtime image is suitable for functional and performance development. It
is not, by itself, release evidence for a benchmark or scalability claim.

The `v0.2.0-ef3affbd-cu128-a100` private build includes the resident Jiuding
workspace executor. Its clean-build wheel SHA-256 is
`f5eac87203950659f78b1342db20f0193882cd1fb07f9002dc7f835c047098e0`.
Pure-image startup and warm Bell-state measurements are recorded in
[`jiuding_warm_executor_image_20260909.json`](../../docs/development/evidence/jiuding_warm_executor_image_20260909.json).

The `v0.2.0-7b588ea5-cu128-a100` build adds resident GPU-side probability and
Pauli-expectation reduction. Its wheel SHA-256 is
`3f8d0b4d1cef2fca0276cf8deb3ee81bede2050cddff0e0db94a94f7c1a205c3`; live
measurements are recorded in
[`jiuding_remote_measurements_20260909.json`](../../docs/development/evidence/jiuding_remote_measurements_20260909.json).

The `v0.2.0-59a517cd-cu128-a100` build adds resident computational- and
Pauli-basis samples and counts. Its wheel SHA-256 is
`ae46789659b29acc884fe2b215a7d2eeeb44532fe8281196907c75be8834c4f1`; live
results and the device-to-host boundary are recorded in
[`jiuding_remote_sampling_20260909.json`](../../docs/development/evidence/jiuding_remote_sampling_20260909.json).

The `v0.2.0-e30b1b0c-cu128-a100` build adds bounded resident measurement
batches. Its wheel SHA-256 is
`9579bad42d56ed0aed8aff51f16a4c597d244a952480362600327062ba9817b3`; live
single-call and batch timings are recorded in
[`jiuding_remote_batch_20260909.json`](../../docs/development/evidence/jiuding_remote_batch_20260909.json).
