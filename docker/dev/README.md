# Development containers

The JAX-enabled variants' main Python environment installs FlagQuantum from the repository checkout,
JAX, plotting and example dependencies, and the Braket, PennyLane, Quafu and
Qiskit interoperability dependencies. A second environment in the same image
installs FlagQuantum, QSteed and `flagquantum-compiler-qsteed`.

| Tag | Platform | Numerical stack |
| --- | --- | --- |
| `cpu` | Linux amd64 | CPU PyTorch and JAX |
| `cuda-amd64` | Linux amd64 | PyTorch CUDA 12.8, JAX CUDA 12 and Triton |
| `cpu-no-jax` | Linux amd64 | CPU PyTorch and QSteed in one environment |
| `cuda-amd64-no-jax` | Linux amd64 | PyTorch CUDA 12.8, Triton and QSteed in one environment |

All four images target Linux AMD64. The pinned Quafu/QSteed dependencies do
not provide the required Linux ARM64 distributions, so these images do not
claim native ARM64 support. On Apple Silicon, CPU images require AMD64
emulation (`--platform linux/amd64`); GPU images require an NVIDIA Linux host.
Compose selects AMD64 explicitly for every variant.

The CUDA image includes CPU execution as well. Images install PyTorch 2.10.0;
other dependency constraints come from `pyproject.toml`. QSteed is installed from
the tested upstream commit in `requirements-qsteed.txt`, followed by the released
adapter (in `/opt/qsteed` for JAX-enabled variants). The compiler remains independently replaceable through FlagQuantum's
extension interface. No provider credentials are included.

## JupyterLab

All four variants include JupyterLab 4 and IPython kernels. Launch it from the
repository root (replace the profile/service for the desired variant):

```bash
docker compose -f compose.dev.yaml --profile cpu-no-jax run --rm \
  -p 127.0.0.1:8888:8888 dev-no-jax \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open the local URL with the token printed in the terminal. The checkout is
mounted at `/workspace`, so notebooks and edits persist on the host. Token
authentication remains enabled. On a remote host, access port 8888 through an
SSH tunnel.

Choose **FlagQuantum** for the main environment. JAX-enabled images also offer
**FlagQuantum (QSteed)** for the isolated compiler environment. No-JAX images
support training and QSteed in the main kernel. Each image build launches its
registered kernels and executes FlagQuantum code; compiler kernels also run
an offline QSteed compilation.

## Without JAX

Choose `cpu-no-jax` or `cuda-amd64-no-jax` to train and compile from the same
script using ordinary `python` and `fq.compile(..., compiler="qsteed")`.
These variants include development tools, plotting, example dependencies and
Quafu. They omit JAX/JAXlib and the broader `interop-all` extra: its PennyLane
version also requires NumPy 2, conflicting with QSteed's NumPy 1 requirement.
No separate compiler environment is needed.

```bash
docker compose -f compose.dev.yaml --profile gpu-no-jax build dev-gpu-no-jax
docker compose -f compose.dev.yaml --profile gpu-no-jax run --rm dev-gpu-no-jax \
  python your_training_and_compilation_script.py
```

For CPU, use profile `cpu-no-jax` and service `dev-no-jax`. To verify GPU
execution and QSteed compilation together after publishing:

```bash
docker run --rm --gpus all ghcr.io/flagos-ai/flagquantum-dev:cuda-amd64-no-jax \
  python /opt/FlagQuantum/docker/dev/smoke.py --gpu --qsteed
```

## Use QSteed with JAX-enabled variants

QSteed and pyquafu currently require NumPy < 2, while JAX 0.10 requires NumPy >= 2.
They cannot share one Python environment. The image isolates the compiler and
provides an explicit command:

```bash
flagquantum-qsteed-python your_compilation_script.py
```

Inside that script, use the usual `import flagquantum as fq` and
`fq.compile(circuit, compiler="qsteed", target=...)`. The main `python` command
runs the training/JAX environment and does not discover the isolated plugin.
Transfer trained parameter values or supported circuit artifacts explicitly
between scripts. The compiler environment uses CPU PyTorch; GPU training runs
in the main environment. No dependency constraints are overridden.

## Run the GPU image

After the publishing workflow succeeds on `main`:

```bash
docker run --rm -it --gpus all --shm-size=8g \
  ghcr.io/flagos-ai/flagquantum-dev:cuda-amd64
```

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. The
image provides CUDA user-space libraries; it does not install the host driver.
See the [JAX installation requirements](https://docs.jax.dev/en/latest/installation.html)
and [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
JAX memory preallocation is disabled so PyTorch and JAX can share the device.

For an immutable revision, use the workflow's `cuda-amd64-sha-<commit>` tag or
image digest. Package visibility is managed separately in GHCR settings; a
private package still requires authentication.

## Build with a live checkout

```bash
docker compose -f compose.dev.yaml --profile gpu build dev-gpu
docker compose -f compose.dev.yaml --profile gpu run --rm dev-gpu
```

Use `--profile cpu` and `dev` on a CPU machine. The two services use different
local image tags, so a CPU build cannot replace the GPU image. Both mount the
current checkout at `/workspace`.

## Verify the installed stack

Every image build runs `pip check` in each installed environment. No-JAX builds
also assert that neither `jax` nor `jaxlib` is installed and exercise QSteed. The main environment
tests PyTorch autograd, JAX JIT and FlagQuantum execution; the isolated compiler
tests PyTorch autograd, FlagQuantum execution and QSteed topology compilation.
On an NVIDIA host, also run:

```bash
docker run --rm --gpus all ghcr.io/flagos-ai/flagquantum-dev:cuda-amd64 \
  python /opt/FlagQuantum/docker/dev/smoke.py --gpu
```

This command fails if PyTorch or JAX cannot execute on the GPU. Build-time CPU
checks do not certify GPU correctness or distributed performance. Quantum cloud
submission still requires separately configured provider access; installing all
these dependencies does not grant access to a quantum device.

The workflow attaches image provenance and an SBOM. These are development
images, not frozen research evidence images; they cannot certify benchmark
claims. Rebuild when dependencies change. For distributed runs, use the same
image digest on every node and synchronize any mounted source checkout.

Maintainers can also publish from an authenticated build host with
`tools/containers/publish_dev_image.sh cpu` or `tools/containers/publish_dev_image.sh cuda`. Append `-no-jax` to the variant
argument to publish its single-environment counterpart.
Keep registry tokens outside source files and Docker build arguments.
