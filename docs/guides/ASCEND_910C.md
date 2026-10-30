# Huawei Ascend 910C through Torch-FL

FlagQuantum supports native PyTorch quantum workflows on Huawei Ascend 910C
through Torch-FL's `flagos` device. The supported source configuration includes
the numerical, communication and runtime fixes pinned below. The previously
published wheels alone do not provide this support.

This is measured hardware development support. It does not promote the recorded
workloads to production, performance, capacity or hardware certification. The
[acceptance record](../../artifacts/ascend_910c_20261030.json) identifies source
commits, binary digests, test reports and remaining boundaries. The older
[split-FP32 record](../../artifacts/ascend_910c_20261029.json) remains historical
evidence for its original scope.

## Validated functions

| Workflow | Measured scope |
| --- | --- |
| Native circuit simulation | Statevector, MPS, tensor network and density matrix; complex64 and complex128; probabilities, expectations, gradients and seeded samples |
| Gates and quantum diagnostics | Registered gate families, non-adjacent wires, entropy and gradient methods, checked against CPU references |
| Hybrid training | `fq.Module`, PyTorch autograd, classical layers, SGD and Adam; quantum parameters and gradients remain on the requested device |
| Noise and TEBD | Noise-channel comparisons and matrix-exponential evolution in both complex precisions |
| Native dynamics | Ising braid execution and instruments; Lindblad and Rydberg RK2, Crank–Nicolson and Krylov evolution, control gradients, optimizer trajectories, adjoint cotangents and sampled-presence replay |
| QEC neural predecoder | Device training, gradient/update parity and exact optimizer/RNG checkpoint restart; the global combinatorial decoder is an explicit CPU boundary |
| Sharded training | Statevector, MPS and tensor network; owner-sharded gradients and optimizer updates; CPU numerical references |
| Communication and restart | Pure FlagOS FlagCX collectives, P2P, asynchronous Work/Future completion, two-host training and shared-filesystem checkpoint restart |

The single-node training ladder uses 2, 4 and 8 logical NPUs. On these 910C
hosts, one physical card exposes two logical NPUs: the eight-device run uses
four physical cards. The two-host tests use one logical NPU on each host.
These small correctness workloads do not establish a maximum qubit count,
capacity expansion or speedup.

## Fixed software configuration

| Component | Qualified configuration |
| --- | --- |
| Hardware | Huawei Ascend 910C, SoC `Ascend910_9382`, Linux aarch64 |
| Python / PyTorch | 3.11.15 / 2.10.0+cpu |
| CANN / host driver | 9.0.0 / 25.5.0 |
| FlagQuantum implementation | `dae8c8fdba33cd9cfff4d61847f6da697cba0215` |
| Torch-FL implementation | `3aebd0c888789f35bb281435ddd44d5f1ff8cc55` |
| FlagTree / FlagGems | 0.7.0+ascend3.5 / 5.4.0, installed from published wheels |
| FlagCX source | Official v0.14.1, `6fe36310fb5c9bd2396f531d7bf78c08fc792340`, with the checked-in Ascend/FlagOS patch |
| FlagCX built version | 0.14.2.dev0+dirty; the source revision and patch identify the build |

Tested Torch-FL wheel SHA-256:
`7abe32e668f4aeca69dfbcfd215b6c86cb7e33bf6221db0e76c423975a25d3ca`.
Tested FlagCX wheel SHA-256:
`35f635bd9b752ea3a50af42613d1c86bd69d1151a1d96472268f0623449052f3`.
Builds on a different toolchain may have different binary digests and require
fresh validation. Preserve the source pins and test the resulting artifacts.

Native complex storage, arithmetic, factorization, sampling and dynamics use
device operators. FP64/complex128 reductions use lossless int64 transport plus
device FP64 arithmetic because this HCCL configuration lacks native FP64
reduction. This path is blocking, uses bounded scratch storage, and broadcasts
once per rank per tensor. Its communication cost is explicit; it is not a claim
of native HCCL FP64 reduction or optimized performance.

## Install in a dedicated environment

Use a provisioned CANN 9.0.0 environment with the selected devices and matching
driver libraries mounted. Use one dedicated acceptance container per host;
overlapping test containers that expose the same NPUs caused driver resource
conflicts during validation. Retain other users' containers and environments.

The real `torch_npu` extension and Torch-FL both claim PyTorch's PrivateUse1
backend. Use an environment without the real extension and import `torch_fl`
before `torch`. Torch-FL may install an internal compatibility shim named
`torch_npu`; that shim is distinct from loading the vendor extension.

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps \
  --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple \
  'flagtree==0.7.0+ascend3.5'
python -m pip install --no-deps \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  'flag_gems==5.4.0'
python -m pip install 'PyYAML==6.0.1' 'SQLAlchemy==2.0.55' \
  'numpy==2.4.4' 'pybind11==3.1.0' packaging setuptools wheel \
  'setuptools_scm==9.2.2'
```

Build Torch-FL from the pinned source with a C++ compiler and CMake available:

```bash
git clone https://github.com/FlagQuantum/Torch-FL.git torch-fl-910c
git -C torch-fl-910c checkout 3aebd0c888789f35bb281435ddd44d5f1ff8cc55
cd torch-fl-910c
FLAGOS_ACCELERATOR=ascend FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --no-deps dist/torch_fl-*.whl
cd ..
```

Build the compatible FlagCX dependency from a fresh checkout. The published
Ascend wheels inspected during acceptance required the real `torch_npu`
extension and could not serve the pure Torch-FL path. The patch fixes ACL event
flags, paired send/receive ordering and Work/Future completion for Ascend/FlagOS.
Other vendor paths retain their existing behavior.

From the FlagQuantum checkout containing this guide:

```bash
git clone https://github.com/flagos-ai/FlagCX.git /tmp/flagcx-910c
git -C /tmp/flagcx-910c checkout 6fe36310fb5c9bd2396f531d7bf78c08fc792340
mkdir -p /tmp/flagcx-json-include/nlohmann
curl --fail --location --max-time 60 \
  https://raw.githubusercontent.com/nlohmann/json/a0e9fb1e638cfbb5b8b556b7c51eaa81977bad48/single_include/nlohmann/json.hpp \
  -o /tmp/flagcx-json-include/nlohmann/json.hpp
export DEVICE_HOME=/usr/local/Ascend/ascend-toolkit/latest
python docker/ascend/build_flagcx.py \
  --source /tmp/flagcx-910c --json-include /tmp/flagcx-json-include \
  --wheel-dir /tmp/flagcx-910c-wheels
python -m pip install --no-deps /tmp/flagcx-910c-wheels/flagcx-*.whl
python -m pip install --no-build-isolation .
```

The build helper verifies the FlagCX revision, checks the single-header digest,
applies [the patch](../../docker/ascend/patches/flagcx-v0.14.1-flagos.patch) and
limits compilation to four workers. It is intended for a fresh source checkout;
reapplying a patch to an already modified tree fails explicitly.

## Run and verify

The normal circuit API uses native complex tensors:

```python
import torch_fl
import torch
import flagquantum as fq

torch.set_float32_matmul_precision("highest")
theta = torch.tensor(0.25, dtype=torch.float64, device="flagos:0", requires_grad=True)
optimizer = torch.optim.Adam([theta], lr=0.02)
for _ in range(3):
    optimizer.zero_grad()
    circuit = fq.Circuit(2, device="flagos:0", dtype=torch.complex128)
    circuit.ry(0, theta).cx(0, 1)
    loss = circuit.expectation_z()[0, 0]
    loss.backward()
    assert theta.grad.device.type == "flagos"
    optimizer.step()
```

Run the checked-in accelerator tests in a fresh process:

```bash
python -m pip install pytest
FLAGQUANTUM_TEST_ASCEND_910C=1 FLAGOS_LOG=fallback \
  python -m pytest tests/test_ascend_910c_training.py \
  tests/integration/test_ascend_circuit_workflows.py \
  tests/integration/test_ascend_native_dynamics.py -q

FLAGOS_DIST_STAGED_GLOO=0 FLAGCX_TORCH_BACKEND=flagos \
  python tools/validate_flagos_statevector_training.py \
  --world-sizes 2,4,8 --steps 3 --output /tmp/ascend-training.json
```

For the tested two-host CANN network, set both `FLAGCX_SOCKET_IFNAME` and
`HCCL_SOCKET_IFNAME` to the reachable inter-host interface (`business` in the
acceptance environment). Set `HCCL_HOST_SOCKET_PORT_RANGE=auto`,
`HCCL_NPU_SOCKET_PORT_RANGE=auto` and `HCCL_CONNECT_TIMEOUT=120`. Fixed default
HCCL ports collided with other users' jobs; acceptance used automatic port
selection instead of changing those jobs.

The hardware tests compare bounded CPU references explicitly. Checkpoint I/O,
serialization, metadata and the global QEC decoder also have explicit host
boundaries. These operations are distinct from a hidden numerical CPU fallback.
The provider suite contains an intentional integer-matmul CPU-fallback logging
test; zero-fallback assertions apply to the recorded quantum workflows, not to
every operator in PyTorch.

Optional JAX accelerators, CUDA-specific external simulators and quantum-cloud
providers retain their own backend requirements. This configuration does not
translate those third-party runtimes to Ascend. Compilation, circuit drawing and
other host-side utilities remain host operations.
