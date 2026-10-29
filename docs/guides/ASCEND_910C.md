# Huawei Ascend 910C

FlagQuantum provides experimental single-device support for Huawei Ascend 910C
through Torch-FL's logical `flagos:0` device. This path uses two FP32 tensors
for a statevector's real and imaginary amplitudes. It does not require native
complex accelerator operators.

## Supported workflow

The measured scope covers the built-in split-statevector gate set, batch size
one, Pauli/Hamiltonian expectations, direct named RX/RY/RZ/RXX/RYY/RZZ parameters,
explicit parameter-shift gradients, and explicit SGD updates. State, gradients,
and parameter updates remain on `flagos:0`. The example transfers scalar
diagnostics to the host explicitly; its analytic reference is diagnostic code.

The ordinary complex statevector runtime, `Module` autograd, MPS, tensor
networks, distributed execution, custom states/matrices, parameter expressions,
and high-precision certification on Ascend are outside this scope. The hardware
evidence is single-device correctness evidence, not a production, performance,
capacity-limit, or hardware-certification claim.

## Tested software

| Component | Tested version |
| --- | --- |
| Device | Ascend 910C, reported SoC `Ascend910_9382` |
| Host architecture | Linux aarch64 |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cpu |
| CANN | 9.0.0 |
| Host driver / npu-smi | 25.5.0 |
| Torch-FL | `704e2822d3bdbbebfe2b78a62b434d5972babc67` |
| FlagTree | 0.7.0+ascend3.5 |
| FlagGems | 5.4.0 |
| FlagCX package | 0.14.0+cann9.0.0; collectives not tested |
| PyYAML | 6.0.1 |

The Torch-FL commit is a prerequisite. The previously published
`torch_fl-2.10.0+cann9.0.0` wheel with SHA-256
`a1ed42fa240a62333b372370199b31e335136da38146f7cae0b573d410f73a72`
does not contain these fixes. Version labels alone do not distinguish it from
the rebuilt wheel. Preserve the source revision and wheel digest.
The dependency fixes are proposed in [Torch-FL #683](https://github.com/flagos-ai/Torch-FL/pull/683).

## Install

Use a dedicated Python 3.11 environment on a provisioned CANN 9.0.0 host or
container with one selected NPU and the matching driver libraries mounted.
The official Torch-FL [Ascend guide](https://github.com/flagos-ai/Torch-FL/blob/main/docs/vendors/ascend/installation.md)
describes device and driver provisioning. Retain other users' environments.

Torch-FL and the real `torch_npu` extension both register PyTorch's PrivateUse1
backend. They must not be imported together. Use a clean environment without
`torch_npu` for this workflow, and import `torch_fl` before `torch`.

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps \
  --index-url https://resource.flagos.net/repository/flagos-pypi-ascend/simple \
  'flagtree==0.7.0+ascend3.5' 'flagcx==0.14.0+cann9.0.0'
python -m pip install --no-deps \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  'flag_gems==5.4.0'
python -m pip install 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' \
  'numpy==2.4.4' pybind11 packaging

git clone https://github.com/FlagQuantum/Torch-FL.git torch-fl-910c
cd torch-fl-910c
git checkout 704e2822d3bdbbebfe2b78a62b434d5972babc67
FLAGOS_ACCELERATOR=ascend FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_JOBS=6 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --no-deps dist/torch_fl-*.whl
```

In the FlagQuantum source checkout containing this guide:

```bash
python -m pip install --no-build-isolation .
python examples/ascend_910c_training.py --device flagos:0
```

Install the build frontend requirements (`setuptools`, `wheel`, a C++ compiler
and CMake) before building. The tested aarch64 FlagQuantum wheel includes its
native CPU extension, so the CPU complex128 reference is available in the same
environment. No Torch-FL dependency is added to the core FlagQuantum package.

The example uses PyTorch's `highest` FP32 matmul policy. The Torch-FL fix makes
both forward and fused backward honor that policy; it also adds a native real
`sgn` route for `abs` backward. No matrix-routing overrides, lowered numerical
thresholds, or skipped capability probes are required.

## Run and verify

The documented example uses public experimental interfaces:

```python
import torch_fl
import torch

import flagquantum as fq
from flagquantum.algorithms import pauli_term

torch.set_float32_matmul_precision("highest")
theta = torch.tensor(0.25, device="flagos:0")
circuit = fq.Circuit(2).ry(0, fq.Parameter("theta")).cx(0, 1)
observable = pauli_term(1.0, "Z", (0,))
for _ in range(64):
    result = fq.experimental.numerics.parameter_shift_split_real_imag_gradient(
        circuit, observable, parameter_bindings={"theta": theta}, device="flagos:0"
    )
    theta = theta - 0.2 * result.gradient[0]
```

The full example additionally compares each gradient with the analytic
derivative `-sin(theta)` and reports state/gradient devices. The measured run
started at expectation `0.9689124217` and ended at `-1.0`, with maximum analytic
gradient error `1.40e-7`.

Run the hardware tests in a fresh process, so another plugin has not already
claimed PrivateUse1:

```bash
FLAGQUANTUM_TEST_ASCEND_910C=1 FLAGOS_LOG=fallback \
  python -m pytest tests/test_ascend_910c_training.py \
  tests/unit/test_split_real_imag_statevector.py -q
```

The gate tests cover both two-qubit registers and twelve-qubit registers with
reversed, non-adjacent gate wires. The executor groups consecutive untouched
wires into one axis: one- and two-wire gates need at most five tensor axes,
avoiding the high-rank noncontiguous-copy failure observed at twelve qubits.

The checked-in [910C evidence](../../artifacts/ascend_910c_20261029.json) also
records depth 8/32/128 forward and gradient scans, two gradient seeds, and
3/8/12/16/20-qubit forward comparisons with CPU complex128. Twenty qubits is a
tested point, not a claimed maximum. The provider's internal routes are not
comprehensively certified, and the support statement remains experimental.
