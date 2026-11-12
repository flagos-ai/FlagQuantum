# Experimental Torch-FL FP64 statevector training

This development experiment classifies execution as `single_device_fast_path`.
It is not a release certification, a distributed result, or a claim about every
circuit family. Both platforms use the same FlagQuantum program on `flagos:0`,
complex128 states and matrices, float64 parameters and SGD, and exact dense
statevectors. No AMP, TF32, shots, noise, truncation, or backend fallback is used.

## What changes

The RY kernel writes both real components directly and fuses inverse rotation
with the first-order angle-gradient reduction. The CX kernel computes exact
source addresses for a sequence rather than materializing a full-state lookup
table. Higher derivatives retain differentiable operations. Large-state matrix
contractions, analytic Hermitian expectation gradients and the exact zero-state
product prefix are included in the same implementation. Arbitrary supplied gate
matrices retain the general matrix path. Routing thresholds and launch settings
are identical on both vendors. New kernels are cataloged as experimental.

## Recorded measurements

Measurements were collected on 2026-11-12 against base
`c2ff8bda879816fc2fa48ce112bfff39b5218764` plus the archived
[evidence patch](evidence/fused-ry-cx-statevector.patch.gz), not against a merge of
this PR. All 1,369 source files matched across the two platforms; aggregate
source SHA-256: `de83cc2b9a295dcc87226fe9d00578200c08d2629ec429d7bde9eb4d6fc09f84`.
The PR integrates onto newer main, formats code, names kernel parameters,
adds context typing and validates the external autograd result type. The
[PR source manifest](evidence/pr-source-manifest.json) distinguishes that code
from the [measured patch manifest](evidence/measured-patch-manifest.json).
Performance must be remeasured on the integrated commit before promotion.

| Qubits | Hygon training ms | A800 training ms | A800 time / Hygon time |
| --- | ---: | ---: | ---: |
| 12 | 57.193263 | 89.200820 | 156.0% |
| 16 | 75.289670 | 114.035490 | 151.5% |
| 20 | 111.785950 | 152.339407 | 136.3% |
| 22 | 119.082050 | 159.062973 | 133.6% |
| 24 | 191.257984 | 196.932112 | 103.0% |

The actual Hygon device reports **BW1101**; the presentation uses the requested
BW1100 family label. NVIDIA reports A800 SXM4 80GB. Hygon: EPYC 9K84,
Python 3.10, Torch-FL 2.10 + DTK 26.04rc4/HIP 6.3. NVIDIA: Xeon 8358,
Python 3.12, Torch-FL 2.10 + CUDA 13.0. Both use two CPU threads; the A800
container has a four-CPU quota. Different host CPUs and vendor stacks limit
attribution to the chip. The 24-qubit result is close to parity; it does not
establish stable universal superiority. New fused kernels activate at 22
qubits, so smaller sizes are not evidence of a gain from these kernels.

Circuit: four layers of RY on every wire followed by an adjacent CX ladder,
batch one, expectation Z(0), SGD learning rate 0.01. Initial angle i is
`0.23*sin(0.73*(i+1)) + 0.07*cos(0.31*(i+1))`. The gradient interface is
PyTorch autograd. All parameters are differentiated; this local observable and
shallow circuit have a limited causal cone and many exact zero gradients.

Each metric uses three rounds. Qubits 12/16/20/22 use three warmups and seven
samples per round; 24 uses two warmups and five samples. Report the median of
round medians, without dropping samples. Synchronized wall time includes public
API planning, circuit construction and execution; training includes forward,
backward and SGD, with zero_grad outside timing. Warm JIT is excluded. Selected
kernels run directly on flagos tensors; remaining Aten operations use explicit
vendor boxing configuration. This does not prove default FlagGems complex128
support. All numerical checks use an independent CPU complex128 reference with
1e-10 tolerance. All recorded runs completed.

[Measurement records](evidence/measurements.json) retain every timing sample,
round median, allocator peak, correctness result and source identity. Only host
identifiers, install/config paths and the full source file listing are omitted;
each extracted record names its original SHA-256. Raw private files are retained
locally. No SSH hosts, credentials or deployment scripts are required by the PR.

## Reproduce the recorded experiment

Prepare the recorded base and apply the archived patch with
`gzip -dc evidence/fused-ry-cx-statevector.patch.gz | git -C /path/to/recorded-source apply -`.
Provision an isolated vendor-compatible Torch-FL/Triton runtime and use the
archived vendor-specific all-boxing configuration in `evidence/`. The latter is an environment prerequisite,
not evidence that the default provider supports complex128. Import Torch-FL
before torch or pytest. Select one GPU using the vendor visibility variable.
Run the same commands on each vendor:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export FLAGOS_BACKEND_CONFIG="$PWD/evidence/hygon-boxing.conf"  # a800-boxing.conf on A800
python recorded_benchmark.py --source /path/to/recorded-source \
  --qubits 22 --depth 4 --rounds 3 --warmup 3 --samples 7 \
  --label reproduced --output q22.json
```

Run each process with an external timeout of 300 seconds. Failures propagate;
incomplete JSON must not be treated as successful evidence. Use qubits
12/16/20/22/24, with `--warmup 2 --samples 5` for 24. The archived script is
verbatim, including recorded revision metadata: do not use it to label a newer
source tree as the recorded revision.

## Existing public training workflow

```python
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["FQ_SV_RUNTIME_MODE"] = "portable"
import torch_fl  # Register the selected vendor before importing torch.
import torch
import flagquantum as fq

angles = torch.nn.Parameter(torch.full((4, 22), 0.1,
                                      dtype=torch.float64, device="flagos:0"))
optimizer = torch.optim.SGD([angles], lr=0.01)
optimizer.zero_grad(set_to_none=True)
circuit = fq.Circuit(22)
for layer in range(4):
    for wire in range(22):
        circuit.ry(wire, angles[layer, wire])
    for wire in range(21):
        circuit.cx(wire, wire + 1)
result = fq.run(circuit, options=fq.ExecutionOptions(
    mode="statevector", device="flagos:0", precision="complex128",
    allow_backend_fallback=False), outputs=fq.expectation(fq.Z(0)))
loss = result.expectation().real.sum()
loss.backward()
optimizer.step()
assert angles.grad is not None
assert angles.grad.device.type == "flagos"
assert angles.grad.dtype == torch.float64
```

Expected result: one exact FP64 training update, with the loss and all parameter
gradients on flagos, without changing the program between vendors.

## Verification boundary

The measured source passed 60 targeted kernel/gradient/catalog tests on each
platform, GPU gradcheck/gradgradcheck and four consecutive SGD updates (maximum
parameter error 2.78e-17). Affected regressions had 403 passes plus a known native
CPU-extension baseline failure; selected public API integration had 13 passes.
These historical checks do not certify the PR head. See PR validation for the
integration reruns.

Other circuit families/devices, torch.func, graph capture/compile/export,
multiple streams, distributed execution and full framework CI are not certified.
No capability maturity is promoted. Kernel errors propagate. The full required
CI and fresh dual-vendor performance measurements remain merge/release gates.
