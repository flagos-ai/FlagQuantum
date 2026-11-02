# Complete-circuit precision frontier

Issue #901 compares existing numerical stages on bounded complete circuits.
The producer is `python -m benchmarks.precision_frontier`. It reuses current
split-real/imaginary and Double-Single executors, matrix encoders, operator
preflights and benchmark source/result contracts. No precision abstraction,
public API, runtime default, vendor dependency or maturity promotion is added.

## Inventory and supported domain

The numerical foundation and P0--P5 contracts remain authoritative:

- P0/P1: split FP32 states, native FP32 gates/reductions and parameter shift.
- P2: FP32 state/gates with compensated reductions; not a full DS simulator.
- P3: four FP32 state words, explicit CPU complex128 gate generation and
  transfer. This host encoding boundary is reported, not hidden.
- P4: four FP32 state words with bounded device-generated trigonometry and
  supported fixed, RX/RY/RZ/RXX/RYY/RZZ gates. No native autograd is assumed.
- P5: existing autograd-delivery and SGD-master boundaries, probed separately;
  they do not turn this parameter-shift experiment into training certification.

Every campaign first exercises the existing small conformance runners for
arithmetic, P0, P1, P2, P3, P4 and both P5 boundaries. Reports retain successful,
inaccurate, unsupported and failed probes. CPU-only P5 autograd is explicitly
unsupported on accelerators. An unknown platform is a stage-support gap, not
permission to execute it on CPU. A probe never authorizes silent fallback.

## Reproduce

NumPy and Torch from the existing optional numerical comparison environment
are sufficient. No framework or vendor package is added to core dependencies.
Use new output directories outside the checkout:

```bash
python -m benchmarks.precision_frontier --smoke --output /tmp/precision-smoke
python -m benchmarks.precision_frontier --staged-controls --output /tmp/precision-cpu
python -m benchmarks.precision_frontier --device cuda:0 --staged-controls --output /tmp/precision-cuda
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m benchmarks.precision_frontier --device mps --output /tmp/precision-mps
python -m benchmarks.precision_frontier --report --output /tmp/precision-cpu
```

The default campaign compares FP32/P1, full DS/P4 and native complex128/FP64.
`--staged-controls` also includes P2 and host-encoded P3. NVIDIA and non-NVIDIA
Apple MPS are explicit optional campaigns. MPS requires a Metal-capable Torch
build and actual access to the device. Unsupported precision or platform stages
remain in the output. This is not a declaration of MPS platform support.
The existing Local GPU Gate has a `precision_frontier=true` manual input that
runs the frozen NVIDIA campaign on its A800 lane, with the existing source
preparation, exclusive accelerator concurrency, watchdog and artifact upload.
Select one manual campaign input at a time.

## Frozen circuits and independent references

`benchmarks/manifests/precision_frontier_v1.json` freezes shapes (2,2), (2,8),
(4,8), seeds 11/29, five repeated samples, one warmup, accuracy targets and a
240-second per-case deadline. Three primary modes give 36 cases; including the
two staged controls gives 60. Smoke uses one shape/family/seed and two samples,
and is not full acceptance evidence.

Two families contain initial Hadamard interference, one named RY parameter,
noncommuting RX/RZ/RXX layers and, for the general family, entangling CX gates.
The echo family inserts each noncommuting block and its inverse to expose
accumulated cancellation and norm error. Its objective retains ordered
`1e8*Z0 + Z0 - 1e8*Z0 + 0.5*X1 + 0.25*Z0Z1` terms. The reference coalesces
identical Pauli terms with `math.fsum` before evaluating the physical operator;
it does not reproduce the candidate reduction's cancellation error.

All routes consume the identical FP32-representable initial angle and dyadic
fixed angles, and exactly representable coefficients. The intended FP64 angle
is separately retained. Its input conversion error and resulting reference
state difference are not mixed with simulation error. All candidate gradients
use the supported occurrence-wise parameter-shift route for the same single
parameter. FP32's rounded shifts and P4's two-word shifts remain their actual
implementations; gradient error is measured, not corrected using the oracle.

The independent reference uses explicit NumPy complex128 matrices, full
big-endian basis embedding and an analytic initial-RY tangent. Tests also check
the tangent and energy gradient by centered finite differences. It never calls
the candidate simulator. State fidelity, maximum/L2 error, norm drift, native
Z0/X1/Z0Z1/objective term reductions and gradient error are all retained.
Existing route matrix encoders are separately compared against independent
local matrices. Coefficient and initial-angle conversion errors are separate.
Diagnostic DS reconstruction transfers words to CPU before combining them;
it is outside candidate timing and is not host execution fallback.

## Timing, memory and output identity

Each case has a fresh worker. Repeated samples synchronize the requested device
before and after complete circuit construction, gate generation, state
execution, native observables and gradient computation. The named statistic
is the median; every absolute sample remains in JSON. First call includes
operator preflight and is separate, as are warmup and process wall. No GEMM
ratio is used as a simulator speedup. Renormalization is disabled for all routes.
Torch intra/inter-op, OMP and BLAS settings, device name/UUID where exposed,
CUDA runtime/driver, Torch build/compiler information, TF32 flags/overrides,
JIT/compile policy and FMA observability limits are recorded. TF32 is disabled;
eager kernels may still use their documented device/build intrinsics.

Worker peak RSS is captured at timing end and includes imports, preflight,
warmup and output-hash transfers. CUDA allocator peak covers repeated samples.
Both differ from exact state payload bytes, which are reported separately.
MPS exposes no allocator peak or physical UUID through this Torch interface;
those fields have explicit unavailability reasons, never fabricated values.
The accuracy/time/peak-memory frontier uses declared tolerances and measured
CUDA peak or CPU RSS. Unknown accelerator peaks cannot win a Pareto comparison.
All slower, inaccurate, unsupported, timeout and failed cases remain visible.

Every retained repeated sample hashes actual state, observable and gradient
word dtype, shape and bytes. Repeated bit identity is only a within-route
statement about those hashes. Equal scalar errors, equal fidelity, or equal
rounded printed outputs never establish output bit identity.

## Exponent range and non-promotion

DS retains binary32 exponent range, not binary64 range. Dekker splitting needs
finite values sufficiently below `float32.max / 4097`; intermediate products,
subnormals and device flush-to-zero behavior also constrain useful precision.
Roughly 48 significant bits is not a universal error bound. P4 direct angles
must satisfy `|theta| <= 1024`. Arbitrary expressions, unsupported gates,
complex accelerator tensors, distributed execution and unbounded gradients
are outside the staged contracts. Existing host/device boundaries are retained.
These measurements do not certify a provider, extend a stage's supported
domain, or promote experimental capabilities to production readiness.
