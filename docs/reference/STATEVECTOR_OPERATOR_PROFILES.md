# Statevector Operator Profiles

FlagQuantum operator profiles are versioned, machine-readable workload
contracts between quantum runtimes and platform providers. They describe the
minimum PyTorch operator surface required by a representation; they do not
contain vendor routing policy.

## `statevector_local_p0`

The packaged profile is
[`flagquantum/runtime/profiles/statevector_local_p0.json`](../../flagquantum/runtime/profiles/statevector_local_p0.json).
It covers the minimum single-device differentiable statevector path used by
fixed and parameterized one- and two-qubit gates:

- complex state allocation and dtype conversion;
- reshape, permute, expand, diagonal, unsqueeze, unbind, and flip;
- complex `bmm`, elementwise multiplication, stack, and concatenation;
- complex construction, real-angle trigonometric functions, and portable
  real/imaginary conjugation primitives;
- absolute value and reduction for observables and gradient objectives.

Every requirement independently declares supported dtypes and whether forward,
backward, and deterministic behavior are required. P0 contains `complex64` and
`complex128`; execution selects only the requested dtype slice.

## `split_real_imag_statevector_p0`

The split profile is a separate forward-only FP32 contract for the explicit
experimental executor. It covers allocation, reshape/permute/transpose,
matrix multiplication, real add/subtract/multiply, stack, and real
trigonometric gate generation. It does not inherit the complex profile and
does not claim gradients. See
[`SPLIT_REAL_IMAG_STATEVECTOR_P0.md`](SPLIT_REAL_IMAG_STATEVECTOR_P0.md).

## `split_real_imag_statevector_p1`

P1 is a separate FP32 profile for bounded Pauli expectation values and
explicit parameter-shift gradients. It adds backward probes for the real
operator surface used by the training primitive, but does not claim native
autograd. Its API, supported gates, observable boundary, and fail-closed
parameter rules are documented in
[`SPLIT_REAL_IMAG_STATEVECTOR_P1.md`](SPLIT_REAL_IMAG_STATEVECTOR_P1.md).

## Execution behavior

For CPU and native CUDA, existing behavior is unchanged. For local statevector
execution on `flagos`, FlagQuantum performs the following before allocating the
user workload state:

```text
load packaged profile
  → compute canonical profile hash
  → run tiny operator probes on flagos:0
  → compare forward and gradients with CPU
  → verify logical flagos residency
  → build CapabilityEvidence records
  → fail-closed preflight
  → run a cached differentiable circuit against CPU complex128
  → enforce the accuracy requirement and precision plan
  → execute the user circuit
```

Evidence is bound to the exact profile hash, device type, dtype, provider, and
probe result. Successful evidence is cached per process for the same profile,
device, dtype, and provider. Failed probes retain their diagnostic exception in
the preflight blocker.

The accepted reports are attached to `ExecutionPlan.routing_plan` as
`operator_preflight`, `accuracy_requirement`, `precision_plan`, and
`numerical_validation`, with content hashes for both numerical contracts. The
end-to-end workload records norm drift, state error and infidelity, expectation
error, gradient relative error and cosine similarity, and bitwise repeatability.
It is deliberately a three-qubit certification workload rather than a shadow
copy of the user's state, so startup evidence remains bounded and is cached.

Callers may supply `accuracy_requirement=` and `precision_plan=` as contract
objects or their machine-readable mappings. FlagQuantum rejects a precision
plan that the current native statevector implementation cannot actually honor;
it never labels silent dtype demotion as compliant. Explicit contract
enforcement is currently scoped to `flagos` local statevector execution.

## Validation commands

Run the profile contract and CPU reference probes:

```bash
pytest tests/unit/test_statevector_operator_profile.py -v
```

Run the full Torch-FL CUDA-backed `flagos:0` integration:

```bash
python tools/validate_flagos_cuda_reference.py \
  --dtypes complex64,complex128 --depths 8,32,128
```

or:

```bash
FLAGQUANTUM_TEST_FLAGOS_CUDA=1 pytest tests/test_flagos_cuda_reference.py -v
```

Run the forward-only split FP32 integration separately:

```bash
python tools/validate_split_real_imag_flagos.py --device flagos:0
```

Run the split FP32 observable and parameter-shift integration separately:

```bash
python tools/validate_split_real_imag_training_flagos.py --device flagos:0
```

The 2026-08-24 A800 reference run passed all 21 profile requirements, both
complex dtypes, and depths 8/32/128. Its machine-readable environment,
isolation record, runtime evidence, and numerical metrics are stored in
[`artifacts/flagos_cuda_reference_a800_20260824.json`](../../artifacts/flagos_cuda_reference_a800_20260824.json).

Validate a checked-in or newly generated result without accelerator access:

```bash
python tools/validate_flagos_reference_evidence.py \
  artifacts/flagos_cuda_reference_a800_20260824.json
```

The gate recomputes the packaged profile hash, checks exact dtype/depth
coverage and numerical bounds, verifies the environment-lock digest and every
runtime/build identity field, requires a clean auditable source revision, and
rejects hardware, production, or scalability promotion fields.

## Claim boundary

P0 proves that the logical `flagos` path can execute the minimum operator set
and a small differentiable circuit within a declared numerical envelope. The
CUDA-backed validator additionally scans requested dtype/depth combinations.
It does not by itself prove that Torch-FL avoided all host fallback, that
application-scale circuits converge, or that a Hygon device meets performance
and complex128 requirements. Production promotion still requires Torch-FL
route/fallback evidence, profiler residency, workload-specific convergence,
and real-device CI.
