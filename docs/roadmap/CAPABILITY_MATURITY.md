# Capability maturity

`capability-maturity.toml` is the authoritative, machine-validated capability
matrix. Marketing text, benchmark summaries, and release notes must not assign
a stronger status than this matrix.

The levels are deliberately non-interchangeable:

| Level | Meaning | Permitted claim |
| --- | --- | --- |
| `experimental` | Unstable research implementation. | The implementation is available for evaluation. |
| `development_evidence` | Executed development or semantic evidence. | The constrained path was executed; no production or scalability claim. |
| `production_supported` | Supported path with integration, hardware, and operational evidence. | The documented workload and environment are supported in production. |
| `release_certified` | Audited and reproducible release evidence with a release gate. | The named release artifact is certified for its exact declared scope. |

Status is capability-specific. A stable public API does not promote an
experimental backend, and CPU semantic evidence does not promote a distributed
runtime. Promotion requires adding every evidence field required by the target
level and passing `python tools/check_capability_maturity.py`.

Schema v2 also makes public performance claims fail closed. Each claim must
identify a checked-in raw JSON artifact and its SHA-256 digest, match the code
version recorded by that artifact, inherit the capability maturity, derive its
displayed measurements and recorded environment from JSON selectors, and state
its exact scope and metadata boundary. The documentation generator emits the
same validated values into the README, capability catalog, and Known
Limitations; edits inside generated regions are rejected by the CI check.

The split real/imag P5 line currently exposes an experimental CPU-only
first-order PyTorch autograd bridge over P4 Double-Single execution. Its
returned loss and `.grad` are explicitly FP32 delivery boundaries. A separate
experimental CPU Double-Single SGD lane consumes explicit high/low gradients
and retains a high/low master parameter without using `.grad`. Accelerator
routes, richer optimizers, higher-order autograd, and distributed execution
remain gated future work.
