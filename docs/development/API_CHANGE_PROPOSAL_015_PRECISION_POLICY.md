# API Change Proposal 015: Remove Unimplemented Precision Controls

## Status

**Approved for the first public alpha.** On 2026-09-06, the API owner explicitly
authorized narrowing `flagquantum.training.PrecisionPolicy` before public release,
without compatibility parameters or checkpoint migration layers.

## Problem

`PrecisionPolicy` exposed `mode` and `accumulator_dtype`, but Runtime, statevector,
MPS, tensor-network execution, and training reductions did not consume them. They
only affected validation and checkpoint serialization, misleading users into
believing mixed-precision accumulation was active.

## Decision

Retain only effective `PrecisionPolicy` fields:

```text
complex_dtype, parameter_dtype, allow_parameter_downcast, atol, rtol
```

Accept only consistent `complex64/float32` and `complex128/float64` pairs. Remove
`mode` and `accumulator_dtype`, reject old keywords, and add no deprecation
wrappers. Device/kernel accumulation precision remains expressed through
TargetCapabilities and execution evidence, separately from user controls.

## Compatibility

This is direct consolidation before the first public alpha. Source using removed
keywords and internal development checkpoints is no longer compatible. The
repository has no released users or promised migration window. Future
mixed-precision controls require a new API proposal, at least one implementing
backend, verifiable result evidence, and clear cross-backend semantics.

## Acceptance

- Check the exact `PrecisionPolicy` field set.
- Reject inconsistent complex/real precision pairs.
- Checkpoint save/restore preserves effective precision policy.
- Module, CPU vertical-path, and public API contract tests pass.
