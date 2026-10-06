# Kaiwu ecosystem boundary

This package owns pure-data interoperability between FlagQuantum and the
QBoson Kaiwu SDK. Its first contract is the symmetric Ising matrix evaluated as
`E(s) = -s.T @ matrix @ s + bias` for spins in `{-1, +1}`.

It must not own credentials, quotas, project selection, task submission,
polling, cancellation, retries, or remote result retrieval. Those operations
belong under `flagquantum.remote.kaiwu`. It must also not select a FlagQuantum
runtime or silently replace unavailable Kaiwu execution with a local solver.

The package may depend on FlagQuantum Core, Python's standard library, and
PyTorch. Imports of optional Kaiwu packages must remain lazy and localized to
the adapter that requires them.

Public entry points currently live in `flagquantum.ecosystem.kaiwu`:

- `canonicalize_ising_matrix` validates the matrix boundary.
- `encode_qubo_as_ising` and `decode_qubo_spins` provide an auxiliary-spin
  QUBO mapping with exhaustive energy-parity fixtures.
- `ising_energy` makes the Kaiwu sign and matrix convention testable.
- `prepare_integer_precision` provides explicit, auditable lossy scaling.
- `KaiwuSampler` implements the synchronous `solve(ising_matrix)` surface used
  by Kaiwu PyTorch Plugin while delegating task ownership to
  `flagquantum.remote.kaiwu`. It deduplicates identical matrices and enforces an
  explicit remote-call budget and timeout. Its `precision_reports` retain one
  report per distinct original matrix, including separate reports when multiple
  inputs quantize to one shared remote matrix. `precision_evidence` additionally
  binds each report to its original and submitted matrix digests, normalized
  coefficient range, original plugin object type and dtype, and exact
  symmetry-normalization and rounding rules. Transfer records retain the same
  original and submitted digests so final evidence can reconcile both sides of
  the CPU boundary. Its read-only
  `client` property exposes the already-bound Remote client solely so evidence
  builders can prove transport provenance; task ownership remains in
  `remote.kaiwu`.

The integer preparation policy is FlagQuantum-owned and is not presented as an
implementation of Kaiwu `PrecisionReducer`. Before using it for real-machine
submission, run version-pinned conformance tests against the installed Kaiwu
SDK and record the resulting coefficient and energy-order evidence.

`symmetry_tolerance` controls acceptance only. Any matrix accepted under a
nonzero tolerance is averaged with its transpose so the returned value is
exactly symmetric and can pass the stricter Remote receipt identity boundary.
Scalar tolerance, offset, and bias arguments must be finite real numbers;
booleans, strings, and complex values are rejected rather than coerced.
Normalization uses half-scaled operands so two finite extreme coefficients do
not overflow merely while being averaged. Conversion, energy evaluation, and
precision evidence still fail closed if their actual derived matrix, bias,
energy, dequantized value, or error becomes nonfinite.
The effective symmetric integer magnitude is capped at the largest consecutive
integer exactly representable by `float64` (`2^53`), so dequantized values and
error evidence cannot hide an `int64` saturation or an inexact large integer.

`KaiwuSampler` never enables integer scaling implicitly. Set its
`integer_target_range` only after choosing and recording a precision policy.
Sampler construction validates timeout, polling interval, project identity,
and integer-range value types before a matrix can reach the Remote layer;
booleans are not accepted as numeric configuration values.
Exhausting the remote-call budget raises before submission; there is no local
or classical fallback. A sampler instance serializes its complete synchronous
solve transaction, so concurrent callers cannot race cache registration or the
remote-call counter and submit the same matrix more than once. A budget slot is
reserved before entering the Remote client and is never released after an
exception, because the provider may already have observed an indeterminate
submission.

For a ten-minute local check, run:

```bash
pytest -q tests/team/ecosystem/test_kaiwu_matrix_boundary.py
```
