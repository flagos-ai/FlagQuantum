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

The integer preparation policy is FlagQuantum-owned and is not presented as an
implementation of Kaiwu `PrecisionReducer`. Before using it for real-machine
submission, run version-pinned conformance tests against the installed Kaiwu
SDK and record the resulting coefficient and energy-order evidence.

For a ten-minute local check, run:

```bash
pytest -q tests/team/ecosystem/test_kaiwu_matrix_boundary.py
```
