# Explicit Qualtran reference comparisons

This directory owns the original nine optional upstream parity modules and their
shared Clifford+T counter. It does not provide product adapters or benchmark
implementations. Tests may use FlagQuantum public/domain APIs and independent
upstream references; product modules must not import this test package.

Install the established Qualtran0.7.0/Cirq1.4.0 environment described in the
[progress record](../../../docs/roadmap/QUALTRAN_API_PROGRESS.md), then run:

```sh
python -m pytest -q tests/reference/qualtran
```

For a small change, run one named reference file first. The suite remains an
explicit invocation, as it was under tools/qualtran_parity: tests/conftest.py
excludes it from default recursive collection while pytest permits direct
directory or file arguments. The same conftest sets early thread defaults before
reference modules import Torch/NumPy. Missing optional dependencies are not
scientific qualification. Timed comparisons and immutable original records live
in benchmarks/qualtran and artifacts/benchmarks/qualtran respectively.
