# Synthesis contract fixtures

These three original Classiq Library files are pinned by commit and Git blob in
[manifest.json](manifest.json). Their bytes are unchanged; retain the accompanying
[MIT license](LICENSE.txt), including the Classiq copyright, when redistributing.
The license was retrieved from the same pinned upstream commit.

The [SYN-01 proposal](../../../docs/api-changes/FQ-SYNTH-20261028.md) defines the
staged acceptance. The fixture integrity tests and handwritten CircuitIR boundary
fakes do **not** parse, synthesize, or execute these Qmod files. Native Qmod
execution coverage remains 0/216 until the implementation PRs pass their gates.
Tests use local files and never call Classiq services.

| File | Future acceptance |
| --- | --- |
| `X.qmod` | Allocate one output qubit and prepare one through the shared synth path (SYN-03). |
| `multiplication_2vars_example.qmod` | Original source produces a=4, b=5, res=20 (SYN-07). |
| `synthesis_tutorial.qmod` | Original three-bit x is uniform; y=x*x+1, with all eight complex amplitudes and clean workspace checked (SYN-07). |

Run `python -m pytest tests/unit/test_synth_contract.py -q` for the current
integrity and existing IR/runtime boundary checks. Later parser tests must read
these files directly, without stripping unsupported statements or rewriting them.
