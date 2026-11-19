# Native paper quantum-walk workflows

These examples reproduce the source quantum programs and compilation workflows
using FlagQuantum. They do not execute foreign SDKs or reproduce private compiler
passes or timing measurements. Each source directory contains a native Notebook,
a CLI, an independent Circuit comparison and a precise source contract.

Shared responsibilities:

- `reference_programs.py`: native arithmetic and exact commented-loop qfuncs.
- `reference_circuit.py`: independent seven-bit permutation Circuit for correctness;
  its dense matrix is not a lowering or resource baseline.
- `reference_rotation_basis.py`: bounded CX/rotation lowering for the fixed source,
  with explicit phase correction on verified clean workspace.
- `reference_workflows.py`: compile, verify, count actual gates and report local
  durations. It composes public FlagQuantum APIs and the existing exact-rule
  extension in `classiq_discrete_quantum_walk/overview/exact_basis.py`.

The family owns example composition and source interpretation. It must not own
compiler defaults, backend dispatch, new public APIs or foreign runtime adapters.
Dependencies are Python, PyTorch and public FlagQuantum interfaces. Fix a reusable
compiler defect in the compiler domain; do not hide it behind an example rewrite.

The two quantum variants are deliberately distinct. Qiskit/Tket sources apply
increment for coin one and inverse increment for coin zero. The Catalyst source
flips the low bit inside every cascade iteration and differs on 32/64 inputs.
All source controls and loop effects are preserved and tested.

Start with:

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.qiskit_discrete_quantum_walk.overview.main
python -m pytest tests/integration/test_synthesis_walk_reference_notebooks.py -q
```

Expected initial named results are `(x=1, coin=1)` and `(x=63, coin=0)`, each with
probability one half. The Catalyst source instead gives `(x=1, coin=1)` and
`(x=3, coin=0)`. Change the native qfunc, compare the complete complex state with
the independent Circuit, then rerun the source Notebook. Tests cover every input
column, complex coherent input, cleanup, resource budgets and actual gate costs.
