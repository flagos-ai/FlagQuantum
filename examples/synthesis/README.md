# Native high-level quantum examples

Write computation, quantum data and constraints in FlagQuantum Python. The compiler
constructs the circuit; examples expose named registers rather than physical wires.

- [Algorithms](algorithms/README.md)
- [Applications](applications/README.md)
- [Community](community/README.md)
- [Functions](functions/README.md)
- [Tutorials](tutorials/README.md)

Start with a complete native program:

```sh
python -m examples.synthesis.tutorials.composition.python_program.main
python -m examples.synthesis.tutorials.basics.qml.main
python -m examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.main
```

Every completed upstream counterpart has native `main.py`, a provenance record,
and an independent Circuit comparison. Comparisons check complete complex
amplitudes, absolute phase, register order and hidden workspace.

The migration scope includes every case in all five categories, including
Notebook and Python workflows beyond the original 216 quantum sources. Auxiliary
modules remain dependencies of their owning cases rather than additional completed examples.

The [native catalog](CATALOG.md) lists runnable counterparts. The
[five-category migration inventory](MIGRATION_STATUS.md) records all pinned
Python, Notebook and quantum-program sources. Pending inventory entries are not
presented as runnable examples. A translated circuit does not certify the rest
of a notebook's data preparation, training or execution workflow.

Original Qmod files, source identity, licenses and importer regressions are kept
in [test fixtures](../../tests/fixtures/classiq_qmod). They are not loaded by
user examples. Importer development tools live under `tools/`.

The [local acceptance report](ACCEPTANCE.md) records the fresh 216-program
comparison run, approved source repairs and remaining workflow/release limits.

Regenerate navigation with `python -m tools.synthesis_catalog --write`.
The separate [Circuit and framework examples](../README.md) cover gate-level
construction, training, simulation and deployment.
