# Evolution orchestration

Owns the existing evolution workflow, preparation and native plan composition. Numerical integration lives in Simulation dynamics. Keep `_plan.py` and `_native_plan.py` internal, and preserve the published lindblad entry points through the historical facade. Do not duplicate integrators or operator semantics.

Allowed dependencies: Core models, Runtime planning and Simulation dynamics interfaces.

Published entry points: Existing flagquantum.lindblad workflow exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_structure_import_compatibility.py -q
   python tools/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
