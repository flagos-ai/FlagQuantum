# Classical optimizers

Owns COBYLA, Nelder-Mead and SPSA updates used by algorithm workflows. It does not own quantum execution, circuit compilation or backend selection.

Allowed dependencies: Numerical Python and existing algorithm types.

Published entry points: Existing optimizer exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_algorithms_nelder_mead.py -q
   python tools/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
