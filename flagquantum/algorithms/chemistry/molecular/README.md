# Molecular chemistry

Owns molecular workflows and integral preparation. Share chemistry data models with the sibling models package; keep general algebra in Core.

Allowed dependencies: Chemistry models and existing molecular numerical dependencies.

Published entry points: Existing algorithms.molecular and chemistry_integrals exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_algorithms_molecular.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
