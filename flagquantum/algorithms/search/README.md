# Search algorithms

Owns Grover search construction. Reuse admitted primitives rather than duplicating arithmetic or oracle implementations. Do not own backend or simulation selection.

Allowed dependencies: Core models and algorithm primitives.

Published entry points: Existing Grover exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_algorithms_grover.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
