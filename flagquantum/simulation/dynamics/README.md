# Dynamics numerics

Owns Lindblad generators, problem validation, numerical integrators, density-matrix evolution and its adjoint. This is the numerical dynamics domain; other admitted evolution methods can join when implemented. Runtime evolution prepares requests and execution policy. Do not select resources, submit jobs or import Runtime.

Allowed dependencies: Core operator and measurement semantics plus tensor numerics.

Published entry points: Existing evolution entry points; runtime.evolution composes user workflows. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_structure_import_compatibility.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
