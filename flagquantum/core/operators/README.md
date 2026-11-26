# Operator semantics

Owns Pauli, fermionic and bosonic algebra, observable models, channels, superoperators, embedding and the immutable operator registry. Measurement requests live in `../measurement.py`; noise composition lives in `../noise.py` and hardware noise profiles in `../noise_profile.py`. Circuit method installation belongs to Compiler construction. Do not import Compiler, Runtime, Simulation or provider SDKs.

Allowed dependencies: Core models and the existing tensor representation.

Published entry points: fq.X, fq.Y, fq.Z, fq.Observable and existing operator exports. Historical import paths are forwarding facades;
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
