# Runtime profiles

Owns training, scale and capacity profile definitions and their validation. Distributed execution consumes these profiles; the definitions are not distributed algorithms. Historical profile schemas and serialized field names remain unchanged. Do not certify physical hardware from CPU fixture results.

Allowed dependencies: Core models and Runtime evidence contracts.

Published entry points: Existing training/scale/capacity profile imports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/team/runtime -q
   python tools/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
