# Compute resources

Directly controlled devices live in `local/`, external task systems in `remote/`, and bounded realtime messaging in `realtime/`. Runtime chooses and schedules work; Compute adapts the chosen resource. The root retains the published device lookup and platform contract exports. Numerical algorithms belong to Simulation, and program transformation belongs to Compiler.

Allowed dependencies: Core contracts and optional vendor SDKs inside their owning adapters.

Published entry points: flagquantum.compute; fq.submit for remote jobs. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/team/compute -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
