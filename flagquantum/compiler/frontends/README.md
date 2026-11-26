# Compiler frontends

Owns Torch capture and the existing structured Python frontend. `_structured/` remains an internal frontend implementation. Register construction, lowering and the main construction pipeline stay in their existing owners until #1084 integration. Do not execute simulations or manage devices.

Allowed dependencies: Core IR and Compiler transformations; Torch for the Torch frontend.

Published entry points: Existing fq.Module capture and compiler.module_builder/compiler.training entry points. Historical import paths are forwarding facades;
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
