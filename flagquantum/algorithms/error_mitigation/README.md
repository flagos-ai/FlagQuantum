# Error mitigation

Owns CDR, folding, PEC and readout mitigation workflows. Preserve the existing public error_mitigation namespace. Noise models and quantum channels belong to Core; backend adaptation belongs to Compute.

Allowed dependencies: Core models and established algorithm/execution interfaces.

Published entry points: Existing error_mitigation, cdr, folding, pec and readout exports. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_algorithms_cdr.py -q
   python tools/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
