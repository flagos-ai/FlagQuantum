# Execution results

Owns execution result objects, result adapters, MPS result handling and measured execution records. Preserve existing result schemas, public export identity and serialization. Do not perform numerical execution, construct circuits or select resources here.

Allowed dependencies: Core models, Runtime result protocols, existing tensor types
and direct Simulation sampling kernels behind published result methods.

Published entry points: fq.run results and existing result extraction functions. Historical import paths are forwarding facades;
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

## Sampling methods

The published `MPSMonteCarloResult.sample` method retains its original input
and retention validation and delegates retained-mixture numerical draws to
Simulation MPS sampling. Its public signature, annotations, result schema and
historical pickle locator are unchanged. Runtime does not retain a duplicate
sampling loop.
