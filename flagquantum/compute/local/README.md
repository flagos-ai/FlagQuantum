# Local devices

Owns platform contracts, device lookup, PyTorch/FlagOS/FlagGems adaptation, target capability observation and allocator peak observation. A local device is controlled by this process; external queues belong to the sibling Remote domain. Do not choose simulation algorithms or training policy here.

Allowed dependencies: Core contracts, Torch and lazy optional accelerator libraries.

Published entry points: flagquantum.compute.get_platform_runtime. Historical import paths are forwarding facades;
the implementation in this directory is authoritative. See the
[relocation ledger](../../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for compatibility ownership and removal gates.

## Ten-minute contributor path

1. Read the implementation matching the behavior under change and its existing test.
2. Make one focused change inside this domain, keeping published contracts unchanged.
3. From the repository root, run:

   ```sh
   python -m pytest tests/unit/test_training_peak_memory.py -q
   python tools/architecture/check_architecture.py
   ```

CPU tests establish behavior and fixture conformance; physical hardware and
distributed capacity require their separate evidence gates.
