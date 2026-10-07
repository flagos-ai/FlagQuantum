# Core

Core defines FlagQuantum's backend-neutral language: circuit IR, operator and
parameter semantics, target capabilities, numerical requirements, and the
versioned contracts exchanged across domain boundaries. These definitions are
the single source of truth consumed by Compiler, Runtime, Simulation, and
Compute or Remote.

Core does not compile or execute programs, choose resources, implement
simulation kernels, call vendor SDKs, or expose service and framework adapters.
It must remain independent of Compiler and Runtime implementations, Simulation,
Compute, Remote, ecosystem frameworks, gateways, and infrastructure libraries.

## Where to start

- `ir.py`: canonical circuit representation, validation, and serialization.
- `operator_schema.py`: canonical operation names and schemas.
- `parameters.py`: symbolic parameters and binding semantics.
- `target_capabilities.py`: vendor-neutral target capability vocabulary.
- `numerics.py`: precision and accuracy requirements, not numerical kernels.
- `contracts.py`: versioned cross-domain execution and evidence records.
- `controlled.py` and `_composition.py`: the per-instruction adapters behind
  `Circuit.adjoint`, `Circuit.power`, and `Circuit.control`. `operator_schema.py`
  declares each opcode's adjoint, power, and control rule; `controlled.py` emits a
  controlled expansion; `_composition.py` reads one recorded instruction, picks the
  route its own record selects, and refuses by name when no route applies. The three
  public methods are the only caller-facing statement of this behaviour.
- `_artifacts.py`: internal artifact construction shared by stable public types.

`runtime_config.py` is configuration data owned by Core; configuration
resolution and execution policy remain Runtime responsibilities.

## Ten-minute change path

For a small semantic change, modify the narrowest owning file, add its Core
contract test, and run:

```bash
python -m pytest tests/team/core tests/api_contract \
  tests/integration/test_cpu_vertical_slice.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Before changing an exported type, serialized schema, IR version, or protected
behavior, read `docs/development/PUBLIC_API_PROTECTION.md`. Such a change needs
an approved contract proposal and migration evidence; do not update snapshots
merely to make checks pass. Ordinary compiler transforms, runtime policies,
backend adaptations, and simulation math should be changed in their owning
domains without extending Core.
