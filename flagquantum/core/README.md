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

- `ir/`: canonical circuit representation, validation, and serialization.
- `operator_schema.py`: canonical operation names and schemas.
- `parameters.py`: symbolic parameters and binding semantics.
- `target_capabilities.py`: vendor-neutral target capability vocabulary.
- `numerics.py`: precision and accuracy requirements, not numerical kernels.
- `contracts.py`: versioned cross-domain execution and evidence records.
- `_artifacts.py`: internal artifact construction shared by stable public types.

`runtime_config.py` is configuration data owned by Core; configuration
resolution and execution policy remain Runtime responsibilities.

## Internal IR levels

`ir/__init__.py` is the public IR: `CircuitIR` schema 1.0, and the only
serialized IR this product publishes. Behind it, `ir/levels.py` is the internal
boundary that crosses to the program level, `ir/diagnostics.py` owns the refusal
vocabulary, and `ir/program/` owns the program-level model, verifier, and
lowering, each in one file. None of it is exported from `flagquantum`, none of it
appears in `docs/public_api_v1.json`, and `IR_VERSION` stays `"1.0"`.

The rules that boundary must keep -- level entry and exit, value identity,
linearity, joins, failure categories, round trip, and rejection -- are declared in
`contracts/multi-level-ir-internal-v1-candidate.json` and read back by
`tools/check_multi_level_ir_contract.py` and
`tests/unit/test_multi_level_ir_contract.py`. To change a level's behavior, change
the owning file under `ir/program/` and run:

```bash
python tools/check_multi_level_ir_contract.py
python -m pytest tests/unit/test_multi_level_ir_contract.py -q
```

A refusal names a code from `ir/diagnostics.py`; adding a code means declaring it
there and in the contract, raising it somewhere in the boundary, and letting the
gate confirm both directions.

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
