# Construction circuit generation

The Compiler team owns conversion of verified construction programs into
`CircuitIR`, using the existing candidate and workspace evidence records.

| Module | Responsibility |
| --- | --- |
| `register_expansion.py` | Static register call expansion and inverse composition |
| `arithmetic.py` | Arithmetic gate emission and retain/recompute scheduling |
| `registers.py` | Register-effect composition, allocation and segment cleanup evidence |
| `boolean.py` | Boolean truth-table lowering |
| `gate_tape.py` | Checked gate-tape expansion |
| `preparation.py` | Uniform interval state preparation |
| `controls.py` | Bounded controlled operations and modular exponentiation |
| `transforms.py` | Pattern gates and real-transform decomposition |

Source capture belongs to `../frontend/`. Semantic verification and candidate
selection belong to `../verification/` and `../scheduling/`. This package must not
select runtime backends, execute circuits, submit jobs or define another IR.
Allowed dependencies are existing Core circuit semantics, construction models,
verification and lifetime helpers, Compiler gate kernels and optimization
passes, and the existing Torch numerical dependency.

## Entry points and compatibility

The public workflow uses `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal entries include
`arithmetic.schedule_arithmetic`, `registers.schedule_register_program`,
`boolean.lower_boolean_value` and `gate_tape.expand_gate_tape`.
Consumers import the authoritative modules directly. Former private lowering
aliases were retired under the [parent migration](../README.md#retired-private-imports).

## Make and verify a small change

For a preparation change, start with `preparation._uniform_interval` and the
accepted/rejected scenarios in `tests/team/compiler/test_construction_registers.py`.
Keep register allocation and cleanup evidence in `registers.py`. Check the full
state and workspace cleanup in the corresponding integration scenario.

Run the existing Bell-state workflow from the repository root.

```bash
python -m examples.synthesis_offline_deployment
```

It captures a two-qubit Python `@qfunc`, emits OpenQASM 3 with H and CX, and
constructs an offline `DeploymentPackage` for `local/offline-line` with 256 shots.
It also produces an OpenQASM 2 deployment package. No credentials are required.
The integration test reads the output back and checks the Bell state.

```bash
python -m pytest -q tests/team/compiler/test_construction_registers.py tests/team/compiler/test_construction_scheduling.py tests/team/compiler/test_construction_boolean_lowering.py tests/team/compiler/test_construction_codegen_hooks.py
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

For changes to controlled gates or real transforms, also run the corresponding
`test_synthesis_native_*.py` scenarios under `tests/integration/`. For arithmetic
scheduling changes, include cleanup and recomputation cases before running the
broader compiler suite.
