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

- `ir/`: canonical circuit representation, validation, and serialization. The
  public IR is `ir/__init__.py`; the internal level boundary behind it, and the
  shortest path for changing a level, are in `ir/README.md`.
- `operator_schema.py`: canonical operation names and schemas.
- `ir/state_space.py`: internal finite product dimensions and ordered explicit
  basis labels shared by dynamics. Explicit labels do not infer tensor factors
  or certify physical sectors; no circuit or root export is added.
- `parameters.py`: symbolic parameters and binding semantics.
- `rydberg.py`: the approved experimental Rydberg frontend over ProgramRecord.
  `ir/program/physical.py` owns its typed control profile and semantic checks;
  `ir/program/physical_codec.py` owns native source encoding using the existing
  scalar codec; the numerical plan stays outside Core.
  `geometry.py` owns finite Cartesian pairs for this profile and the separately
  scheduled Remote normalizer replacement. Neither file executes dynamics.
- `ising.py`: the experimental four-sigma, total-vacuum Ising source builder.
  It reuses the same ProgramRecord, typed physical profiles and source codec.
  Braid-only sources retain version 1.1; exactly one `dephase_pair()` observation
  selects source 1.2. It observes the current first pair without an outcome,
  physical time or classical bit. Rydberg source 1.0 is unchanged.
  Charge projectors are CPU mathematical inputs. Native braid execution,
  executable plans and numerical F/R/Kraus actions remain outside this source gate.
  Runtime admits matching native plan 1.2 separately; Core owns no numerical
  observation algorithm or execution state.
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

For native-control construction, run:

```bash
python -m examples.rydberg_program_geometry
python -m pytest tests/unit/test_rydberg_program.py \
  tests/unit/test_rydberg_source_codec.py \
  tests/unit/test_rydberg_program_compatibility.py -q
```

A typical geometry change belongs in the typed profile and its consumer tests;
numerical evolution belongs in Simulation. Do not reserve physical opcode names
globally or modify old plan fixtures to hide a failure.

For discrete braid source construction and its admission/refusal boundaries:

```bash
python -m examples.ising_braid_source
python -m pytest tests/unit/test_ising_braid_source.py \
  tests/unit/test_ising_braid_contract.py \
  tests/unit/test_rydberg_program_compatibility.py -q
```

Append order is braid order, never a timestamp. Keep typed source verification
in the existing physical-profile boundary and exact serialization in its codec;
do not introduce circuit lowering, a Hamiltonian logarithm or another executor.
