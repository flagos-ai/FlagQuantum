# Core

Operator algebra, observables, channels and embedding share `operators/`. Measurement requests live in `measurement.py`; noise composition and device profiles in `noise.py` and `noise_profile.py`. Core does not import Simulation to validate measurement semantics.

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
  selects source 1.2. Exactly one `measure_pair()` selects source 1.3 for a
  declared sampled observation; the two forms are mutually exclusive. Both
  observe the current first pair without adding physical time or a classical
  bit. The sampled builder returns the source, never an outcome. Rydberg
  source 1.0 is unchanged.
  Charge projectors are CPU mathematical inputs. Native braid execution,
  executable plans and numerical F/R/Kraus actions remain outside this source gate.
  Runtime separately admits matching native plans, including one explicitly
  seeded CPU trajectory for sampled source 1.3. Core owns no numerical observation algorithm, RNG,
  execution state or plan/environment version activation.
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
  The approved v2 circuit payload budget is 65,536 mapping/list entries, separate
  from the 4,096-entry structured envelope budget. Upgrade receiving readers
  before sending expanded artifacts. Executable v2 and structured v3 admission
  limits are unchanged; downstream compilation evidence has separate limits.
- `_compilation_evidence.py`: the explicit version dispatcher for stored Compiler
  evidence. `_compilation_evidence_v4.py` implements the approved unallocated,
  undirected many-to-many block profile. It checks structure, exclusive input
  and output membership, schedule dependencies, topology-space mapping replay,
  resource bounds, and canonical identities. It does not verify a claimed
  quantum operator residual. Compiler must recompute that witness from actual
  retained programs; Runtime and Deployment admission are separate changes.
  The v1-v3 readers and frozen identity fixtures remain unchanged.

`runtime_config.py` is configuration data owned by Core; configuration
resolution and execution policy remain Runtime responsibilities.

## Ten-minute change path

For a small semantic change, modify the narrowest owning file, add its Core
contract test, and run:

```bash
python -m pytest tests/team/core tests/api_contract \
  tests/integration/test_cpu_vertical_slice.py -q
python tools/architecture/check_architecture.py
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
  tests/unit/test_ising_sampled_source.py \
  tests/unit/test_rydberg_program_compatibility.py -q
```

Append order is braid order, never a timestamp. Keep typed source verification
in the existing physical-profile boundary and exact serialization in its codec;
do not introduce circuit lowering, a Hamiltonian logarithm or another executor.

## Rydberg presence source

`RydbergProgram.measure_presence(atom, *, at, efficiency)` records one named-atom
observation in the existing physical IR. It uses source 1.4 only when present.
Snapshot validation requires the time strictly inside the original control
horizon. Core owns scalar, roster and source validation; output-grid placement,
private sampling and conditional numerical execution belong to later domains.
The existing unconditional Lindblad assembler refuses this source before grid
assembly. Ordinary pulse/transport sources retain source 1.0 bytes.

## Compilation evidence reader change path

Execution classification: `single_device_fast_path`. This codec establishes no
simulation, hardware, distributed, performance, or SDK parity claim. The v4
contract landed in Integration before this Core implementation; see
`docs/api-changes/FQ-COMPILER-CZ-BLOCK-LINEAGE-20261118.md`.

For the shortest example, run the cancellation and SWAP/reversed-CX round-trip
scenarios in the codec tests. They keep collective input ancestry, including an
empty output, and demonstrate rejecting a resealed stale source or target:

```bash
python -m pytest tests/unit/test_compilation_evidence_v4.py \
  tests/unit/test_compilation_evidence_legacy_identity.py \
  tests/team/core/test_compilation_evidence_bundle*.py -q
```

The shared `read_compilation_evidence_bundle_json` entry dispatches exactly by
version. Never regenerate the v1-v3 fixtures, downgrade collective records into
single-source records, add numerical imports to Core, or treat successful codec
reading as permission to execute the artifact. The experimental read-only
`load_compilation_evidence` view preserves the v4 identity and canonical JSON;
it does not add execution admission. No production writer or automatic
block stage is enabled by the reader alone. The current support boundary remains
codec-only until those separately owned consumers pass their handoff tests.
