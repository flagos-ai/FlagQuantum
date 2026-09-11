# Phase 36 physical circuit plan evidence

Date: 2026-09-10

## Outcome

Compiler now produces an immutable `PhysicalCircuitPlan` from verified target
legalization. It composes source lineage, topology mapping, native-gate
decomposition, and dependency scheduling while keeping Core `CircuitIR` as the
only circuit-semantic authority.

## Evidence

| Claim | Evidence |
| --- | --- |
| Source objects survive topology and native legalization | Both result objects retain and hash-check `source_program` |
| Routing provenance is exact | Every mapped operation and inserted SWAP records `source_instruction_index` |
| Mapping is replayable | Each routing SWAP becomes a before/after `MappingTransition` and replay reaches the recorded final layout |
| Final operations are topology legal | Every final two-wire instruction is checked against the matching coupling map |
| Native expansion remains traceable | Each physical instruction records topology index and replacement ordinal |
| Schedule evidence is joined, not copied into another IR | Physical records reference the authoritative final `CircuitIR` index and existing schedule record |
| Identity is deterministic and tamper evident | Plan identity hashes mapping, instruction, target, legalization, and schedule evidence |
| Artifact compilation cannot bypass the plan | `ArtifactCompilationResult` requires the actual plan object and includes its identity in compilation identity |

## Verification

The Phase 36 focused tests cover native decomposition without topology,
topology mapping replay, native lowering of inserted routing SWAPs, coupling-map
misbinding, record tampering, source-object retention, and complete
artifact-compilation integration.

The combined hybrid compiler, Core artifact, binding, artifact compilation,
Runtime/Deployment adapter, local execution, and private-contract suite passed
**285 tests**.

## Deliberate exclusions

The plan does not introduce TargetIR, timed or pulse scheduling, directed-edge
synthesis, calibration-aware placement, crosstalk modeling, physical ancilla
allocation, provider submission, public exports, default-path changes, or
performance claims.
