# Recursive phased component evaluation

`simulate_phased_component_calls` composes the existing
`ComponentClassicalState` for stored typed-call hierarchies. A caller previously
had to implement a recursive callback that reconstructed the same state for
each nested component. This entry now owns that composition once. It follows
Qualtran v0.7.0
[`PhasedClassicalSimState._recurse_impl`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py)
without introducing a second numerical interpreter.

## Boundary and order

Compiler provides an explicit mapping from function name to dependency-ordered
call positions. Simulation does not import Compiler, discover methods, flatten
graphs or invent a call order. Direct value/phase leaf actions retain their
existing dispatch and require no body schedule. Only a value result of
NotImplemented with no phase enters the named stored body. Missing schedules
fail explicitly and retain the recursive exception context already implemented
by the state owner.

Each recursive state starts with unit phase and returns its accumulated phase
to its parent. The same RNG or fixed-outcome mapping is inherited at every
level, with RNG precedence unchanged. Fixed measurement identities are local
operation positions, just like source BloqInstance IDs; this is not a global
path-to-instance remapping. The existing field encoders, value validators,
assignment order and measurement feedback owner are reused. The order mapping
is snapshotted without modifying caller metadata.

The interface returns final named values and their unnormalized accumulated
phase. It does not return a normalized quantum branch amplitude: measurement
projection factors still belong to physical execution. The state class remains
available for explicit single-step inspection and custom recursive evaluators.

Pure source recursion uses a separately constructed default random handler.
That policy is not generalized from the phased implementation or silently
changed; this new entry supports phased evaluation only. Foreign lazy Python
decomposition exceptions and wider Core dtype admission remain open.

## Source and physical verification

The original `ComposedPhasing` fixture is imported from the frozen source test
module without changing its code. Its OneState followed by n Z gates is
represented with native owned allocation/X and Z leaves. At n=0/1/2/3/7 and
depth 1/2/4, recursive values and phase match original source dispatch. Full
source gate names and operands match the physical stream in order. Full
source tensors match native append amplitudes, including their global sign;
the unused clean spectator stays zero. The notebook marker is registered only
in the test fixture; the source notebook is not collected or executed.

Nested MeasureX cases cover both input bits and both outcomes at depth 1/2/4.
All 12 basis branches and six complex-superposition branches match physical
unnormalized amplitudes, with unused clean spectator zero. Nine seed/depth
cases compare 16 successive samples each, complete generator state and RNG
precedence over an empty fixed mapping. Three Add hierarchies cover all 192
input/spectator combinations without caller-written recursion. Six negative
cases cover missing root/body order, invalid dependency visitation, banned
randomness, missing fixed outcomes and invalid child phase.

| Consumer | Measured primitives | Source model | Work/branches |
| --- | --- | --- | --- |
| Recursive numerical composition | 0 | Not applicable | No quantum work |
| OneState then n Z | X1, Zn | Original QEC model checked independently: zero T and measurement counts | One retained output, one unused clean spectator, no measurement |
| Nested MeasureX | H1, measurement1 | Original measurement model remains distinct from physical H | Four basis branches per depth; 1/sqrt(2) amplitude factor |

This is a `single_device_fast_path` inspection entry. Ordinary physical use
continues through `circuit.append(op)`. No protected API, Core schema, graph
transformation, backend policy or physical decomposition changes.

Q-0882/Q-0883 remain open. The fixed file inventory and existing checked-child
count are unchanged; recursive composition extends existing evidence rather
than creating new scenario IDs.

## Local verification

- 51 new tests, 3,106 concentrated regressions and 84 CPU/tensor-network checks passed. The concentrated suite retains 207 pre-existing numerical warnings; integration retains two optional-dependency/deprecation warnings.
- Strict typing of both changed Python files, Black, Ruff, team scope, architecture, multi-level IR and protected API checks passed. The state owner measured 97% coverage (141 statements, 54 branches); three existing refusal sites remain uncovered.
- Seven frozen reference blobs match the inventory, including original ComposedPhasing, OneState/ZGate and MeasureX sources.
- Default smoke/unit: `7 failed, 22011 passed, 300 skipped, 1037 warnings in 118.93s (0:01:58)`. Observed failure IDs match the seven known baseline IDs; 0 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result.
- Workspace evidence: `ftqc-reference-env/g2-component-phased-evaluation-{source,default-comparison}.json` and corresponding verification logs.
