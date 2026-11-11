# Classical component graph execution

Simulation owns `component_classical_state.ComponentClassicalState`, a numerical
consumer of the existing Core `ProgramRecord`. Compiler supplies a complete,
dependency-respecting tuple of call positions; it retains graph construction,
ordering and transformation ownership. This is a `single_device_fast_path`
inspection path. It creates no IR, registry, numerical backend, quantum gate,
physical work bit or execution policy. Protected APIs remain unchanged and users
continue `circuit.append(op)` for quantum execution.

## Source and interfaces

The unchanged reference is Qualtran v0.7.0
`simulation/classical_sim.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Copyright and provenance remain in
the numerical owner. Seven source Git blobs, including original tests, registers,
composite graphs, arithmetic addition and basis gates, match the frozen inventory;
workspace evidence is `g2-component-classical-state-source.json`.

| Source responsibility | Native owner or boundary |
| --- | --- |
| Input assignment and `_get_in_vals` | Existing `classical_registers` binding and collection; assignments keyed by Core `ValueRef` |
| `ClassicalSimState.step` | `ComponentClassicalState.step`, explicit one-call visitation and value/phase callbacks |
| Direct values versus recursive evaluation | Existing `evaluate_classical_action`; explicit caller-supplied recursion |
| Distribution handlers | Existing `draw_classical_value`, explicit occurrence IDs and RNG/fixed outcomes |
| Pure and phased `_update` | Existing `commit_classical_outputs` and `commit_phased_classical_outputs` |
| `finalize` and `simulate` | Return-metadata collection and step-to-exhaustion |

The admitted graph has one typed logical block, explicit returned values and
unconditional typed calls. Each operation is visited exactly once after all its
operands are available. Function definitions and caller values are not rewritten.
Ports use integer/bit scalar or row-major array storage and existing numerical
codecs. These word assignments do not reinterpret physical bit order; existing
emitters retain MSB-first register decomposition and work ownership.

A call evaluates values before phase, assigns outputs before applying or rejecting
phase, and retains the assignment prefix and last position on failure. Pure mode
rejects even an explicit identity phase. Phased mode resolves measurement phase
after sampling and binding. Occurrence identity is independent of visitation
position: tests use ID 73 at position 0. RNG takes precedence over fixed values;
no configuration bans stochastic values. Input distributions use native sentinel
ID -1 rather than a foreign graph dangle. A native step visits one actual call;
source sentinel visits have no native counterpart. No implicit decomposition
method discovery or foreign exception translation enters production.

## Verification

- 40 new cases include 192 exhaustive Add graph evaluations: depths 1, 2 and 3,
  each with all 64 input/spectator combinations. Every call's output assignment
  and final output match original source stepping; recursive evaluation uses
  this same native numerical consumer.
- Scalar, vector, matrix and empty-array ports test storage, immutability,
  unfinished collection and exhausted iteration. Ordering, output-domain,
  distribution, callback and phase failures retain explicit assignment prefixes.
- X measurement uses an actual H-before-measure native physical body. All four
  input/outcome pairs match the source numerical phase and physical branch
  amplitude, including its unnormalized factor 1/sqrt(2). The unused clean bit
  remains zero. Seeded sampling compares complete RNG state and precedence.
- An X/S/Z graph reconstructs the full complex matrix from numerical values and
  accumulated phase. It matches the original tensor and ordinary append execution
  on a complex superposition, with its clean spectator zero.
- Concentrated regression: 553 passed without warnings. Existing component
  emitter tests retain source gate/operand/feedback comparisons, all 16 forward
  and 256 roundtrip measurement histories and work erasure.
- Original `classical_sim_test.py`: 28 passed unchanged, one notebook test
  deselected, two original collection/marker warnings. This is source-only
  regression, not a complete original-file native replacement.
- CPU, representation and tensor-network regression: 84 passed, two recorded
  optional-dependency/deprecation warnings. The new numerical owner has 97%
  statement/branch coverage (99 statements, 36 branches); typed-block and
  return-control-flow refusal lines 85 and 88 remain uncovered.
- Strict typing of seven files, Black, Ruff, Simulation scope, architecture,
  internal multi-level IR and protected API checks pass.
- Default smoke/unit run: 6 failed, 21,868 passed, 300 skipped, 1,037 warnings in
  137.00 seconds, stopped at its maxfail bound. The six failure IDs are a subset
  of the seven recorded baseline IDs; one baseline failure was not observed.
  No new failure was observed; the default suite is not fully green.

## Separate resource accounts

| Consumer | Measured primitive stream | Original QEC model | Workspace |
| --- | --- | --- | --- |
| Numerical interpreter | Zero gates | Not applicable | Zero physical work/slots; numerical assignment storage |
| Two Add(2,3) calls with spectator | H12, T8, Tdg8, CX40, S4, measurement4, conditional CZ4, reset4 | AND4, Clifford22, measurement4 | Two clean work bits, four classical slots |
| X measurement | H1, measurement1 | measurement1 | One unused clean spectator |
| Coherent X/S/Z graph | X1, S1, Z1 | Clifford3 | One unused clean spectator |

The source X-measurement model omits the physical H; its measured primitive
stream is retained separately. Model categories are not primitive gate counts.
The interpreter neither changes nor substitutes these existing append emitters.
No hardware, GPU or distributed performance claim is made.

## Explicit remaining boundaries

Q-0882 and Q-0883 remain open. The complete source includes finite-field/dtype
traversal, truth-table/formatting helpers and foreign lazy decomposition exception
semantics beyond this slice. Underlying numerical codecs support signed values,
but Core executable graph admission still excludes signed metadata, so this is
not full signed graph support. Non-call primitive entries, conditional calls and
control-flow graphs are deliberately refused here.

The source initializes empty external array input but raises
`KeyError(RightDangle)` on finalization. Native typed empty-array return metadata
can collect that value. Tests retain the original failure and document the
representation difference rather than modifying the reference. No whole-file
closure, new child ID or larger completion denominator follows from this batch;
G3 remains unstarted.
