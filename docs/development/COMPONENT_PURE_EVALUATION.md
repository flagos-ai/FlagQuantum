# Pure classical evaluation of stored component bodies

`simulate_pure_component_calls` composes the existing `ComponentClassicalState`
with Compiler-supplied schedules and optional per-function persistent IDs.
Numerical callbacks, integer/field/opaque binding, output commits and phase
rejection remain with their existing owners. Simulation imports no Compiler
implementation, discovers no foreign decompositions and emits no physical gates.

The source is unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, specifically
`call_cbloq_classically` and `ClassicalSimState._recurse_impl`. The original
helper retains one default random handler for the process lifetime. Pure
recursion calls that helper without forwarding its parent's random handler.
The native module therefore owns one default generator: root calls with neither
RNG nor fixed values and every nested body use that stream. An explicit root
RNG takes precedence over root fixed values. Neither reaches nested bodies.
The existing phased evaluator continues to propagate the parent's policy.
These are separate upstream semantics, not interchangeable execution modes.

Every non-None phase, including numerical one, is rejected after output commit.
Missing schedules retain existing recursive exception contexts. No Core dtype
admission, graph structure, foreign exception translation or protected API is
changed. Users continue to append physical operations with `circuit.append(op)`.

## Acceptance evidence

- 36 RNG cases cover depths 0/1/3, default/seeded/fixed/both root policies and
  three seeds. Each compares 24 consecutive calls and both complete generator
  states against the original source. Tests temporarily seed the existing source
  handler, restoring it afterward; reference files are never edited.
- Nine phase cases reject 1, -1 and i at the same three depths in both engines.
- Three stored Add hierarchies compare all 64 inputs each with the original pure
  helper. Their exact physical primitive streams and operands match the source
  decomposition. Each reconstructs a complex superposition across all 64 logical
  inputs and checks all 16 measurement branches, including the 1/4 projection
  factor. Both work bits return to zero in every branch.
- Two missing-schedule cases preserve the root/nested cause chain.

The numerical helper emits zero primitives and owns no work bits. Physical
consumer counts and the original QEC model are separate: two Add operations
use the existing two work bits and four measurements; the source model reports
T-count 16 and measurement-count 4. Detailed primitive counts are recorded in
workspace evidence, independently of the model.

Q-0882/Q-0883 remain open for source-wide dtype admission, foreign decomposition
errors and complete original-test replacement. Empty-array and input-distribution
identity differences recorded by earlier batches remain open. This batch closes
only the missing pure recursive RNG policy, without starting G3.

## Local verification

50 new tests, 3,200 concentrated regressions (207 existing numerical warnings)
and 84 CPU/tensor-network integration checks (two existing warnings) passed.
Strict typing, Black, Ruff, Simulation scope, architecture, internal IR and
protected API checks passed. The new evaluator has 100% statement coverage
(20 statements); it adds no independent branch owner. Six frozen source blobs
match the inventory. Physical primitive counts are CX40, H12, T8, Tdg8, S4,
measurement4, CZ4 and reset4, compared independently with source T-count16 and
measurement4. Workspace evidence is recorded in
`ftqc-reference-env/g2-component-pure-evaluation-source.json` and verification logs.

Default smoke/unit result: `5 failed, 21904 passed, 300 skipped, 1037 warnings in 132.66s (0:02:12)`. All observed failure IDs belong to the seven known baseline IDs; 2 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result. The comparison is retained in `ftqc-reference-env/g2-component-pure-evaluation-default-comparison.json`.
