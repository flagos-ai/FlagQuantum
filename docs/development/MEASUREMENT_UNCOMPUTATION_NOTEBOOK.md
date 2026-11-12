# Measurement-assisted uncomputation notebook

The reference is unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/MBUC.ipynb`, frozen blob
`631fa11a4a9392b8874e1f5956446e7b1b4c2e66` (Q-0879).
Native assets are `examples/notebooks/measurement_uncomputation.ipynb`,
`examples/algorithms/measurement_uncomputation.py`, and
`tests/unit/test_mbuc_notebook.py`. Apache-2.0 attribution is retained.

Run the notebook from the repository root, or:

```bash
python -m examples.algorithms.measurement_uncomputation
```

## Whole notebook disposition

All 23 cells are mapped: ten code and thirteen Markdown. All ten original code
strings execute without changing numerical statements, with only two optional
`show_bloq` presentations captured. The native notebook runs independently with
Qualtran/Cirq/Quimb imports forbidden. The original assertions' defects are
preserved and tested separately, not silently repaired in the reference.

| Original cells | Native disposition |
| --- | --- |
| 0,1,2 | Gidney references and AND-adjoint classical validity; all four valid inputs checked |
| 3,4 | Inconsistent target failure; all four invalid basis inputs also checked independently, with native diagnostic wording |
| 5,6 | MeasureX and discard without feedback, composed from existing owned components |
| 7,8,9 | Seed123 ten-trial fuzz test reproduces the original phase sequence and relative-phase failure |
| 10,11 | MeasureX, CBit-controlled CZ, discard; explicit native feedback graph and physical append |
| 12,13 | Ten corrected seeded trials finish with phase one |
| 14,15,16 | Exhaustive-outcome explanation and explicit occurrence-ID selection; the sole measurement has ID zero |
| 17,18 | Four valid inputs times two outcomes, with current `sim.phase` checked; source stale-variable defect disclosed |
| 19,20,21 | Read-only per-call phase trace replaces a debug subclass that reads global state; every forced branch checked |
| 22 | Intermediate negative phase, immediate feedback correction, complete complex physical branches and resource distinctions |

Native cells have unique stable IDs, empty metadata, null execution counts and
no outputs or attachments; kernel metadata is normalized to Python3. There are
no embedded image assets. Both native instruction-table figures render and were
visually inspected. They replace the two logical musical scores with actual
physical instruction order, measurement-slot writes and explicit conditions.
They do not claim source layout equivalence. A generic circuit picture that
omits feedback metadata would be misleading here and is not used.

## Interfaces and ownership

`uncomputation_program(feedback=...)` composes existing MeasureX, owned CBit
feedback-CZ and classical-discard definitions with ComponentProgramBuilder.
Ports q1,q2 are THRU; target is LEFT-only. All are scalar quantum bits, ordered
q1,q2,target MSB-first. The measurement result is an owned CBit until discard.
There is no newly allocated quantum scratch, alternate graph or registry.

Compiler supplies the typed calls and their occurrence positions. Simulation's
existing ComponentClassicalState binds inputs and outputs, resolves fixed
outcomes or RNG, and accumulates phases. Explicit numerical handlers reuse
MeasureX's deferred output sign and the controlled-CZ phase owner. The example
never interprets arbitrary foreign methods or primitive bodies. Fixed ID0
outcomes override RNG using the existing state contract.

For both naive and corrected graphs, all eight basis assignments, including
inconsistent targets, and both forced outcomes match source. Every actual call's
committed values and accumulated phase are compared. Source dangling input and
output visits add no operation or phase; native traces list only actual calls.
The independent AND-adjoint guard separately accepts precisely consistent
predicates. Neither the source feedback circuit nor its native analogue inserts
that guard into a physical quantum computation.

`phase_trace(state)` reads the given state after each call. For controls11,
target1 and outcome1 the corrected trace is (-1,1,1): measurement, conditional
CZ, discard. Other valid cases stay at phase one. It does not create a second
simulator, use a global state variable or normalize quantum branches.

Users still call `circuit.append(EraseAnd())`. The corrected primitive stream
comes directly from the existing exact AND adjoint. The deliberately naive
example omits only conditional CZ. Typed graph lowering matches the same
H/measurement/conditional-CZ prefix; the example explicitly resets consumed
caller-owned target afterward. This physical reset is separate from discarding
the numerical classical record. No pool-retirement contract is changed.

## Source defects retained as open evidence

Cells 18 and 21 assert `phase == 1.0`, referring to the preceding fuzz loop's
variable instead of the state just simulated. An independent test executes the
unchanged cell with stale phase -1 and observes rejection of a correct current
simulation. With stale phase +1 and the naive graph substituted as input data,
the original assertions pass even though the final current phase is -1.

Cell 20's DebugPhasedClassicalSim.step reads global `sim.phase` instead of
`self.phase`. It happens to work in the original loop because the global points
to that instance. A second live instance demonstrates that its negative phase
is printed as positive when the global names another state. Native tracing
observes the actual state directly. These are explicit native corrections,
not claims that the frozen source was changed or already checks these cases.

The original full feedback composite's QEC cost query also raises
DecomposeNotImplementedError for C[CZ]. That unavailable estimate is tested and
reported as unavailable, not zero. Q-0879 therefore remains open under the
frozen source-defect rule, despite delivery of the complete native workflow.

## Complex branches and resources

Exact physical tests retain both unnormalized measurement branches, including
1/sqrt(2). A complex superposition covers every valid control assignment and
both spectator values. Corrected erasure preserves those phases and spectator
entanglement with target zero. Naive erasure has a relative sign on the11
sector in outcome1, providing a real negative case. Both forms also execute
through ordinary dynamic append across eight trajectories. No global-phase
correction hides the negative case.

| Reading | Result |
| --- | --- |
| Corrected emitted stream | H1, measurement1, conditional CZ1, reset1; zero T |
| Naive emitted stream | H1, measurement1, reset1; zero T; intentionally wrong relative phase |
| Physical ownership | controls0/1 retained, caller-owned target2 consumed/reset, spectator3 untouched, classical slot0 used |
| Unchanged And(uncompute=True) QEC model | Clifford1, measurement1, T0; not a syntax count of its emitted stream |
| Unchanged MeasureX QEC model | measurement1; H is not separately counted by this model |
| Unchanged full CBit-controlled feedback composite model | Unavailable: C[CZ] lacks cost decomposition |
| Numerical evaluation and trace presentation | Zero emitted quantum primitives; no physical scratch allocation |

This batch changes examples/tests/docs only. There is no production-layer,
protected-API, dependency or CI modification, and Simulation does not import
Compiler. G3 and merging remain unauthorized. The previous default-suite timeout
remains recorded and is not rerun; CPU results are not GPU or scalability proof.

## Local acceptance

Implementation `98117c39`; seven new cases, 1629 related tests pass in 57.39s
(24 existing numerical warnings), and 84 CPU/TN checks pass in 8.88s (two optional
dependency/deprecation warnings). Both Python files pass strict mypy, Black and
Ruff. Scope, architecture, multi-level IR, protected API and example checks pass.
The notebook, classical_sim.py, and_bloq.py, x_basis.py, discard.py and controlled.py
match all six frozen source blobs. All ten original and ten native code cells
execute, and both final ordered tables were rendered and viewed.

Helper coverage is 89% in the regression and 97% after the instrumented standalone
example (110 statements, 24 branches). Remaining uncovered paths reject an
unbound callback name and an invalid CZ callback value; no whole-repository
coverage claim is made. No default-suite timeout or CI/GPU gate was rerun.
Q-0879 moves from pending to audited/in progress but is not closed. Parent
counts remain 22 closed/1258 open; active parents become 391, with 1184 capability
records unchanged. Next: Q-0898 complete classical/tensor cross-check source
ownership and failure-domain audit, reusing existing example/test dispositions.
