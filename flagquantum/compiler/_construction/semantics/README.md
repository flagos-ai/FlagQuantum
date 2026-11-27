# Shared construction semantics

The Compiler team owns this package. Exact numeric, fixed-word, phase, predicate and unitary semantics shared by syntax consumers belong here.

`numeric_expression.py` owns exact signed-word and binary-scale inference.
`fixed_point.py` and `fixed_point_program.py` own bounded quantized arithmetic.
The phase, predicate, rotation, QPE and unitary modules validate and expand their
named construction effects through existing models and gate helpers.

This package must not capture source syntax, select execution backends, execute
numerical workloads, submit jobs or define another IR. Allowed dependencies are
the standard library, existing Core semantics, Compiler's shared HybridProgram
and construction records, sibling construction implementation modules, and the
existing Torch numerical dependency where required by current validation.

## Entry points and a small change

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal entry points are the existing named helpers consumed by the frontend,
verification and codegen modules.

Start with the named implementation and an accepted/rejected scenario in the
focused tests below. Preserve diagnostic ordering, exact numeric semantics and
cleanup evidence. Run from the repository root with project dependencies.

```bash
python -m pytest -q tests/unit/test_synth_fixed_point.py tests/team/compiler/test_construction_numeric_expression.py
python -m examples.synthesis.synthesis_offline_deployment
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

The offline Bell-state example captures a Python `@qfunc`, emits OpenQASM 3 with
H and CX, and builds a `DeploymentPackage` for `local/offline-line` with 256 shots.
It requires no credentials. The integration test reads the output back and
checks the Bell state. Change the source gates to exercise the compilation path.

## Numerical matrix decomposition

`unitary.py` expands bounded dense matrices using Givens rotations and registered
gates. Its elimination residual floor is `1e-15`, matching its existing phase
residual floor. Complete matrix reconstruction still has to pass the unchanged
`1e-12` absolute bound. This numerical matrix path does not edit submitted gate
parameters or symbolic expressions. It introduces no backend or public policy.

The previous `1e-16` floor admitted eigensolver/BLAS roundoff as extra controlled
rotation networks. In PR #1229's Linux coverage job `135597091704`, the unchanged
Lanchester case emitted 20,227 operations and depth 18,028, exceeding its existing
20,213/18,016 ceilings. The same source on macOS ARM64/Torch 2.14.0 emitted
20,213/18,016. With the residual floor aligned, the authoring host emits
19,194/17,107 and retains the full-state, absolute-phase, named-field and clean
workspace checks. These are diagnostic observations, not a timing or SDKBench
result. Linux confirmation remains required; the test ceilings are unchanged.

```bash
python -m pytest -q tests/team/compiler/test_construction_unitary_matrices.py
python -m pytest -q tests/integration/test_synthesis_native_lanchester.py
```

The matrix checks replay every computational-basis column of complex Fourier
and seeded random matrices with and without coherent controls, including global
phase and clean workspace. They cover both the allocator's usual workspace
count and a controlled four-qubit matrix with only two clean work bits.

### Controlled matrices with two clean work bits

When a controlled block has fewer clean work bits than its full conjunction
ladder needs, `unitary.py` reuses `mcx.one_clean_mcx` to compute the conjunction
into one clean flag using a second clean work bit and temporarily borrowed
controls. The work bit, flag and borrowed controls are restored. The block's
registered gates are controlled by the computed flag, then the conjunction is
uncomputed. The register scheduler supplies only the available proven-clean tail, capped
by the existing synthesis width and workspace constraints. It no longer
requires the entire preferred tail before passing any work bits. The ordinary
full-workspace path is unchanged; fewer than two clean work bits still use the
existing ancilla-free expansion.

[Issue #1240](https://github.com/flagos-ai/FlagQuantum/issues/1240) records the
original two-work-bit Fourier example: 51,125 gates and approximately
`2.2e-12` inactive-branch simulator drift. Reusing the existing conjunction
implementation emits 2,673 gates for that example and gives zero observed
inactive-branch error on the same macOS ARM64 / Torch 2.14.0 environment. The
full-workspace example emits 2,431 gates. These counts describe this input,
not a universal cost guarantee or a hardware performance claim.

The regression checks all input columns of both Fourier and seeded random
matrices on active and inactive branches, retaining the absolute `1e-12`
acceptance bound, the input's `0.37` global phase and clean return of both work
bits. This correction does not establish the accuracy of every smaller-workspace
fallback, and does not change workspace allocation, IR, public contracts or
simulation tolerances.

### Harmonic oscillator resource regression

Aligning the residual floor also removes roundoff-generated rotations from the
native harmonic oscillator. A controlled replay on macOS ARM64 / Torch 2.14.0
changed only the elimination floor: `1e-16` produced 7 qubits, 387 operations and
depth 335; `1e-15` produced 7 qubits, 203 operations and depth 173. Both used two
workspace qubits and passed all 32 input columns against the independent
reference, including absolute phase and clean workspace at the existing bounds.
Linux job `135633176038` also reported 203 operations and depth 173, passing those
semantic checks before failing the old exact resource assertion.

The regression test now pins the improved exact resource tuple `(7, 203, 173)`;
it does not widen a tolerance or replace equality with a looser resource ceiling.
The historical counts in the example provenance remain unchanged. The complete
harmonic suite additionally exercises nonzero bindings, resource refusal and
invalid-input rejection.

```bash
python -m pytest -q tests/integration/test_synthesis_native_harmonic.py
```

The public regression in
`tests/integration/test_synthesis_native_unitary.py` prepares a coherent control
and four-qubit input, applies a complex Fourier matrix and compiles with
`max_workspace=2`, `max_qubits=7` and `max_operations=10_000`. Independent dense
Circuit amplitudes and hidden-zero checks verify the emitted result. Previously
the scheduler ignored the tighter clean tail for this operation and proposed
three work bits, which the unchanged resource acceptance rejected.

The same scheduler correction makes the harmonic oscillator's one-work-bit
schedule and Lanchester's three-work-bit schedule feasible. Their previous
rejection assertions are replaced by explicit success, exact resource-bound
and independent state checks: all 32 harmonic input columns and the complete
Lanchester output state, including hidden-zero workspace. Operation, depth and
memory refusal tests remain in place. No test accepts either success or failure.

On the development host, the constrained harmonic schedule uses 6 qubits,
1,889 operations and depth 1,653 (its ordinary two-work-bit schedule uses
7 qubits, 203 operations and depth 173). The constrained Lanchester schedule
uses 15 qubits, 20,350 operations and depth 18,261; its observed full-state
maximum error is `9.788462989003254e-14`, with zero observed workspace leakage.
These input-specific observations show a width versus gate/depth tradeoff;
they do not replace the ordinary resource ceilings or qualify a benchmark.
