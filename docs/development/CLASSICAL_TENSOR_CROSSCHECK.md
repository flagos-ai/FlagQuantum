# Native classical/tensor cross-check composition

The later [complete notebook workflow](CLASSICAL_TENSOR_NOTEBOOK.md) closes the
concrete 13-cell notebook replacement left open by this batch. Generic provider
and foreign-interface coverage remain separate open work.

The original Qualtran v0.7.0 `xcheck_classical_quimb` examples compose input
integer kets, a real component, output effects, tensor selection and contraction.
This example uses existing FlagQuantum owners for that complete numerical path.
No reference package or test fixture is imported by the executable example.

Compiler owns `component_gate_function.bit_gate_function`, the shared unchanged
primitive wrapper formerly private to control predicates. It retains primitive
SSA lowering, one-bit THRU element order, explicit returns and physical bodies.
All production consumers import this owner directly; the old definition is
removed. Existing Program verification still owns signature admission.

`examples/algorithms/classical_tensor_crosscheck.py` explicitly selects X, CNOT,
Toffoli or unsigned Add. X/CNOT/Toffoli use the shared wrapper; Add uses the
existing factorized addition graph. Canonical integer definitions supply their
own preparation/effect values. The three small primitive unitary matrices come
from their actual physically bound bodies and use Simulation's existing unitary
projection/binder. Add uses existing factorized CNOT/AND/adjoint suppliers; it
never obtains a whole-word unitary or classical truth table. Numerical callbacks,
Compiler selection and physical binding retain their established boundaries.

The scalar returned by `contract_classical_crosscheck` is an exact complex
amplitude, not a success probability. Explicit outputs bypass classical output
inference; omitted outputs use the existing native classical callbacks. This
example's greedy contraction path is intended for the small original examples;
wide Add uses the separately documented explicit cotengra path. Provider choices
are declared, not inferred from Python method names or process-global state.

```bash
python examples/algorithms/classical_tensor_crosscheck.py
```

This composes the four original concrete inputs, including Add(5) at a=2, b=3,
and retains X's explicit incorrect output with overlap zero. Source-wide
provider discovery, full notebook execution under replacement, open-port text
and planner-performance differences are not closed by this composition alone.

## Verified scope

The CROSSCHECK-PIPELINE batch adds 115 cases. Complete provider matrices for X,
CNOT, Toffoli and Add(1..3) match the unchanged source tensors. Exhaustive basis
inputs compare source classical outputs and check both correct and incorrect
output effects. Seven closed physical graphs cover the three primitives and
Add(1,2,3,5): every measurement history preserves the unnormalized complex
all-zero amplitude `2**(-m/2)` and leaves the physical pool zero. Each also runs
four trajectories through `circuit.append(op)`. Existing entangled-input,
spectator, ordered-gate and source resource tests remain in the regression.

The four unchanged upstream test functions now use this executable composition
in their factory replacement. The test-only whole-Add matrix remains solely as
an independent comparison in other tests; it no longer supplies these four
examples. A refusal test prevents Add(5) from calling the primitive unitary
supplier. Explicit output effects bypass classical inference.

Local results: 1,509 concentrated cases passed (47.82 seconds), with 18 existing
reference numerical warnings; 84 CPU/tensor integration cases passed (6.31
seconds), with two existing warnings. Strict typing covers all nine changed
Python files. Black, Ruff, Compiler scope, architecture, multi-level IR and
protected API checks pass. Combined example/wrapper coverage is 99% (117
statements, 34 branches); the direct module-entry guard is exercised separately
by running the example. The extracted wrapper's normalized AST equals its old
definition. Eight reference blobs, including the notebook, match the frozen
inventory. The earlier default-suite timeout remains open and was not repeated.
Evidence logs use `g2-crosscheck-pipeline-` in the reference workspace.

## Separate resource accounts

Counts below are freshly measured for the original closed concrete examples.
The source model includes logical vector costs, rather than only physical X
preparations. Atomic CCX is retained; it is not a zero-T synthesized Toffoli.

| Example | Actual closed primitive instructions | Source closed QEC model | Physical pool / source peak |
| --- | --- | --- | --- |
| X | X2 | Clifford3 | 1 / 1 |
| CNOT | X3, CX1 | Clifford5 | 2 / 2 |
| Toffoli | X5, CCX1 | Toffoli1, Clifford6 | 3 / 3 |
| Add(5) | X6, H12, T8, Tdg8, CX45, S4, measure4, conditional CZ4, reset4 | AND4, Clifford29, measurement4 | 14 / 14 |

This batch changes composition and ownership, not the existing arithmetic
emission or cost definitions. No parent file closure or source-wide parity is
inferred from these local examples.
