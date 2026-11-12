# Native integer bindings and static classical branches

The user authorized the next native-library batch: classical integer arrays,
static classical `if`, the complete retained BB84 program, independent full-state
Circuit validation and a fresh 216-source regression. This extends the unreleased
[classical candidate](../../contracts/synthesis-native-classical-candidate.json)
to version 1.1. The existing public signatures and Stable Core snapshots remain
unchanged. Classification is `single_device_fast_path`; implementation is
allowed and `release_authorized=false`.

## Meaning and boundaries

In a typed native qfunc, `int` and `list[int]` declare synthesis-time classical
values, separate from quantum `QNum` and `QArray`. Keep the existing `float` and
`list[float]` behavior. Classical inputs must exactly match the signature, and
are copied into immutable per-synthesis data. Integer values must be built-in
`int`, excluding `bool`, within ±2**53. A float such as `1.0` is not an integer
binding. Flat built-in lists/tuples contain 1..4096 elements; total bound scalar
and array elements cannot exceed 4096. Integer helpers admit bound arrays and
literal arrays, with the same type/range checks as root binding. They retain
integer types across helper calls, arithmetic and in-bounds element reads.
Real formals still specialize integers to floats, so passing a real formal back
to an integer formal or using it as an index is refused.

Typed integer values can supply existing closed static `range` and index syntax:
literals, names, array elements, unary signs and `+`, `-`, `*`. Each static integer
use must stay within -4096..4096, while array indices remain nonnegative and in
bounds. Existing range, total iteration, helper-depth and operation limits stay
in force. This does not add variable-width allocation: `allocate` still requires
its existing literal width. Numeric gate angles convert at their real use sites.
The older pure arithmetic `def f(x: int) -> int` entry retains its quantum-input
meaning and continues to use `inputs={"x": UInt(...)}`; it is not reinterpreted
as a typed quantum program.

Python `if`/`elif`/`else` selects a classical branch at synthesis time. Admit one
finite numeric comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`), boolean literals,
and combinations with `and`, `or`, `not`. Validate all operands even beyond a
short-circuit point. Numeric truthiness, comparison chains, host calls, mutation,
quantum reads and measurement-dependent branching are refused. Quantum coherent
conditions continue to use `control`. Classical expression/condition depth and
branch nesting are bounded at 32. Branch `return`, `break`, `continue` are refused
in either arm. Only the selected arm is captured; source/AST caps include both
arms, and existing ownership, initialization and resource checks apply to the
captured program. Unselected statements are never executed or interpreted as a
promise of support. Shared classical and quantum bindings restore after helpers.

Statement dispatch moves to a private compiler module so the existing program
builder stays within its maintainability boundary. No new gate, IR opcode,
executor, dependency, public marker or source-import path is introduced.

## User example

```python
import flagquantum as fq
from flagquantum.synth import Output, QArray, QBit, X, H, allocate, qfunc, synthesize


@qfunc
def encode(message: list[int], bases: list[int], q: QArray[QBit]) -> None:
    for i in range(len(message)):
        if message[i] == 1:
            X(q[i])
    for i in range(len(bases)):
        if bases[i] == 1:
            H(q[i])


@qfunc
def main(message: list[int], bases: list[int], q: Output[QArray[QBit]]) -> None:
    allocate(8, q)
    encode(message, bases, q)


program = synthesize(main, classical_inputs={
    "message": [0, 0, 1, 1, 0, 0, 0, 1],
    "bases": [1, 1, 0, 1, 1, 1, 0, 1],
})
print(dict(program.probabilities(fq.run(program), registers=("q",))))
```

Acceptance must compare the complete original BB84 program and two distinct
nonzero datasets with independent raw complex128 Circuit states, absolute phase,
bit order and zero workspace, under both objectives. Also test coherent borrowed
inputs, branch ordering, typed root/helper binding refusals, shadowing, alias and
Output escape refusal, branch/loop budgets, and compatibility of existing real
and arithmetic entries. Replaying all 216 originals must preserve frozen source
identities, recorded outcomes and resources; approved revisions remain separate.
The BB84 quantum program does not complete its related security-protocol Notebook.
