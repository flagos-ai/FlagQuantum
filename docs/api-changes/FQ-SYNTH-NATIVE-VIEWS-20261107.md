# Native bit arrays, static loops and writable helpers

The user authorized the next native completion batch: fix the integration CI,
provide the existing Hadamard/Bell examples, and implement quantum arrays,
indexing, bounded static loops and writable helper borrows. This is an additive
unreleased successor; Stable Core exports and synthesize/run/read signatures do
not change. Historical admission artifacts remain unchanged. The
[candidate contract](../../contracts/synthesis-native-views-candidate.json)
records QArray/H/S and the existing operation annotations widened to accept bit
arrays. Release authorization remains false.

## Quantum semantics

`QArray[QBit]` is a distinct storage annotation, not a number or a host list.
`Output[QArray[QBit]]` declares a fresh root output; allocate(size, output) or a
supported preparation infers its 1..256-bit width. Existing QNum behavior is
preserved. Array arithmetic is rejected. Only bit arrays are admitted; nested
arrays and arrays of QNum need their own element/shape contract.

Element zero is the least-significant logical bit, matching the original Qmod
bit-array indexing. Named readout continues to pack the complete register using
Program.registers; users do not supply physical wire numbers or decode bitstrings.
Indices are checked static integers, with no negative indexing or clamping.
Nonempty half-open slices permit only step one. Nested slices retain their
ordered physical locations. Views alias storage and never allocate or copy it.
Existing gates and H/S accept scalar views; Hadamard accepts a whole array or a
view. RZZ accepts exactly two distinct bits. Compiler's existing indexed gates
check aliases, initialized storage and resources and own their lowering.

`for index in range(...)` accepts builtin range with one to three static integer
arguments. Integer literals, literal/loop-name +,-,*, unary signs and builtin
len of a known quantum view form the closed static subset. Bounds and static
integer intermediates lie in -4096..4096. One range has at most 256 iterations,
nesting at most eight and the program at most 4096 expanded iterations. Empty
ranges do no work. Loop names are scoped to their loop and cannot shadow quantum
ports, active loop names, range or len. Host functions and dynamic quantum
indices never execute. Existing 4096-node/effect and final resource caps remain
independent.

A plain `q: QArray[QBit]`, `q: QBit` or `q: QNum` helper port borrows writable storage. Every gate read requires initialization. Roots still require Output-only ports. The first mutable helper
profile allows only gate operations, static loops and other mutable helpers,
with all its ports writable. No allocation, fresh assignment, Output publication,
preparation, mixed Const/Output signatures, inverse or coherent control is admitted
inside this profile. Existing Const-only arithmetic/phase helpers retain their
separate behavior. Borrowed actuals must be disjoint physical locations; scalar
formals require one bit. Overlapping slices and read-only caller storage refuse.
Calls are acyclic, depth at most 32 and total mutable expansions at most 256.

Writable helper bodies are expanded at the call site into the existing owned
register effects. The root verifier remains authoritative for initialization,
gate semantics, existing ownership rules and budgets. There is no new IR,
unverified mutable flag, Qmod round trip or quantum execution in Python capture.

## User example

```python
import flagquantum as fq
from flagquantum.synth import CX, H, Output, QArray, QBit, allocate, qfunc, synthesize

@qfunc
def entangle(reg: QArray[QBit]) -> None:
    H(reg[0])
    for index in range(len(reg) - 1):
        CX(reg[index], reg[index + 1])

@qfunc
def main(reg: Output[QArray[QBit]]) -> None:
    allocate(6, reg)
    entangle(reg)

program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("reg",))))
```

The distribution is approximately `{(0,): 0.5, (63,): 0.5}` with positive
amplitudes and no workspace. The catalog retains the unchanged entanglement
original, whose loop is in main, and its independent Circuit comparison.

## Acceptance

Compare the exact original entanglement and classical-shadow tomography programs
against native qfunc and independent Circuit at their original sizes. Check
complete complex amplitudes without phase alignment, asymmetric indexed states,
scoped loop/helper bindings, nested slices, disjoint and overlapping borrows,
initialized targets, preparation/Output escape refusal and independent budgets.
Re-run existing native semantics, API contracts and all 216 approved Qmod sources;
retain strict original 210/216 and six original refusals as a separate count.
