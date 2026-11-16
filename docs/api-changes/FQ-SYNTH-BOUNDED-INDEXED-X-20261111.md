# Constraint-aware indexed XOR composition

The user authorized filling the native qfunc capability gaps and completing all
216 native programs. This contract concerns unreleased Compiler internals. It
adds no public symbol, syntax, option, serialized opcode, execution backend or
Stable Core change. Release authorization remains false.

## Measured problem

Nested raw-word updates currently replace indexed carry X effects with
H-phase-H. Further coherent controls then expand the three effects separately.
Bounded predicate-X callbacks also use a phase representation. This is exact,
but can obscure the available reversible permutation and produce costly phase
decompositions. The complete native DoubleSlit model currently exceeds the
unchanged 100000-operation construction budget.

An isolated, nonproduction experiment combines direct indexed carry gates,
exact Boolean predicate-X compression, existing negative-control masks and clean
MCX ladders. All three schedules construct the original 95-cycle, 192-phase
program with 83907 intermediate instructions. Its final estimate is 80758
operations, 23 physical qubits and eight work qubits. This is compile-only
exploration, not production, statevector or native216 acceptance. Several smaller
experiments still exceed the cap; retain those diagnostics. No experiment is
permission to change source parameters or resource ceilings.

## Contract

Keep modular updates as existing indexed X effects when composing them under
controls and inverse regions. Preserve carry order, negative offsets, wraparound,
nonzero targets, ordered partial/composed storage and all ownership checks.
Zero updates still witness initialized writable storage.

A bounded predicate-X callback may use the existing exact Boolean word/phase
analysis to emit XORs selected by verified Boolean cubes or parity monomials.
Every selected permutation acts on arbitrary marker amplitudes. Preserve source
read witnesses, including canceled reads, and a checked writable marker even
when the predicate is always false. Do not discard a source check merely because
its physical unitary is identity. Keep the existing read/phase-degree reservation
for enclosing controls and retain the checked computation path outside bounds.
No example names, filenames, literal model data or parameter patterns participate
in choosing a representation.

Reuse the existing indexed control-state schema. State bit zero describes the
first control operand. Prepending a positive outer control therefore transforms
the state as `(old_state << 1) | 1`; the X target has no state bit. Preserve actual
index order, physical disjointness and branch phase in nested helpers, QPE,
adjoint and within/apply. Other unsupported effect profiles remain refusals.

Clean MCX decomposition may use only proven-zero scratch already owned by the
lowerer and must uncompute it before reuse. Reuse the existing clean-ladder and
controlled-instruction implementations. Preserve a no-extra-scratch path when
the user's width or workspace ceiling cannot afford a ladder. The complete
selected program, including locals, nested calls and arithmetic, remains subject
to the ordinary final constraint checks.

An internal decomposition hint may derive an absolute scratch ceiling from
existing user constraints. Translate that ceiling when mapping a nested unitary
layout. It must never replace the verifier's construction capacity, reinterpret
abstract temporaries as physical allocation, assume borrowed data is zero, or
suppress a capacity refusal. All source, AST, loop, call, model-node, phase,
workspace, memory, operation and candidate-completeness limits remain unchanged.
Keep complete evaluation of retain, recompute and any admitted Boolean strategy;
do not return a best-so-far result after an incomplete strategy.

## Acceptance

- Compare complete complex states and absolute phase against independent
  Circuits for predicate XOR with zero, one and coherent nonzero markers;
  signed raw predicate values; negative differences; canceled reads; partial
  and permuted unsigned borrows; nested controls; adjoint and within/apply.
- Compare modular increment/decrement and nontrivial offsets on every small
  basis word and coherent input, including surrounding bits and wraparound.
  Keep zero-update, Const, uninitialized, overlap, Output-escape and offset
  refusal tests. No signed/fixed-point view reinterpretation is added.
- Exercise mixed negative control masks through a second outer control and
  inverse composition. Verify operand/state bit order independently.
- Demonstrate both scratch-free and clean-ladder paths with real final resource
  measurements, gate/work/width refusal tests and clean scratch on coherent
  inputs. Tighter constraints must select an available exact path or fail with
  the actual checked reason; they cannot silently allocate extra work.
- Execute the complete unchanged DoubleSlit native module against its
  independent Circuit/reference, retaining all 95 cycles and 192 phases,
  complete amplitudes, absolute phase, bit order and zero work. Compilation
  alone and reduced-cycle probes are insufficient.
- Run relevant native predicate/view/callback/update/QPE/phase regressions and
  quality checks. Replay all 216 original sources and six approved revisions,
  preserving immutable earlier outcomes and resource baselines and explaining
  each actual change. Archive exact measured sources and default-tier failures.
- Finish a stable-source native216 module audit and all named scenario/refusal
  tests before integration PR or remote full CI. The existing 216 recorded
  counterparts, Notebook24/213 and revision9merged14/19 are separate counts.

## User workflow supported

The existing public workflow remains directly composable:

```python
import flagquantum as fq
from flagquantum.synth import (
    Output, QBit, QNum, RZ, X, allocate, control,
    hadamard_transform, qfunc, synthesize,
)

@qfunc
def increment(word: QNum) -> None:
    word += 1

@qfunc
def step(word: QNum, flag: QBit) -> None:
    control(word[1:3] == 2, lambda: X(flag))
    control(flag, lambda: increment(word))

@qfunc
def main(word: Output[QNum[3]], flag: Output[QBit]) -> None:
    allocate(word)
    allocate(flag)
    hadamard_transform(word)
    RZ(0.31, word[0])
    step(word, flag)

program = synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("word", "flag"))))
```

The capability and ownership semantics already exist; this work must preserve
their complete unitary while improving the internal representation and bounded
decomposition. The snippet is not a substitute for the independent references.
