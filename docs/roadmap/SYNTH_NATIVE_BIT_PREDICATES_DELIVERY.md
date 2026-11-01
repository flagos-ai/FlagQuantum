# Native one-bit logical composition

Follow #996, merged at 8a6d976580b975b5c3550a49f210719bc453dc44 after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-BIT-PREDICATES-20261101.md).

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    phase((x<2)|(y==3),pi);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=13)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Every pair has probability 1/16; full amplitudes are
(-1)**int(x<2 or y==3)/4. Only x/y are returned and hidden work is zero.
Run `python -m examples.synthesis_qmod_bit_predicates`. The user writes the
condition, preparation and constraints; no temporary flag, gate or decoding is
required. Uniform probabilities alone do not establish the sign.

## Ownership and supported semantics

Only two existing Ecosystem production files change. Parser adds &, |, ^ and
unary ~ to the existing bounded expression records. Adapter checks complete
one-bit unsigned domains and emits existing multiply/equal/add/less operations:
AND multiplies, XOR inverts equality, OR tests a positive sum and NOT tests zero.
Compiler owns gate selection, alias handling, cleanup, lifetimes and resource
selection. There is no new API, IR opcode, kernel, allocator, executor, dependency
or second role checker.

Classical 0/1 operands fold. Quantum-dependent fixed results retain their reads;
0&x and 1|x do not short-circuit initialization. No host bool conversion or
measurement is introduced. Multibit domains are refused even after x=0.
Signed complement, keyword and/or/not and in-place ^= remain unsupported.
Precedence is **, ~, *, +, &, ^, |, comparisons, highest first. Parenthesize
comparison operands. Unary recursion and tree depth retain existing limits.

The [Classiq phase documentation](https://docs.classiq.io/qmod-reference/language-reference/statements/phase/)
lists these bitwise operators for one-bit operands. Direct comparison expressions
inside phase remain the separately documented FlagQuantum-native extension;
this delivery does not claim an additional upstream original is verified.

## Execution and measured constraints

63 new cases pass: coherent value/phase truth tables and aliases under both
objectives; complex external reference entanglement; nested logic, precedence,
literal folding, multibit refusal, eager uninitialized reads, comparison-producer
replacement, unary depth boundaries and public budget-driven candidate selection.
The broad native/API regression passes 936 cases in 88.68 seconds, with all
578 statements and 260 branch exits covered across both production modules.
Four subsequent public-example constraint cases pass with the full 63-case new
suite in 2.30 seconds. Sets overlap and are not added.

CPU complex128 / Torch 2.13.0 costs for the complete example:

| Objective | Allocated / used | Hidden work | Operations | Depth |
| --- | ---: | ---: | ---: | ---: |
| Width | 13 / 11 | 9 | 99 | 69 |
| Depth | 14 / 13 | 10 | 71 | 53 |

Width counts: 4 H, 34 X, 40 CCX, 20 CX, 1 Z. Depth counts: 4 H, 22 X,
26 CCX, 18 CX, 1 Z. Under max_qubits=13, depth selection uses the width-feasible
candidate with depth 69; max_qubits=12 is refused. Allocated unused bits still
count for the memory/budget contract. These are existing selector choices, not
global Boolean simplification, optimality, performance or hardware claims.
The example (x<y)|(x==y) and existing x<=y also replace one another semantically;
the selector does not automatically rewrite all equivalent formulas.

Strict mypy 2.x passes 879 production/example sources. API snapshot, architecture
and generated documentation checks pass unchanged. The default smoke/unit run
is recorded separately in the PR; known local QFT fusion assertion failures are
not fixed by changing unrelated Simulation code or weakening tests.

Original source/license digests and the four-original cohort remain unchanged.
No general multibit bitwise arithmetic, arbitrary angles, consuming Input,
free/drop, general control/inversion or hardware admission. Independent trials
remain 0/2 and release_authorized=false.
