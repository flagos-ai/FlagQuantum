# Factorized addition components

`compiler/registers/arithmetic/addition_component.py` owns
`addition_component_program(left_width, right_width=None, name="main")`.
It returns an existing typed `ProgramRecord`, with scalar unsigned `a` and `b`
ports. Both words use MSB-first physical binding. The result is
`a, (a + b) mod 2**right_width`; positive concrete widths satisfy
`left_width <= right_width`. Symbolic and signed executable admission is unchanged.

`addition_structure.py` owns the literal source-order CNOT/AND/AND-adjoint
sequence. Both the existing forward/exact-adjoint gate emitters and the new graph
constructor consume it. There is one ripple algorithm, rather than a separate
numerical addition implementation. The existing predicate program owns each clean
carry allocation, four-T AND body, measured erasure, feedback and clean release.
Core schemas and protected APIs are unchanged.

Simulation retains tensor generation and contraction. Tests explicitly supply
native CNOT and AND tensors from the existing classical action owner. Only the
two-control primitive is enumerated, not a whole addition matrix. The AND
adjoint uses the conjugate transpose of that logical tensor; its physical measured
erasure retains branch normalization and its valid-work promise. Split/join
suppliers preserve bit connectivity without materializing full-word identities.
These are explicit numerical suppliers at the test boundary, not automatic
production provider discovery.

## Existing append workflow

The graph binds to the same operation used by existing arithmetic examples:

```python
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.dynamic import DynamicCircuit

record = addition_component_program(2)
bound = bind_register_program(record, ((0, 1), (2, 3)), clean=(4,))
circuit = DynamicCircuit(5)
circuit.append(FieldOperation(5, bound.circuit.instructions))
```

For basis inputs `a=2, b=3`, the data result is `a=2, b=1` and the carry is zero.
Tests additionally surround this operation with matching preparation/effect calls,
then run four trajectories per two-/three-bit example and restore the entire pool.

Run `python -m pytest tests/unit/test_addition_component.py -q` with the locked
reference environment. Gate comparisons cover equal and unequal widths through
32 bits. Classical slot names are alpha-renamed consistently before comparing
primitive gates, physical operands, per-wire dependencies and all feedback clauses.
The logical predicate owns fresh slots in erase order; the flat emitter uses
caller-provided carry-indexed slots. This naming difference changes no feedback.
Complete small matrices, arbitrary complex states entangled with a spectator and
every small measurement branch verify amplitudes and zero work.

## Original notebook and resource accounts

The unchanged pinned notebook `xcheck_classical_quimb.ipynb` was restored with
blob `9d20a4bf4468b6013b8c3c26263cf58f8c78854e`. All 13 code cells execute in an
isolated reference Jupyter environment, including drawings and its large Add.
The source notebook is not rewritten to make a reference pass.

Native five-bit `a=2,b=3` and 32-bit `a=5555,b=6666` closed factorized networks
contract to `1+0j`; deliberately incorrect outputs contract to zero. Native
network input tensors contain at most 32 elements. The 32-bit network has 501
nodes, a peak intermediate of 32 elements and one unsliced contraction. The source
reports contraction width 5, also 32 elements. These are tensor element counts,
not a measurement of total process memory.

The native run explicitly uses existing optional cotengra 0.8.3 greedy planning
and native PyTorch pair contraction. It does not call Quimb for native numerical
execution or fall back to a statevector. An initial built-in greedy planning
probe was interrupted after 104.82 seconds. Built-in planner speed parity is not
certified by the successful external-path execution.

| Width | Actual body primitives | Source body model | Source closed model | Physical/source peak qubits |
|---|---|---|---|---|
| 5 | H 12, T 8, T-adjoint 8, CX 45, S 4, measurement 4, conditional CZ 4, reset 4 | AND 4, Clifford 25, measurement 4 | AND 4, Clifford 29, measurement 4 | 14 / 14 |
| 32 | H 93, T 62, T-adjoint 62, CX 369, S 31, measurement 31, conditional CZ 31, reset 31 | AND 31, Clifford 214, measurement 31 | AND 31, Clifford 218, measurement 31 | 95 / 95 |

Preparation/effect X gates are additional to the body primitives. The source
model charges four abstract Cliffords for the four whole-word vectors, independent
of their popcounts. Measurements and physical work are not replaced by logical
resource estimates. The single-bit source model's negative Clifford count remains
an existing source exception.

The original notebook's native drawing replacement and automatic tensor-provider
composition remain open. Q-0897/Q-0899 are not marked fully closed solely because
the original notebook or its native numerical cases pass. G3 is not started.

Evidence prefix: `g2-addition-component-`; original notebook execution:
`g2-classical-vector-notebook-original.log` and its separately saved executed
notebook. Four source blobs match the fixed inventory. Upstream autoray warnings
remain visible in reference comparisons.


## Local validation

25 new cases, 8287 concentrated regression cases, 1453 coverage-run cases and
84 CPU/TN integration cases pass, without skips in those runs. Strict typing of
ten files, Black, Ruff, team scope, architecture, IR and protected API checks
pass. The new component graph has 100% statement/branch coverage (46 statements,
10 branches); the shared sequence has 96% (39 statements, 14 branches), with
its direct invalid-layout refusal not exercised by the coverage run.

Bounded default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings
in 157.01s`. All observed failures equal the seven-case recorded baseline;
this is not a green full-suite claim. The interrupted early greedy planning probe
is retained separately from the successful final external-path tests.
