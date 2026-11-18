# Basic SWAP source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0259 `swap.ipynb` | `b2ccb653c700e5a2628a6b6649218dbc68d8a118` | Closed by the complete native tutorial. |
| Q-0260 `swap.py` | `5c2481f7a1ee0a592d1192999ddf7e6dba8facc3` | Complete native capture, display, adapters and factories verified. Original nested multi-control failure remains open. |
| Q-0261 `swap_test.py` | `9f5d563a9e642e05e46bb4141438686a7b349a08` | Original tests covered by reused classical/control/factory evidence and actual atom captures, native JSON, display and external adapter assertions. |

## Signatures and execution

The atomic SWAP binds `x, y`, and atomic Fredkin binds `ctrl, x, y`.
The wider versions keep both data words through-going and exchange corresponding
bits in most-significant-bit order. Both exact operations are self-adjoint.
The native tutorial retains the two atomic signatures and the symbolic, 4-bit,
and 64-bit signatures for both register operations. It presents the symbolic
one-level graphs with weight `n` on the corresponding atomic child. The
4-bit and 64-bit signatures and graph weights are verified as metadata.

The tutorial renders the locked exact Fredkin sequence in the native circuit
drawer. A two-bit pair is appended as a register operation and checked against
the entire expected complex permutation in both control sectors. The inverse
restores identity. Existing tests cover noncontiguous operands, full
amplitudes with a spectator, and every tested branch of the outer control
measurement cleanup. Exact and relative-phase Fredkin paths retain their
separate behavior.

## Emission and resource accounting

| Path | Emitted instructions | Declared cost |
|---|---|---|
| Width `n` SWAP | `n` native SWAP instructions | `n` Cliffords, peak `2n` |
| One exact Fredkin | 8 CX, 2 H, 4 T, 3 Tdg | One Fredkin leaf. The source legacy exact synthesis reports 7 T and 10 Cliffords. Its default QEC T-equivalent policy assigns 4 to one Fredkin. |
| Width `n` exact CSwap | `n` copies of that exact Fredkin stream | `n` Fredkin leaves, peak `2n + 1` |

These paths use no scratch or measurements. The conditional wider or nested
paths have separately verified shared predicate work and feedback. The locked
source raises `TypeError` when a merged multi-bit control group is wrapped
again in the tested case. The native adapter retains an explicit refusal.

The original tests cover matrices, classical permutations, Cirq and PennyLane
conversion, bound decomposition, exact legacy costs and the small, large and
symbolic factories. Existing native controlled and classical tests provide the
gate and value evidence. Generic source serialization, foreign method
dispatch and symbolic factory equivalence remain separate from that coverage.

The original 37 cells and 39 native cells cover all four source examples.
The physical Clifford+T circuit and the register permutation checks run via
the ordinary `circuit.append(op)` path.

Both notebooks passed independent kernel execution. The native kernel rejected
Qualtran, Quimb and Cirq imports. All six SVGs and the Fredkin PNG were inspected.
The original suite recorded 38 passes and eight mode skips. The concentrated
suite passed 623 cases and the final focused suite passed all ten cases,
including the six source factory shapes and graphs. All 113 integration cases
passed. Strict typing, formatting, team scope, architecture, multilevel
Program IR and public API checks passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_swap.ipynb
```


## Complete method disposition and actual source binding

| Source definitions | Native authority and evidence |
|---|---|
| `_swap_matrix`, `_controlled_swap_matrix`, atomic tensors/signatures and adjoints | Existing native exact matrices plus actual source atom snapshots cover both control sectors, every complex column and self-adjoint identity. Atomic decomposition refusals remain in the original source. |
| TwoBitSwap as_cirq_op and both atomic as_pl_op methods | Actual named capture retains one native SWAP or the exact 17-leaf source Fredkin. Native PennyLane operation equality, complete complex matrices and bidirectional conversion cover both source assertions. |
| Four on_classical_vals methods | Reuse the 361-case original-value suite, including object preservation, ordered control comparisons, invalid controls and array ambiguity. No numerical behavior is replaced by gate simulation. |
| Swap/CSwap signature, build_composite_bloq and build_call_graph | Existing ordered bitwise binding and symbolic graphs retain source width, call multiplicity and dtype. Actual one/two-bit named captures and the complete four-bit controlled operator agree with the source. |
| get_ctrl_system methods | Existing single/mixed/negative control dispatch, layered predicate allocation and all small measurement histories retain exact operands, feedback and clean work. The original merged multi-control rewrap TypeError remains a separate source failure. |
| CSwap.make_on | The original helper infers width from x and binds ctrl/x/y. The four-bit actual helper is checked against the native exact full operator. Native emitters retain ordered logical ports and caller-owned physical bindings. |
| Explicit wire_symbol methods and CSwap circuit diagram | Drawer preserves atomic exchange boxes, word-qualified x/y labels, filled controls, complete ASCII/Unicode diagram tuples and source unknown-register behavior. Concrete integer-index widths, including SymPy Integer, work. Unresolved symbolic diagrams retain the original TypeError. |
| All original examples and doc declarations | Reuse the six small/64/symbolic factory signatures, counts and native tutorial. New actual captures cover both atomic examples. Concrete source execution is serialized in existing native snapshots, with no foreign object in the schema. |

`test_swap_atom_source_binding.py` compares all four source operations with
native exact streams and full matrices in both original directions. Logical
bindings are reversed while a final wire remains an entangled spectator. Every
complex amplitude is preserved through the actual source adjoint. These four
paths use no work or classical slots, so measurement histories are inapplicable.
The separately reused outer-control suite supplies its complete small history
and work-cleanup evidence. The four-bit CSwap full operator additionally covers
the original postselected control-zero/control-one matrix assertions.

Source and native emitted inventories agree for these captured paths: one SWAP
per word bit, or 17 exact Fredkin primitives per bit. Each Fredkin contains
eight CX, two H and seven T-family gates. Declared source QEC costs still count
Fredkin atoms, whose default T-equivalent weight is four. Existing legacy exact
T-complexity assigns seven. Relative-phase exchange remains a separate four-T
path and its prior branch evidence is reused. No cost channel is substituted
for another.

Q-0261 closes for the complete original test disposition. Original generic
harness symbolic serialization modes are explicitly skipped by the source for
SymPy assumption equality. Native concrete snapshot JSON passes; this does not
claim arbitrary foreign-object serialization. Q-0260 remains open exclusively
for the recorded original nested-control defect. The unmodified reference and
native explicit refusal are retained.

The whole-file batch passed 2,398 concentrated regression cases with no skips.
Both changed Python files passed strict typing, Black and Ruff. Compiler scope,
architecture, multilevel IR and protected API checks passed. The original test
file closes independently of the retained source nested-control failure.
