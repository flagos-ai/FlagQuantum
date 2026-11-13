# Binary field inverse whole-file audit

This batch covers Q-0515, Q-0516 and Q-0517 at frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, with directly required field
addition, Frobenius, multiplication and register/workspace dependencies.

## Ownership and interfaces

Compiler owns the existing source-ordered raw inverse decomposition, exact
compute-copy-uncompute XOR wrapper, caller-owned workspace and independent
source declarations. Simulation owns field values and the original result/junk
callback. Examples compose these owners with `circuit.append(op)`. These are
single-device correctness paths using existing APIs and IR contracts.

Field words are big-endian. Raw input is THRU, result and junk words are RIGHT.
The caller provides clean result/junk/scratch storage. Only recursive scratch
returns to zero in the raw computation. Junk remains a logical output. The
clean XOR wrapper accepts an arbitrary target, retains the source and restores
all workspace to zero. Physical output maps are distinct from free logical
labels. Measurements belong to the separately accepted fixed-wire inverse
chain and its erasure owner.

The complete source inventory includes one class, its constructor and declared
field, six cached properties, four further operational methods, two factories
and one documentation specification. All six original test functions and the
complete eleven-cell notebook are audited with their direct dependencies.

## Complete definition disposition

| Original definition | Native ownership and disposition |
|---|---|
| `qgf`, `__init__` and its converter/deprecated `bitsize` path | Existing Simulation `binary_field_encoding` and neutral register metadata supply degree/polynomial construction. A foreign Bloq constructor and its deprecated keyword are outside the native interface. Generic constructor/serialization admission remains open. |
| `signature`, `bitsize`, `n_junk_regs`, `bitsize_hamming_weight` | `field_inverse_catalog` and Compiler `inverse_model` retain all port names, directions, shapes and degree-one omission. Degrees 1–16 and both original symbolic assumptions match the unchanged source. |
| `_bits`, `gf2_multiplier`, `build_composite_bloq` | Existing `field_inverse` owns the complete Itoh–Tsujii chain, recursive Karatsuba calls, logical output maps, scratch ownership and exact linear adjoints. Accepted `4e6aa871` gate/operand/output evidence is reused. |
| `build_call_graph` | Independent `inverse_source_calls` retains every concrete square power and adjoint identity, addition/product multiplicity and symbolic shape-only linear child. Complete one-level edges and leaf counts match degrees 1–16 and both symbolic instances. |
| `my_static_costs` | Neutral signature right-side bit totals match `QubitCount`, including THRU x. Static output size and actual allocated workspace are separate accounts. |
| `on_classical_vals` | Simulation `finite_field_inverse` retains scalar assertions, zero mapping, original intermediate junk values and exact NumPy empty-array dtype. Every element at degrees 1–8 and selected elements at 9/12/16 match. |
| `_gf16_inverse`, `_gf2_inverse_symbolic`, documentation specification | Concrete degree four and positive-integer symbolic m have full native signatures/declarations. The notebook's unrestricted m is also preserved, with its distinct original bit-width expression. |

The shared multiplication model previously expressed the symbolic Karatsuba
exponent as exact `log(3)/log(2)`. The frozen source evaluates numeric `log2(3)`
to a floating value before exponentiation. The shared owner now preserves that
original expression. Direct inverse QEC comparisons cover both symbolic forms.
No gates are emitted to derive declarations or symbolic costs. Concrete square
costs come from the independently owned numerical matrix and PLU cost formula.

## Whole original test disposition

| Original function | Native checks and remaining boundary |
|---|---|
| `test_gf16_inverse` | Full degree-four signature/graph, scalar/raw complex outputs, exact reversal and existing clean XOR/gate-stream checks. Generic foreign serialization remains open. |
| `test_gf2_inverse_symbolic` | Positive-integer source factory and unrestricted notebook symbol both retain full signature, junk shape, static width, graph and QEC expression. Symbolic gate emission remains outside concrete admission. |
| `test_gf2_inverse_symbolic_toffoli_complexity` | Exact original symbolic Toffoli expression and right-side static width assertions pass. Independent native models match source QEC at degrees 1–16 and both symbolic forms. |
| `test_gf2_inverse_classical_sim_quick` | Degree-one source behavior is checked unchanged. Native raw output has no junk port. The numerical callback's empty junk array is retained. |
| `test_gf2_inverse_classical_sim` | All original slow degrees 2/3/4/5 complete. Stronger scalar coverage at degrees 1–8 and existing clean execution checks are reused. |
| `test_gf2_equivalent_bloq_counts` | Original degree 1–16 declaration/decomposition count assertions complete. Native complete graphs and independently composed models match. Source linear-adjoint defects remain separately registered. |

The untouched original full test file, including slow cases, reports **27 passed,
3 skipped and 2 xfailed** in 1.28 seconds. Its three skips concern symbolic-shaped
decomposition, symbolic counts available only through the declared call graph,
and dtype checks requiring decomposition. Both xfails concern unsupported
example serialization. One warning concerns the reference's unregistered slow
marker. None of these outcomes is reported as native acceptance.

## Complete teaching asset

`examples/notebooks/binary_field_inverse.ipynb` preserves every topic of the
original eleven cells, including recurrence, parameters, registers, references,
both original examples, concrete graphical signature and full one-level graph.
It adds a coherent raw-junk execution, exact reversal and arbitrary-target clean
XOR workflow through ordinary `circuit.append(op)`.

The unchanged original eleven-cell notebook executes in an independent kernel.
The native thirteen-cell notebook has seven code cells and two SVGs. It executes
in a separate kernel blocking Qualtran, Cirq and Quimb imports. Both signature
and graph previews were visually inspected. Native names, tuple graph identities
and table layout replace foreign presentation objects. The source prose's
incorrect junk shape is explicitly distinguished from the implemented formula.

## Resource accounts

All rows below use the original default degree-eight polynomial `0x11d`.

| Account | Raw source-ordered inverse | Clean XOR inverse |
|---|---:|---:|
| Actual emitted CCX | 108 | 216 |
| Actual emitted CX | 920 | 1848 |
| Independent raw source model | 108 Toffoli, 920 Clifford | Twice raw plus 8 CX copy |
| Static original right-side bits | 40 | Separate wrapper contract |
| Caller-provided quantum workspace | 32 | 32 |
| Logical junk words retained | 3 | 0 |
| Measurements | 0 | 0 |

Raw junk is part of the output. Raw recursive scratch is zero. Clean execution
restores every workspace wire. Primitive counts use logical free output maps,
with no implicit fixed-wire binding SWAPs. Runtime samples and measurement costs
belong to their own accounts. Earlier accepted `0x11b` evidence retains its
216 CCX and 1894 CX clean count. Polynomial identity explains the different
linear gate total.

## Validation and remaining whole-file gaps

- 106 focused checks pass, including literal repaired-reference gates, all raw
  result/junk complex amplitudes with an entangled spectator at degrees 1–4,
  exact primitive reversal, and existing arbitrary-target clean XOR cleanup.
- 1852 concentrated field/resource checks and 96 independent-kernel/CPU/TN
  integration checks pass with no skips. Two existing TN warnings remain.
- Strict typing passes for all eight touched Python files. Black, Ruff, both
  ownership scopes, architecture boundaries, multilevel IR and public API
  migration checks pass.
- Five immutable source blobs match the frozen inventory. New numerical,
  declaration and example owners cover 101 statements and 34 branches at 98%
  combined coverage. Uncovered paths are an internal malformed-child guard and
  the mathematically unreachable zero-addition branch for admitted degrees.
- Accepted fixed-wire Kickmix inverse/erasure complex-branch and workspace
  evidence is reused through `test_kickmix_field_inverse`. This batch's raw
  source-chain and clean XOR execution are coherent, with zero measurements.

Q-0515 closes as a complete teaching asset. Q-0516 and Q-0517 remain open for the
inherited `SynthesizeLRCircuit` adjoint defect and full generic
Program/foreign/automatic inverse/serialization admission. The reference remains
unchanged. Native exact linear inverses match the explicitly repaired reference
at gate level. Equality of resource counts does not close unchanged-source
mathematical gate parity. Existing source defect witnesses and accepted
measurement histories remain authoritative, with no duplicate capability IDs.
