# Multi-control Pauli and multi-target CNOT source audit

The frozen Qualtran v0.7.0 reference is commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The audit verifies the five
original Git blobs before running native parity checks.

| File | Source blob | Ledger item |
| --- | --- | --- |
| `multi_control_multi_target_pauli.ipynb` | `65993bfff5d7faa7526b3840e7a4033e05324325` | Q-0555 |
| `multi_control_pauli.py` | `2953333a1cdd9408b03cdec844b5907af091d5a5` | Q-0556 |
| `multi_control_pauli_test.py` | `eed8610ca4c31125b96638a91cd90b58906fe01e` | Q-0557 |
| `multi_target_cnot.py` | `c4df15202f0ab9a76e97132560692989fe28ad90` | Q-0558 |
| `multi_target_cnot_test.py` | `5525a34f7aab590a899084f77cf30f2b60a16791` | Q-0559 |

## Multi-control Pauli

Logical ports are an ordered array of unchanged controls and a single target.
Zero controls emit the target Pauli. One control emits the selected controlled
Pauli, with the zero pattern wrapped in X gates. Larger controls keep the source
And compute/call/erase order and use one clean prefix work wire and one
classical slot per And. The measured erase clears all work in reverse order.
The added Y entry reuses the same compiler emitter as X and Z.

The native audit maps both source factories, all 78 decomposition cases, all
60 source unitary cases, and the cases in both classical X tests. The
decomposition grid compares gates, operands, per-wire order, measurement
feedback and output bindings. The full matrix grid uses the native tensor
provider. Independent append execution compares complex amplitudes across all
measurement histories for the small fixtures and two histories for coherent
sparse nine-control inputs, with work wires zero in each comparison.

| Concrete X tutorial | Emitted T/T† | Emitted measurements | Modeled And | Modeled Clifford | Modeled peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Activation 1, 0, 1, 0, 1 | 16 | 4 | 4 | 13 | 10 |

The symbolic factory preserves the source modeled T count `4*n - 4` and its
logical-width expression. Q-0556 remains open for the deprecated generic
arbitrary-child Cirq adapter and equivalent drawer wire-symbol dispatch.
Named Pauli gates have source-matched decomposition and tensor paths.

## Multi-target CNOT

The fanout signature retains a single unchanged control bit and one packed
target word. Every target can begin in an arbitrary coherent state. Fanout
uses neither work wires nor measurement. Both runtime and reference run the
same CNOT layers in the same order, with `2*n - 1` gates and depth
`2*ceil(log2(n)) + 1` on positive widths. The five-target fixture emits nine
CNOTs, has modeled peak width six, and counts as one Clifford in the source's
modern QEC profile. The legacy profile counts the emitted CNOTs.

Full complex-state checks cover all source target widths and the four
original classical widths. The existing value callback is verified on the
same truth-table rows. The symbolic fanout is a terminal call with modeled
width `n + 1`; symbolic wire diagram labels refuse the request as in the
source. The append example also rejects nonpositive target counts, for which
the source generator has no central target to index.

The [native notebook](../../examples/notebooks/multi_paulis.ipynb) retains both
tutorials, all three original examples, both signature figures and both
one-level graphs. Both execution sections use `circuit.append(op)`. The
unchanged source notebook passed, and the two unchanged non-notebook suites
reported 174 passes and four decomposition-related skips.

## Complete generic-child and wire-dispatch closure — 2026-11-21

The remaining deprecated generic-child entry is handled by the existing named
snapshot boundary, without introducing a source-branded implementation or a
second adapter. Actual `MultiControlPauli` objects with X, Y, Z, S, Rx, H,
global-shifted Z power and a nonsymmetric complex matrix child are exercised.
Empty, zero, positive and mixed three-bit control patterns retain their exact
full complex matrices, including global phase, in both original directions.
Captured IR round-trips JSON and executes with optional source SDK imports
blocked through `circuit.append(op)`. The source unitary protocol is independently
checked against the controlled child. Both directional snapshots restore the
entire coherent state, including a spectator.

This source Cirq entry selects an unmeasured controlled-unitary decomposition.
It has zero private work, feedback and binding SWAP. Its empirically counted primitive
operations are recorded independently for all 64 directional fixtures. These counts
must not be equated with the source typed graph's And/erase declarations. The
existing typed generic controlled-component owner is additionally exercised for
all eight children with a mixed three-bit activation. Its two clean prefix wires
and two result slots follow the accepted source call order. Every one of the
four measurement histories has the same complete unnormalized complex output
and both work wires return to zero. The earlier 78 literal source decomposition
checks and 60 matrix checks remain the independent gate/operand/feedback/model
proof for the complete Pauli and shared And call chain.

The new noncanonical phase fixture exposed a missing normalization step inside
snapshot recursion. The Ecosystem owner now reuses the existing eigen-phase
factorization followed by existing controlled-scalar phase conversion only when
the unchanged source decomposition cannot continue. Existing accepted source
recipes retain their primitive streams. This preserves both the canonical gate and its exact
scalar or conditioned phase. It uses the already approved parameter contracts.
It introduces no new gate, descriptor or protected operation.

Drawer `control_symbols.multi_control_wire_symbol` delegates the target to an
owned child callback and uses the existing ordered control-glyph owner. Every
control polarity and child glyph agrees with the source. Rx delegates to the
existing radian-label owner, preserving `Rx(0.3)` instead of the raw Cirq
pi-normalized label. The source's generic Z-power and matrix children retain
the original register-name fallback glyph. Symbolic control arrays preserve
the title and target but refuse individual unresolved polarity glyphs.
The normal native register-score path uses this callback with the existing
Pauli glyph supplier. As an explicit native presentation adaptation, the
connector renders the identical source title as Text with font size ten,
where the source returns a TextBox. Register grouping and execution are shared.

The source's misleading unitary claim for a reset child still raises TypeError.
A two-qubit child still raises the original IndexError through the source
unitary protocol. Native capture refuses both at its existing conversion or
signature boundaries. Arbitrary nonunitary/multi-target behavior is not claimed.
The abstract source base and specialized X/Z classes map to existing signature,
control-pattern, generic child, glyph, tensor and resource owners. Both factories,
deprecation warning, symbolic declarations, zero-control behavior, original
classical tests and complete tutorial evidence above remain applicable. This
disposes the remaining Q-0556 source-file boundary. Q-0557's accepted original
assertion mapping is unchanged.

The final focused lane passes 410 cases without skips in 7.92 seconds. Local
integration passes 17 cases. Cirq 1.6.1 passes all 76 new consumers, with 100
existing NumPy dependency deprecation warnings. Three-file strict typing, Black,
Ruff, both domain scopes, architecture, multi-level IR and API checks pass.
Two frozen source/test blobs and the locked runtime are reverified. Accepted
64-directional and eight measured-wrapper resource rows are stored separately
in `g2-multi-pauli-handoff-evidence-accepted-20261121.json` and
`g2-multi-pauli-measured-evidence-accepted-20261121.json`.

An initial eager normalization diagnostic changed accepted primitive recipes
and failed 46 checks in the broad lane. The final implementation preserves the
original decomposition first and normalizes only otherwise unsupported leaves.
The rejected diagnostic logs and intermediate resource snapshots remain
historical evidence. The accepted records named above supersede them. Initial
new-test glyph/array fixtures and import/type diagnostics were corrected before
acceptance. The frozen source was unchanged.

The final complete shared-snapshot lane passes 1,609 cases without skips in
198.00 seconds. It includes the existing controlled-Hadamard, scalar-phase,
random-Pauli SELECT, uniform preparation, rotation, LCU and finite-field
consumers whose literal streams rejected the eager diagnostic. Accepted results
are in `g2-multi-pauli-handoff-broad-accepted-20261121.log`.
