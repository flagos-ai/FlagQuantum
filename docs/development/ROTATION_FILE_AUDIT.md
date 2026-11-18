# Basic rotation source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0250 `rotation.ipynb` | `79016d2d85080d494b121afc47600772b4ca9953` | Closed by the native Rotation tutorial. |
| Q-0251 `rotation.py` | `69d2f19d636db2de0105919fb6b53421ba5b94b4` | Complete source method disposition verified, including actual atom and controlled captures, native presentation and adapter assertions. |
| Q-0252 `rotation_test.py` | `63480cc29919427e516c71f7453e8bdfa7cf01a3` | Original source unchanged. Native adapter, full source formatting and all eight factories covered, with symbolic model-only admission retained. |

The 50 original cells present ZPow, CZPow, Rz, CRz, XPow and YPow in that
order. The native notebook retains the six instances, the two source symbols,
six graphical signatures, four call graphs with leaf counts and two
decomposition scores. The 52 native cells add a complete append example.

## Phase, binding and resource accounting

The three Pauli powers preserve their source global phase relative to radian
rotations. A controlled application makes that phase relative to the inactive
control block. The example compares full complex matrices, explicitly checking
the Pauli-power phase against the three native radian rotations. Numerical
execution uses finite bound angles. Symbolic angles and precision remain
visible in resource declarations.

`CZPow` binds the two entries of `q` in source order and computes an AND on
one clean work wire, applies ZPow to the work wire, then uncomputes with
measurement and feedback. Existing tests compare the actual sequence against
the original for both outcomes, including full unnormalized complex states,
work reset and the inverse. The notebook checks both normalized trajectories
with one clean work wire and one classical slot.

`CRz` binds scalar `ctrl` then `q`. Its order is positive half-angle ZPow,
CX, negative half-angle ZPow, CX. Existing tests compare the source sequence
and precision halving. The example checks the complete controlled Rz matrix.

| Path | Executed stream | Source declaration |
|---|---|---|
| CRz at 0.37 radians | 2 phase, 2 CX, no work or measurement | 2 rotations, 2 Cliffords, peak 2 |
| CZPow at exponent 0.123 | 3 H, 2 T, 2 Tdg, 6 CX, 1 S, 1 phase, 1 measurement, 1 CZ, 1 reset, one work wire | 1 AND, 1 rotation, 1 measurement, 1 Clifford, peak 3 |
| XPow and YPow | A phase conjugated by basis changes, with scalar shift handled by the existing emitter when supplied | Each is an atomic leaf in the teaching graph |

The locked tests cover angle conversion, special exponents, tensor and
adjoint identities, controlled dispatch, the external Cirq and PennyLane
operations, source strings and the factory harness. Existing native tests
cover the concrete matrix relationships, primitive flow, classifications and
measurement cleanup. The native tutorial uses native labels and does not
claim external object or factory test compatibility.

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports. All ten SVGs and two score PNGs were
inspected. The locked test suite recorded 36 passes and 14 mode skips.
The 115-case direct regression and 110-case integration profile passed.
Strict typing, formatting, team scope, architecture, multilevel Program IR
and the public API baseline also passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_rotation.ipynb
```


## Complete source closure

| Source definitions | Native authority and evidence |
|---|---|
| ZPowGate, XPowGate, YPowGate signatures, cirq_gate, atomic refusal and adjoint | Native Pauli powers preserve full complex matrices and source global shift. Actual named emitters preserve the original atomic Cirq dispatch in both directions, with zero clean work or measurement slots. JSON restoration, exact comparable source streams, every matrix column and coherent spectator inputs are verified. |
| Rz, Rx, Ry signatures, cirq_gate, atomic refusal, adjoint and as_pl_op | Native radian rotations and both Cirq/PennyLane adapter round trips preserve source operation semantics and full matrices. Cirq's exponent-to-radian conversion is checked with the existing 4e-12 stream tolerance. |
| ZPowGate get_ctrl_system and Rz get_ctrl_system | Source specialized dispatch to CZPow and CRz is covered by prior source control/model evidence and the new actual named captures. Generic predicate shells retain prior mixed-polarity and measured-control evidence. |
| ZPowGate and CZPowGate __pow__ | Signed exponent multiplication and source precision are retained. Negative native repetition and actual original adjoints have their respective exact gate-order evidence. |
| CZPowGate build_composite_bloq, signature and adjoint | Native AND, phase and measured inverse agree at the source dependency and operand/feedback level. The single-call capture and existing two/three-repeat capture verify all forward/reverse histories, full unnormalized complex amplitudes and work reset. |
| CRz signature and build_composite_bloq | Exact two-phase/two-CX source stream and precision splitting are reused. Actual original adjoint is the reverse/inverse stream, verified with the existing native exact-adjoint primitive. A negated-angle reconstruction has the same operator but a different stream, so it is checked separately. |
| Six explicit wire_symbol methods and strings | Drawer supplies the empty root title and exact source power/radian labels for concrete and symbolic parameters. Full score geometry agrees. CZPow and CRz inherit the existing generic symbol behavior, with their source names retained as logical step labels. |
| Eight factories and documentation declarations | All examples are instantiated unchanged. Six concrete examples use the named emitter. Original Rz(a) retains native Parameter capture and executes after explicit binding in both directions. CRz(theta) retains its exact symbolic model and concrete capture refusal until resolved. The existing six-example tutorial and original factory-mode evidence are reused. |

The original X/Y power atoms have a numerical two-by-two unitary, while their
Cirq decomposition hook returns NotImplemented for general powers. Their named
capture previously failed. Ecosystem now recognizes these exact bare Cirq atom
types as matrix leaves when no existing native opcode applies. Canonical X/Y,
square-root X and radian rotations retain existing opcodes. Controlled wrappers
continue their original source decomposition before admission, preserving the
unchanged 71-case controlled-H frontend. This is bounded two-by-two capture of
a source atom, with no whole-algorithm matrix contraction. Foreign objects end
at the Ecosystem boundary.

For arbitrary X/Y powers, the source atom and captured native matrix each count
as one emitted leaf. The separate native basis-change implementation emits
three primitives for X and five for Y at zero scalar shift. A nonzero shift
adds its owned scalar recipe. These inventories are explicitly distinct, with
full complex operator equivalence verified. Radian rotation capture is one
primitive, CRz is four, and the measured CZPow inventory retains the previous
AND/phase/uncompute evidence. Independent source QEC classifications remain
separate from every emitted inventory.

Q-0251 and Q-0252 close with these complete method and original-test
dispositions. Symbolic executable parameters require resolution, source atom
decomposition refusals are retained, and protected static SDK admission remains
owned by the existing converter contract. No reference source or contract was
modified. The existing state-preparation rotation-binding test file was restored
unchanged after detecting a name collision with the new test. New basic-gate
cases use test_basic_rotation_source_binding.py.

The completed batch regression passed all 1888 cases with no skips, including
the restored pre-existing state-preparation tests, exact controlled-H frontend,
shared feedback import, Pauli rotation and measured controlled-Z consumers.
Strict typing of the three changed Python files, Black, Ruff, both owner scopes,
architecture, multilevel IR and public API checks passed.
