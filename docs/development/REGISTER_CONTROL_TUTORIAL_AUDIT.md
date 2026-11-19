# Complete register control tutorial

Q-0140 covers all 19 cells, including all nine code cells, of frozen
`qualtran/Controlled.ipynb` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`069c6045a1e3c7f971e039bf67ced9e5d3389f9f`. Copyright 2023 Google LLC,
Apache-2.0. The native notebook is `examples/notebooks/register_controls.ipynb`.
Each cell records its exact `frozen_source_cell` index. The original notebook
and its complete original test file execute unchanged.

## Interface, bit order and work ownership

`examples/algorithms/controlled_programs.py` composes existing typed component
builders and control transforms. `controlled_x_program` receives control
registers and matching activation records. `controlled_serial_program` retains
three distinct child atom identities. `controlled_each_x_program` shares one
two-bit predicate across three target X calls. Register arrays are row-major,
word bits are MSB-first and controls precede target ports. All logical ports
are THRU. The binder receives explicit clean workspace, which the predicate
compute/erase sequence restores to zero.

Core owns the typed ports. Compiler owns control selection, shared definitions,
predicate computation and measurement erasure. Simulation owns activation and
complex state execution. Drawer owns graph and register symbols. The examples
introduce no additional graph representation or public root exports.

## Every source cell

| Cell | Original content | Native realization |
| --- | --- | --- |
| 0 | Coherent branching introduction | Native register control semantics and relative phases. |
| 1 | X and default controlled X diagrams | Plain and positive controlled Programs rendered with the native graph owner. |
| 2 | Control interface and activation kinds | Typed registers and Compiler patterns with the same equality semantics. |
| 3 | QUInt(4) equality to six | Four-bit word predicate, matching activation and actual bound decomposition. |
| 4 | Bit-array activation description | Same four values and shape. |
| 5 | Bit array 0110 and score | Native array control Program and rendered graph. |
| 6 | Multiple independent register clauses | Same two groups and order. |
| 7 | Two arrays 01 and 10 | Native grouped predicate, activation and graph. |
| 8 | Controlling a composite | Stored-call propagation and retained child identities. |
| 9 | Serial three-atom composite and its control | Original three atom names and three native controlled X leaves. |
| 10 | Specialized controlled implementation | Explicit native control adder and reusable exact CX leaf. |
| 11 | Specialized CNOT class display | Exact one-CX bound stream and independent count. |
| 12 | Eight-bit control description | Source signed/unsigned discrepancy explicitly recorded. |
| 13 | Signed QInt(8) value 255 diagram | Same 255 word glyph and signed register metadata, with explicit physical encoding refusal. |
| 14 | Generic default fallback | Native stored hierarchy and explicit atomic leaf resolution. |
| 15 | Atomic default controlled wrapper | Native atom identity with independently checked source wrapper tensor. |
| 16 | Implementation ownership and protocol discussion | Current frozen tensor/export behavior and native domain owners. |
| 17 | OnEach(X,3) under two positive bits | Shared predicate, three target calls, rendered graph and actual appended DynamicCircuit execution. |
| 18 | Endomorphism and directional limitation | Native generic THRU requirement and existing specialized directional paths. |

## Exact physical and numerical evidence

`test_register_control_tutorial.py` executes all cells with imports of Qualtran,
Cirq and Kickmix blocked for the native notebook. Seven physical cases compare
with the original source graph walker. Every emitted primitive, operand and
measurement feedback agrees at the normalized primitive boundary. For the
source atomic `TestAtom`, its original controlled wrapper supplies the exact X
matrix but has no decomposition. The walker maps that independently verified
atomic action to CX and retains its distinct atom identity.

Each case starts from a full arbitrary complex vector entangled with a spectator.
All eight histories for each four-bit predicate and both histories for the
two-bit shared predicate are enumerated. Unnormalized branch amplitudes agree
with the original full controlled tensor, including measurement probability
factors, and every work wire is zero. Unit-only cases execute through
`Circuit.append(op)`. Measurement cases execute through
`DynamicCircuit.append(op)` and the existing trajectory runtime.

Signed typed controls additionally reuse `test_component_structured_control.py`.
Scalar QInt(2), shaped QInt(2) and mixed unsigned/signed clauses compare source
operations, operands, exact inverse histories, complex state, clean workspace
and resource models. The test's optional NumPy import is now guarded for the
minimal development environment.

## Observed primitives and separate models

| Case | Observed emitted primitive counts | Work / measurement histories |
| --- | --- | --- |
| Positive X | 1 CX | 0 / 1 |
| Word six, bit array, grouped arrays | 8 X, 9 H, 6 T, 6 T-dagger, 19 CX, 3 S, 3 measurements, 3 feedback CZ, 3 resets each | 3 / 8 each |
| Three serial atoms | 3 CX | 0 / 1 |
| Atomic fallback | 1 CX | 0 / 1 |
| Three target X under two controls | 3 H, 2 T, 2 T-dagger, 9 CX, 1 S, 1 measurement, 1 feedback CZ, 1 reset | 1 / 2 |

Five supported cases compare the native analytical `controlled_component_cost`
and all fields of original `QECGatesCost` independently of those observations.
The source serial and default atomic wrappers raise `DecomposeTypeError` for
QEC estimation. Their synthetic child cost of 100 T cannot supply a valid
controlled model. The native physical CX observation is kept separate.
Exact receipts are in `ftqc-reference-env/g2-control-tutorial-resources.json`.

## Source boundaries and presentation adaptations

The source cell 13 constructs signed QInt(8) with 255 while calling it unsigned.
The diagram can display that value, while the physical encoder rejects it.
Native presentation retains 255 and records the execution refusal. The generic
score renderer labels the eight-bit boundary QAny(8), while its underlying
native Register retains the signed descriptor. Graphviz pixel layout is a
native presentation choice. Graph and signed metadata images were rendered
and visually inspected.

The source prose says tensors require decomposition and no direct Cirq operation
is provided. The frozen implementation actually supplies `_tensor_data` and
`as_cirq_op`. The native prose follows that implementation. Full control-spec
transport, SDK diagrams and further descriptor admission keep Q-0158/Q-0159
open as detailed in `CONTROLLED_INTERFACE_SOURCE_AUDIT.md`.

## Acceptance

All 3344 concentrated cases pass with one existing source deprecation warning.
SDK integration passes 96 cases with two existing warnings. Strict typing passes
seven files including notebook cells. Black, Ruff, team scope, architecture and
multi-level IR checks pass. The original control test file passes 42 cases with
four existing warnings. Frozen blobs remain unchanged. Whole-repository default
and dependency findings are tracked independently of this batch.

The full dependency scan reports 688 existing findings. This batch has zero
changed-path findings after guarding the optional NumPy test dependency.
