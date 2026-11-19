# Component infrastructure tutorial acceptance

The native notebook maps every cell of the frozen
`qualtran/_infra/Bloqs-Tutorial.ipynb` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its Git blob is
`89419aa389e3da01c6da1bd49783ce9d4e45e746`. All 70 original cells, including
36 code cells, execute unchanged in the reference environment. The native
notebook preserves their cell kinds and records `frozen_source_cell` on every
counterpart. Two appended cells demonstrate executable application code.

Run the notebook from the repository root. Its final example uses
`circuit.append(WordSwap(5).make_on(tuple(range(5)), tuple(range(5, 10))))`
and verifies a complex state with an entangled spectator.

## Complete source mapping

| Source cells | Native owner and behavior |
| --- | --- |
| 0–13 | Core `FunctionRecord` and `Signature` declare functions without bodies. Drawer renders their ports. |
| 14–27 | `ComponentProgramBuilder` constructs the three alternating CNOT calls and their ordered returns. The shared example builds the same standalone program. |
| 28–32 | Existing linear binding checks reject aliasing, consumed handles and missing outputs. |
| 33–43 | Immutable `WordSwap` records finite wide words. A 10,000-bit declaration remains compact. Direct indexing of a word fails. |
| 44–56 | Existing array bindings and explicit split/join construct four-element and five-bit swaps. Drawer renders the connected graph and musical score. |
| 57–63 | Existing LEFT/RIGHT metadata, declared state/effect leaves and explicit tensor suppliers give the zero ket and PlusState/ZeroEffect scalar. |
| 64–69 | Existing `ModularExponentiation` fields feed a connected program containing IntState, Split, three controlled multipliers and Join. |

The example helper lives in `examples/algorithms/register_component_tutorial.py`.
Core continues to own declarations and linear validation, Compiler owns graph
construction and physical binding, Simulation owns tensor contraction, and
Drawer owns rendering. No new public exports, opcodes or serialized fields are
introduced.

## Direct dependency and numerical evidence

The declared-CNOT source has no gate implementation. Its three-call operand
chain matches the native stored calls, including the reversed middle control.
The executable extension explicitly supplies the existing native CX primitive.
It emits exactly `CX(x,y), CX(y,x), CX(x,y)` per pair. Widths 1 and 5 and the
four-element shaped case match the expected full complex swap action, including
an entangled spectator, with exact equality. Binding validates widths and rejects
aliased or negative wires. This extension does not claim that the original
signature-only CNOT can independently execute a tensor contraction.

The zero ket, projection amplitude and squared projection agree exactly with
the unchanged source. The effect is evaluated by the tensor owner as a projection.
The three deliberately invalid linear examples and opaque indexing error execute
and are caught in both notebooks. Missing component bodies remain errors at
physical lowering and binding.

For `(base, modulus, exponent width, output width) = (8,221,3,1024)`, the
connected native graph matches the executed source decomposition. The calls use
multipliers 8, 64 and 118 and exponent elements 2, 1 and 0, passing each x result
into the next call. A repeated-multiplier case `(4,15,3,4)` verifies definition
reuse without losing call occurrences. The empty-exponent case `(2,7,0,3)`
returns the initialized one word and lowers to one X. Independent Split and
IntState may swap their traversal order under the existing native scheduler;
stored order and every dependent connection are checked separately.

Existing `test_modular_exponentiation.py` and `test_rsa_phase_estimation.py`
remain the physical gate, resource and measurement-channel evidence for their
small supported instances. The tutorial's 1024-bit example constructs and draws
the one-level graph, as the source does. It does not materialize a 1024-qubit
state or expand the multiplier's physical workspace.

## Resource accounting and boundaries

| Scope | Emitted primitives | Source model |
| --- | --- | --- |
| Native executable five-bit swap extension | 15 CX, no measurement, no work bits | Original custom CNOT supplies no QEC model. |
| State/effect tutorial | Tensor contraction; no physical gate-emission claim | Original notebook makes no resource query. |
| 1024-bit ModExp tutorial | Logical connected calls only | Existing symbolic ModExp model remains separate from this graph. |

The displayed source code excerpt omits the initial base reduction while the
actual source method uses `base % mod`; the native helper follows the executed
method. Native score boundary names and truncated long labels follow the
existing renderer. The graph and score were rendered and visually inspected.

The batch regression passes 69 cases with no skips, including the reused
modular-exponentiation and phase-estimation checks. Strict typing passes for the
helper and acceptance test. Notebook cells are Black-formatted and Ruff-checked.
Architecture, scope and dependency scans are recorded with the batch. The
repository-wide dependency scan retains its existing 706 findings; none are in
these new paths. Declaration contract, machine checks and 74 IR/API cases were
verified in prerequisite commit `bdff86d7a`.
