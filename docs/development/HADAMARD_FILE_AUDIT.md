# Hadamard source family

Reference is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The original files retain the Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0239 `hadamard.ipynb` | `60247421ec3c158535e3ff467d7c78cc661dbb7a` | Closed by `register_hadamard.ipynb`, including both signatures, the controlled score and the exact specialty circuit. |
| Q-0240 `hadamard.py` | `371672f6e39c8a5ee95e8ede95d6e3cb3c29b95c` | Complete H/CH source disposition, measured control, capture, display and adapters, including approved static CH conversion. |
| Q-0241 `hadamard_test.py` | `9af68dcaca8ef829fe4a65a6299cb4956294fb0c` | Original suite unchanged; native assertions and approved static CH round trip covered. |

## Operations and port ownership

`Hadamard` has one scalar `q` THRU port. Its tensor takes output then input
axes and its one-control dispatch uses `CHadamard`. `CHadamard` has scalar
`ctrl` and `target` THRU ports in that order. Its tensor block is identity for
control zero and H for control one. The ordered operand tuple is control then
target in the native `ch` instruction. Both atoms are their own adjoints.
The source rejects classical basis-value evaluation for H. Native matrix
execution produces a superposition and assigns no classical callback to H.

The two factories construct the corresponding atomic gates. The native
signature and score use the shared drawer and display H on the target with
a filled control. Native positive single control emits `ch`; negative control
adds two X gates on the control. Multiple controls reuse the source-style
conjunction and measured inverse. Each extra control uses one clean work wire
and one classical feedback slot. Existing tests compare exact operands and
feedback, all small forward/inverse histories, complete complex spectator
states and work cleanup.

## Original test disposition

| Original function | Native evidence and remaining boundary |
|---|---|
| `test_to_cirq` | The original OneState-to-H Cirq circuit and state check passes. Native Cirq conformance includes H import/export. The Qualtran builder fixture remains in the original suite. |
| `test_hadamard` | The native factory example renders the signature and executes H. The generic source harness skips decomposition, block counts and decomposition-dependent QDType modes for the atom. |
| `test_unitary_vs_cirq` | Existing native H execution agrees with the locked tensor and binds to the same ordered Cirq operands. |
| `test_not_classical` | The unchanged source raises the expected exception. Native H uses the quantum path. |
| `test_chadamard_vs_cirq` | Native `ch` implements the full locked matrix. The separate specialty construction matches Cirq leaf by leaf. |
| `test_cirq_interop` | The unchanged source CH round trip passes. Proposal 122 adds the matching native static round trip with exact ordered operands and absolute phase. |
| `test_pl_interop` | The unchanged source H operation and tensor match PennyLane. Native H operation, matrix and bidirectional adapter assertions now pass. |
| `test_active_chadamard_is_hadamard` | The native exact specialty circuit restricted to the control-one block matches the entire source H matrix. |
| `test_chadamard_adjoint` | Native CH twice yields identity; measured multi-control tests also verify all pairs of forward and inverse histories. |

The locked original test suite recorded ten passes and three atomic-mode skips.
The concentrated native regression passed 131 cases. The notebook, CPU,
representation and tensor integration run passed 107 cases. The original and
native notebooks each ran all 17 cells in independent kernels. The native
kernel rejected Qualtran, Quimb and Cirq imports, and the two signature SVGs
and one score PNG were inspected.

## Instruction streams and source model

| Path | Instructions | Independent source accounting |
|---|---|---|
| H | 1 H | 1 Clifford, peak 1 |
| Positive CH | 1 CH | 4 Cliffords, 2 rotations, peak 2 |
| Negative CH | 2 X, 1 CH | The source wrapper adds 2 Cliffords to the CH model. |
| Specialty circuit | 1 phase, 3 single-wire matrix leaves, 2 CX | The source uses this Cirq construction to justify the CH cost categories. |

The specialty example specializes the frozen notebook's Cirq single-control
construction to H. It emits the determinant phase on the control and the C, B
and A matrices on the target, interleaved with two CX instructions. Tests
independently compare each operand tuple and exact leaf matrix to the original
Cirq sequence, then compare the complete circuit against atomic CH. No scratch
or measurements are required by either single-control path.

The native notebook appends the operations and verifies that the six-leaf
specialty circuit equals atomic CH, with H on the active block and exact
self-inverse behavior.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_hadamard.ipynb
```


## Whole-source named binding and display closure

`test_hadamard_source_binding.py` verifies both frozen source blobs, both
example factories, atomic decomposition refusal and exact self-adjoint identity.
The existing named emitter captures one H or one CH in both original directions,
with no private work, classical slots or binding swaps. JSON-restored native
snapshots bind reversed physical operands and execute through `circuit.append(op)`.
Full matrices and coherent states entangled with a spectator agree, and the
actual source adjoint restores every amplitude. No measurement histories apply
to these atoms. Existing measured multi-control evidence continues to cover all
small forward/inverse histories, exact feedback and clean work.

Drawer owns the native typed H and CH callbacks. Whole-atom musical-score
geometry matches the source, including the empty root title, filled ctrl circle,
H target and source unknown-register refusal for CH. H retains its constant
label for any supplied register. Foreign register objects remain in the tests
and Ecosystem boundary.

The original PennyLane H operation assertion now has a native adapter round
trip and full complex matrix check using the already installed optional
environment. The ordinary Cirq H round trip and source CH reverse conversion
also pass. Proposal 122, explicitly approved on 2026-11-25, closes the remaining
static CH adapter gap. Q-0240 and Q-0241 now have complete-file evidence.

The focused run exposed a pre-existing importer regression in positive H
front-end streams with a zero final control. Matrix-leaf recognition unwrapped
controlled scalar phases and then treated the zero-wire coefficient protocol
as an explicit matrix, bypassing the source diagonal decomposition. Excluding
scalar phases from this fallback restores the original source primitive stream.
The existing 71-case controlled-H frontend suite passes unchanged. This fix
also restores the nine-primitive zero-final-control GlobalPhase path, whose
prior one-matrix capture is corrected in its separate audit.

Emitted H and CH remain one primitive each. Independent models remain one
Clifford for H and four Cliffords plus two rotations for CH, with logical peaks
one and two respectively. The specialty six-leaf matrix construction and the
measured multi-control path retain their separately verified inventories.

The batch regression passed 1589 cases with no skips, including existing native
source-binding, controlled-H, explicit matrix and feedback import consumers.
Strict typing of all four changed Python files, Black, Ruff, both owner scopes,
architecture, multilevel IR and protected API checks passed.

## Approved static CH closure

The existing Cirq converter maps one canonical positive, binary controlled H
to one existing `ch` instruction and back. It preserves control then target
order, reversed placement, full complex operator columns, spectator phases and
exact self-inverse behavior. Appending the imported native operation does not
require the external SDK. No private wire, reset, measurement or feedback is
introduced; the source model remains separately recorded above.

The recognizer checks the original child exponent and global shift as well as
the canonical gate equality. Cirq regards H exponent minus one as equal to H,
but that noncanonical form remains outside the approved profile. Negative,
multiple and nonbinary controls, symbolic or fractional H powers, shifted H,
tags, CY and sparse static wire extents retain rejection. Existing multi-control
Compiler operations are independent of this narrow static conversion profile.

Final source-file disposition reuses all earlier source/tensor/model/display,
notebook, PennyLane and measured-control evidence. This batch adds 220 passing
regression cases on Cirq 1.7.0, 149 on 1.6.1 and 145 local integration cases.
The unchanged original suite passes ten cases with three source atomic-mode
skips. The additional 71-case controlled-H frontend fixture locks Cirq 1.7
source hashes and runs on that lane; it is not relabeled as a 1.6 comparison.
Six touched Python files pass strict typing, formatting and owner checks.
Both SDK contract verifiers, language, architecture, IR and public API checks
passed. External receipts are
`g2-static-ch-cirq1.7-20261125.log`, `g2-static-ch-cirq1.6-20261125.log`,
`g2-static-ch-original-20261125.log`, `g2-static-ch-quality-20261125.json`
and `g2-static-ch-contract16-20261125.log`.
