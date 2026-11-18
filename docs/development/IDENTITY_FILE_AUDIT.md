# Identity source family

Reference is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen source files retain their original Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0242 `identity.ipynb` | `d2aa47aa42aa2c3e84237a5a7b0d5912287ca1bd` | Closed by the native tutorial, including all three examples and the symbolic leaf graph. |
| Q-0243 `identity.py` | `a78c6c7d91295613958772a8fcab1845446da417` | Closed. Complete quantum capture, grouped controls, classical dispatch, tensor/value/resource semantics, symbols and source adapter assertions are verified within the recorded physical admission boundaries. |
| Q-0244 `identity_test.py` | `23e70493c487bccb4e539befdd7ee6d0ff4a19e3` | Closed. All original scenarios have native counterparts or retained source atomic/notebook dispositions, including external adapter and whole-call capture evidence. |

## Register binding, tensor and cost

The source factories create one THRU port `q` of width 1, 4, or the unqualified
symbol `n`. Identity is atomic, is its own adjoint, and returns the input word
unchanged in classical evaluation. Its tensor is one rank-two identity per
bit, ordered by register bit index, with output then input axes. The native
tensor identity nodes and register binder preserve the same per-bit action.
The native truth table reuses the scalar domain and formatter with a direct
through action.

For quantum controls, the source builder packs ordered control groups before
`q`, applies identity to the whole word, then partitions it back into the
original groups. Native physical binding sends one `i` to each wire in that
order; a zero-width binding emits an empty stream. Activation values require
no predicate, measurement, or work wire. The source delegates controls with
classical bits to its generic control system. The concrete native wire binder
covers the quantum packing path. For a width-four identity, the stream has
four `i` instructions, whereas the declared source graph has one atomic leaf
and the QEC cost is zero. The opaque resource key preserves the symbolic
width, including the width of any packed quantum controls.

## Original test coverage

| Original function | Evidence and open boundary |
|---|---|
| `test_to_cirq` | The original OneState/Identity diagram and state check run unchanged. Native single-wire `i` is admitted by the existing Cirq bridge. |
| `test_pl_interop` | The native single-wire adapter operation, exact matrix and round trip match the unchanged source in the isolated optional environment. |
| `test_to_cirq_n_qubit_id` | The original Cirq bridge produces one multiwire identity. The complete source snapshot preserves its empty decomposed body; the native register primitive separately retains one `i` per wire. Exact matrix and counts for each representation are checked. |
| `test_identity`, `test_examples_symb` | All three factory signatures and their atomic graphs are compared to the original. Generic test-harness metadata and external adapter behavior are separate boundaries. |
| `test_unitary_vs_cirq`, `test_tensor` | Full complex append action, locked tensors and native tensor identity evidence cover widths 1 through 4. Existing tests also cover zero-width binding and entangled spectators. |
| `test_i_truth_table` | Native scalar-domain table text matches the full locked output. |
| `test_identity_controlled`, `test_identity_ctrl_adder` | Existing evidence checks symbolic `width + 1`, grouped `Partition`/identity/`Partition` order, operand mapping, complex identity action and zero cost. |
| `test_notebook` | The original tutorial runs unchanged in its own kernel. The native independent kernel runs without reference imports. |

The original test suite outside the notebook recorded 14 passes and six
atomic-mode skips. The concentrated native regression passed 432 cases and the
notebook, CPU, representation and tensor integration profile passed 108 cases.
Both native SVGs were inspected. Strict typing, formatting, team scope,
architecture, the multilevel IR contract and the public API baseline passed.

## Runnable tutorial

The native notebook retains all 12 original cell topics and adds a two-cell
append example that preserves the full complex state. The symbolic graph has
one leaf and no edges. The concrete example prints the four identity operands
in binding order alongside the independently queried source cost.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_identity.ipynb
```


## Complete Identity capture, adapters and classical dispatch

The 2026-11-18 whole-file pass reuses the existing tensors, scalar value callback,
truth table, source factories, shaped control-adder and independent notebook.
`test_identity_source_binding.py` captures widths one through four, with and
without the source's grouped QBit-array/QUInt controls. Both actual directions
are owned snapshots with zero private work, zero feedback and zero binding swaps.
Every complete complex matrix matches the source and the existing native register
identity path. JSON restoration preserves the instruction stream.

The exact physical inventories differ between named paths. A source single-wire
identity captures one `i`. The source multiwire Cirq atom decomposes to no
instructions and the snapshot preserves that empty body. The existing native
register primitive emits one `i` per bound physical wire. Each source declaration
still has one atomic call, zero QEC gates and peak logical width equal to the
whole packed word. These are separately checked and never equated as primitive
counts. No physical multiwire identity opcode is added.

The source zero-width constructor omits its q register, but `as_cirq_op` still
requires q and raises TypeError. The direct importer retains that failure.
Unbound symbolic ports fail concrete physical admission while the existing
symbolic resource path stays available.

CBit control delegates to the existing source ClassicallyControlled owner, whose
complete tensor is an identity on the classical and quantum ports. The shared
native controlled-register tensor matches it exactly for both polarities. The
existing classical instruction guard, preceded by a real measurement, matches
both full unnormalized branch vectors on a coherent input. The named quantum
register emitter refuses this classical port path. It is not promoted to a
second classical execution owner.

The native single-wire Cirq and PennyLane round trips match the actual source
operation and matrix. The PennyLane comparison runs in the existing isolated
optional environment. The new owned Identity score callback supplies the source
empty Text title and constant I TextBox. Complete one- and three-bit score
geometry matches the original atom through the existing renderer.


## Complete method disposition

| Source method | Native authority and acceptance |
|---|---|
| signature, three factories, document specification | Existing native metadata, atomic graph, symbolic model and independent notebook retain all original examples. |
| adjoint and atomic decomposition | Actual imported directional snapshots agree; source decomposition refusal remains atomic. Native register identity is self-inverse. |
| tensor and classical callback | Existing factorized identity nodes, value pass-through and truth table, plus complete imported matrices, cover the numerical semantics. |
| Cirq and PennyLane conversion | Single-wire round trips and actual foreign operation matrices match. Multiwire source capture retains the source's zero-primitive decomposition. |
| wire symbol | Native Identity callback matches complete one- and three-bit musical-score geometry. |
| quantum and classical get_ctrl_system | Existing shaped partition/identity/partition adder preserves group mapping. Whole source quantum captures and shared native classical tensor/guard cover both dispatches. |

Q-0243 and Q-0244 close. Unknown external objects remain governed by Ecosystem
admission. The source CBit-controlled Cirq conversion raises ValueError because
Cirq control values cannot represent its classical spec; that source failure is
retained and the native classical owner handles numerical execution. The source
zero-width export requires its omitted q input and fails, while the native
explicit empty register primitive remains available. Symbolic metadata/resource
queries remain separate from concrete physical capture. These boundaries are
recorded rather than promoting failed source exporters to executable support.

The final concentrated run passed all 1387 cases with no skips, using the
existing isolated PennyLane package path. The new Identity cases all executed.
Changed-surface strict typing, formatting, lint, Compiler scope, architecture,
multilevel IR and public API checks passed. Tightening the recorded classical
export failure to its exact ValueError class and message was rechecked directly.
