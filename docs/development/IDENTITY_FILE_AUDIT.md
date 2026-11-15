# Identity source family

Reference is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen source files retain their original Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0242 `identity.ipynb` | `d2aa47aa42aa2c3e84237a5a7b0d5912287ca1bd` | Closed by the native tutorial, including all three examples and the symbolic leaf graph. |
| Q-0243 `identity.py` | `a78c6c7d91295613958772a8fcab1845446da417` | Native quantum register binding, tensor action and resource model verified. Multiwire foreign conversion and the source classical-control dispatch remain open. |
| Q-0244 `identity_test.py` | `23e70493c487bccb4e539befdd7ee6d0ff4a19e3` | Source assertions mapped below. External adapter and generic factory equivalence remain open. |

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
| `test_pl_interop` | The unchanged PennyLane assertion verifies external identity and its matrix. A native adapter assertion remains open. |
| `test_to_cirq_n_qubit_id` | The original Cirq bridge produces one multiwire identity. The native physical stream contains an `i` per bound wire and does not recreate that foreign atom. |
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
