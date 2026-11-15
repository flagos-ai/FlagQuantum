# Hadamard source family

Reference is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The original files retain the Google LLC 2023 Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0239 `hadamard.ipynb` | `60247421ec3c158535e3ff467d7c78cc661dbb7a` | Closed by `register_hadamard.ipynb`, including both signatures, the controlled score and the exact specialty circuit. |
| Q-0240 `hadamard.py` | `371672f6e39c8a5ee95e8ede95d6e3cb3c29b95c` | Native atomic H and CH and measured control are verified. Foreign adapter contracts remain open. |
| Q-0241 `hadamard_test.py` | `9af68dcaca8ef829fe4a65a6299cb4956294fb0c` | Original suite runs unchanged. Native external CH round-trip, PennyLane adapter and generic factory test coverage remain open. |

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
| `test_cirq_interop` | The unchanged source CH round trip passes. The native Cirq converter has no direct CH round trip. |
| `test_pl_interop` | The unchanged source H operation and tensor match PennyLane. A native equivalent of the external adapter assertion remains open. |
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
