# Modular inverse comparison and conditional exchange layers

These compiler modules retain the original `_KaliskiIterationStep3` and
`_KaliskiIterationStep4` in Qualtran v0.7.0
[`mod_division.py`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py),
commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
Google copyright and Apache-2.0 source attribution are preserved. They emit IR
through shared primitives, with no runtime, device or new stable public API.
They are register layers, not a completed modular inverse.

## Comparison flags and temporary ownership

`inverse_ordering_ir` takes equal-width positive big-endian words and arbitrary
coherent logical flags branch, parity, marker and active. It XORs
`(left>right) & active & ~branch` into parity and marker. All other logical data
is preserved. Supply `n+4` clean quantum work wires and `max(n,2)` distinct
classical slots; the layer performs `n+2` measurements, then returns every
quantum work wire to zero.

The original half comparator retains an `n+1`-bit difference and one comparison
flag until its original adjoint erases them. Its two sign wires return to zero
at the forward component boundary, so the next MultiAnd uses those same two
physical wires. MultiAnd is erased before the half-comparator inverse needs
zero sign wires again. No extra two-wire pool is introduced. Source QubitCount
independently verifies this peak. The whole adjoint reverses the two flag-copy
CNOTs inside the unchanged compute/erase shell; it does not reverse primitive
measure/reset instructions.

## Exchange ownership

`inverse_exchange_ir` exchanges the two words and the two coefficients under
a coherent positive control. All four registers have the same positive width,
with disjoint physical wires. It calls shared `controlled_swap_ir` twice, using
the exact original seven-T Fredkin sequence. The original adjoint reverses the
two CSwap components before decomposition. There is no quantum work,
measurement or classical slot allocation.

## Reference component and lifetime binding

The tests read the unmodified original component graph. Whole-graph flattening
can reorder independent temporary births and reuse names across components;
those virtual names are not the physical register contract. The common
`tests/unit/register_source_graph.py` walks original CompositeBloq/Soquet
connections, shared by existing bitonic networks and the new comparison layer.

The comparison leaf adapter binds explicit original half-comparator outputs,
its internal signs, and original MultiAnd outputs. All half/MultiAnd gates are
exported from the unchanged source; CNOT is mapped as a primitive on its source
operands. The walker checks every output binding, and existing strict full
primitive dependencies, operands and measurement-feedback assertions remain
unchanged. No upstream algorithm is reimplemented in the reference or patched
to obtain agreement. Original forward/adjoint leaf decompositions and logical
ports are checked at widths 1/2/3/5/8. Old bitonic consumers are regressed after
extracting the shared reader. This test oracle is not a new production graph API.

## Channels and resources

The ordering layer checks all logical input columns through width four and named eight-bit boundaries via a
compositional certificate for all erasure histories. Widths one and two also
execute every raw measurement branch on complete complex superpositions of
both words, every logical flag and an entangled spectator. Unnormalized branch
probabilities, exact phases, zero work and original-inverse restoration are
checked. Exchange checks all inputs through width three, named eight-bit edge
samples, and complete complex/spectator states through width three in both
directions. It adds no measurement branches.

| Eight-bit layer | Measured T | Measured CX | Measured X / H | Measurements / resets | Clean work | Total qubits | Classical slots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ordering flags | 40 | 148 | 58 / 30 | 10 / 10 | 12 | 32 | 8 |
| Paired exchange | 112 | 128 | 0 / 32 | 0 / 0 | 0 | 33 | 0 |

Separately, ordering source QEC estimates 40 T / ten measurements and QubitCount
32 qubits. Exchange's source **default four-T CSwap model** estimates 64 T;
the **exact seven-T CSwap model** estimates 112 T, matching literal primitives.
Both exchange models have zero measurements and QubitCount 33. Logical and
expanded Clifford costs are not conflated, and the default model is not claimed
as a measured 64-T implementation.

Run the native user workflow, without NumPy or Qualtran:

```bash
python -m examples.algorithms.register_inverse_selection
```

`circuit.append(InverseSelection())` composes the two one-bit layers on every
coherent logical input. It checks complex amplitudes and five zero work bits,
then appends original reverse layers to restore all words and flags. The
example prints measured primitives separately from source estimates: default
CSwap 20 T, exact CSwap 26 T, three measurements and thirteen peak qubits.

Local acceptance: 46 new cases and 341 cases in the directly affected dependency
closure pass without skips. Both production modules and the shared source
reader have 100% statement coverage. This includes the original bitonic source
consumers, half comparisons, multi-control and exact swap dependencies; it is
not described as a new run of every historical G2 test. Strict mypy passes on
966 production files and separately on the example and shared reader.
Minimum/current Black, Ruff, scope, architecture and multi-level IR checks pass.
The original division, comparison, sorting and swap Git blobs match the source
inventory. The native example runs without NumPy or Qualtran.
