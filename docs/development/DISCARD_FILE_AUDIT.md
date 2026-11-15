# Discard source family

This audit covers Q-0234 `discard.py` and Q-0235 `discard_test.py` from
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. Both frozen
files retain their Google LLC 2025 Apache-2.0 headers. Their Git blobs are
`90df55c367703ec017018b5776d2f309624a8469` and
`0a1e3abc87c4e7abea02b4e1a8fe6f42943b8e02`, respectively. The source family
contains two classes and three tests.

| Source contract | Native owner and evidence |
|---|---|
| `Discard` consumes scalar `c` (`CBit`, `LEFT`). Its classical callback returns an empty mapping. Its tensor callback marks index zero attached to `c` for trace. | `compiler/registers/component_classical.py` supplies the owned classical operation. Program verification and binding enforce single consumption and preserve independent feedback snapshots. `simulation/tensor_network/trace_indices.py` implements the conjugate-index merge. |
| `DiscardQ` consumes scalar `q` (`QBit`, `LEFT`). Its tensor callback marks index zero attached to `q` for trace. Classical evaluation refuses the operation. | The same numerical index merge and `tensor_network_discard_qubit` support quantum partial trace. `compiler/registers/primitives/measurement.py` supplies separate dynamic reset lowering when execution retains the physical slot. |

The scalar ports have no bit-order choice. Native contractions retain surviving
wires in their original order and sum outer products of unnormalized outcome
kets. An explicit tensor supplier can request trace over an internal edge or
provide the identity effect for an isolated input. Automatic discovery of a
logical quantum discard from a Program-owned operation remains open. A physical
reset only verifies retirement to zero at fixed width. Those boundaries keep
Q-0234 open.

## Test disposition

| Original test | Retained or added parity evidence |
|---|---|
| `test_discard` | `test_register_discard_contract.py` checks the empty classical result for both values. `test_tensor_trace_indices.py` checks the full MeasureZ-to-Discard effect against the original tensors, including unit weight for the zero input. |
| `test_discard_vs_project` | `test_tensor_network_discard.py` prepares `PlusState` via a one-wire circuit and compares both closed scalar contractions to the unchanged source. The zero-state effect has weight 0.5 and the sum over discarded measurement outcomes has trace 1. |
| `test_discardq` | `test_tensor_trace_indices.py` compares the complete rank-four dephasing channel. `test_tensor_network_discard.py` checks its action on complex amplitudes and verifies the source classical refusal. |

Q-0235 is closed. The unmodified source suite passed all three tests. The
concentrated native unit profile passed 608 cases, with ten existing
`n_wires` deprecation warnings. It includes X and Z outcome traces, exact
complex feedback histories, source pure-state refusal, mixed gradients, and
dynamic slot cleanup. The 66 CPU and representation integration cases passed.
Strict typing, Black, Ruff, Simulation scope, architecture, the multilevel IR
contract and the public API baseline also passed.

## Resource and execution contract

| Operation | Native instruction stream | Pinned source model |
|---|---|---|
| Owned classical discard | Zero instructions, zero classical slots added, zero quantum width | Zero qubits and empty QEC counts |
| MeasureZ followed by classical discard | The measurement supplies two outcome branches. Tracing the classical result adds zero instructions. | One measurement, zero T-equivalents, one peak qubit |
| Quantum discard with a retained physical slot | One reset, zero exposed classical result bits. The slot returns to zero. | One qubit for standalone `DiscardQ`. The QEC query raises `DecomposeNotImplementedError`. |

Numerical contractions add zero primitive instructions. A fixed-width dynamic
reset gives the reduced survivor density tensored with the zeroed slot. Tests check this
for each slot in a complex three-wire input and confirm that trajectory results
expose no extra classical bit. The runnable test appends the bound operation
with the existing `circuit.append(op)` workflow:

```bash
python -m pytest -q tests/unit/test_register_discard.py tests/unit/test_tensor_network_discard.py
```
