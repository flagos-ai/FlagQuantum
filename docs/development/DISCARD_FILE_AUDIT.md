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
provide the identity effect for an isolated input. At the original audit, binding
a logical quantum discard from a Program-owned declaration remained open. A
physical reset verified retirement to zero at fixed width. The explicit typed
provider closure below resolves that logical binding obligation.

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

## Complete typed discard binding on 2026-11-23

The complete source has two classes, two cached signatures, the classical
callback and two tensor callbacks. Both class declarations and their five
methods/properties map to existing native responsibilities. The classical signature and callback keep the
accepted `discard_word_function` and classical execution evidence. Compiler's
`component_discard.discard_qubit_function` now supplies the exact QBit LEFT `q`
signature as a bodyless `FunctionRecord`. It retains the original absence of a
classical action and physical decomposition. It uses the already admitted typed
declaration contract without adding an opcode, serialization field or root API.

Simulation's `tensor_network/discard.discard_tensor_indices` supplies the single
incoming bit label for either declaration, replacing the source `DiscardInd`
wrapper with the existing native integer protocol. The caller binds the exact
function name through `ComponentTensorProviders`. Compiler selection and mixed
Simulation supply then consume the same binding. Matching input signatures do
not imply discard semantics because projection and release can have those
signatures too. No Python source-name discovery or new registry is introduced.

The owner validates one scalar exact CBit `c` or QBit `q` LEFT port and its
single nonnegative integer input label. It returns that existing label without
allocating a numerical edge or tensor. Mixed SSA wiring, conjugate doubling,
trace merging, contraction and external input effects stay with their existing
owners. Pure graphs keep their discard/classical-port refusal. Missing providers
retain the existing explicit failure. Both original standalone mixed discard
contractions raise KeyError; native direct-input trace effects extend that
failure to the identity effect and are recorded separately from source matches.

### Complete channel and physical evidence

Both precision profiles, complex64 and complex128, verify full quantum channels
with an entangling H/S/CNOT prefix, environment phase in two placements and an
idle coherent spectator. The original gate tensors and unchanged source mixed
contraction supply the independent channel. Original Cirq gate matrices supply
every unnormalized complex hidden-outcome sector. The native explicit provider
also replaces the earlier manually supplied classical discard in a two-basis
measurement graph with classical array spectators. Complete channels retain the
original port order and agree within the recorded precision tolerance.

The existing append consumer applies H, S, CX and a physical reset to retain the
discarded slot. It matches the logical reduced density on survivor axes. Both
hidden measurement outcomes retain their complete complex amplitudes, while
the retired physical slot and an independent clean work wire are zero. Full
input-state gradients agree with an independent direct partial trace. Existing
source tests retain classical discard's empty result, quantum classical refusal,
projection-versus-discard distinction and dephasing construction.

| Account | Observed result |
|---|---|
| Numerical discard supplier | Zero new tensors, labels, quantum primitives, measurements or workspace |
| Quantum logical declaration | No stored physical decomposition or classical action, matching source absence |
| Existing retained-slot append | One H, one S, one CX and one reset, zero exposed result slots |
| Explicit hidden-outcome inspection | One measurement added only by the branch-inspection fixture, then reset |
| Original classical discard model | Zero qubits and empty QEC categories |
| Original quantum discard model | One logical qubit, QEC query raises DecomposeNotImplementedError |

The reset path and modeled source resources remain distinct. No source gate
stream is invented for an operation with no decomposition. Existing physical
reset trajectory checks and all raw measurement/parity/feedback histories are
reused from the accepted dependencies rather than duplicated.

The complete Q-0234 source and direct dependencies are accepted. The final
concentrated selection passed 694 cases, including all three unchanged original
tests and all 17 explicit-provider cases, with no skips. The affected local
integration selection passed 145 cases with 18 existing deprecation warnings.
Strict typing passed the two implementation files and their test. Black, Ruff,
both domain scopes, repository language, architecture, IR and API checks passed.
The dependency-policy scan retains exactly the prior 720 findings, with no new
findings. All 20 frozen source and dependency blobs were verified unchanged.
The initial sandbox-only notebook failure required a local Jupyter socket;
the final socket-enabled rerun passed the same full selection. These results
qualify the recorded CPU paths and retain the original missing decomposition,
classical-action refusal and standalone contraction failure described above.
