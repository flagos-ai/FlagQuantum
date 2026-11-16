# Multi-control Pauli and multi-target CNOT source audit

The frozen Qualtran v0.7.0 reference is commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The audit verifies the five
original Git blobs before running native parity checks.

| File | Source blob | Ledger item |
| --- | --- | --- |
| `multi_control_multi_target_pauli.ipynb` | `65993bfff5d7faa7526b3840e7a4033e05324325` | Q-0555 |
| `multi_control_pauli.py` | `2953333a1cdd9408b03cdec844b5907af091d5a5` | Q-0556 |
| `multi_control_pauli_test.py` | `eed8610ca4c31125b96638a91cd90b58906fe01e` | Q-0557 |
| `multi_target_cnot.py` | `c4df15202f0ab9a76e97132560692989fe28ad90` | Q-0558 |
| `multi_target_cnot_test.py` | `5525a34f7aab590a899084f77cf30f2b60a16791` | Q-0559 |

## Multi-control Pauli

Logical ports are an ordered array of unchanged controls and a single target.
Zero controls emit the target Pauli. One control emits the selected controlled
Pauli, with the zero pattern wrapped in X gates. Larger controls keep the source
And compute/call/erase order and use one clean prefix work wire and one
classical slot per And. The measured erase clears all work in reverse order.
The added Y entry reuses the same compiler emitter as X and Z.

The native audit maps both source factories, all 78 decomposition cases, all
60 source unitary cases, and the cases in both classical X tests. The
decomposition grid compares gates, operands, per-wire order, measurement
feedback and output bindings. The full matrix grid uses the native tensor
provider. Independent append execution compares complex amplitudes across all
measurement histories for the small fixtures and two histories for coherent
sparse nine-control inputs, with work wires zero in each comparison.

| Concrete X tutorial | Emitted T/T† | Emitted measurements | Modeled And | Modeled Clifford | Modeled peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Activation 1, 0, 1, 0, 1 | 16 | 4 | 4 | 13 | 10 |

The symbolic factory preserves the source modeled T count `4*n - 4` and its
logical-width expression. Q-0556 remains open for the deprecated generic
arbitrary-child Cirq adapter and equivalent drawer wire-symbol dispatch.
Named Pauli gates have source-matched decomposition and tensor paths.

## Multi-target CNOT

The fanout signature retains a single unchanged control bit and one packed
target word. Every target can begin in an arbitrary coherent state. Fanout
uses neither work wires nor measurement. Both runtime and reference run the
same CNOT layers in the same order, with `2*n - 1` gates and depth
`2*ceil(log2(n)) + 1` on positive widths. The five-target fixture emits nine
CNOTs, has modeled peak width six, and counts as one Clifford in the source's
modern QEC profile. The legacy profile counts the emitted CNOTs.

Full complex-state checks cover all source target widths and the four
original classical widths. The existing value callback is verified on the
same truth-table rows. The symbolic fanout is a terminal call with modeled
width `n + 1`; symbolic wire diagram labels refuse the request as in the
source. The append example also rejects nonpositive target counts, for which
the source generator has no central target to index.

The [native notebook](../../examples/notebooks/multi_paulis.ipynb) retains both
tutorials, all three original examples, both signature figures and both
one-level graphs. Both execution sections use `circuit.append(op)`. The
unchanged source notebook passed, and the two unchanged non-notebook suites
reported 174 passes and four decomposition-related skips.
