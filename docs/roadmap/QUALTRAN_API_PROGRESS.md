# Qualtran API progress

The [inventory](QUALTRAN_API_INVENTORY.csv) tracks 232 entries. It contains
226 entries from the official [API Reference](https://qualtran.readthedocs.io/en/latest/reference/index.html)
and six explicitly tracked Bloqs from the algorithm roadmap.

The reference count is one row per documented symbol page under
`reference/qualtran/` in the [Sphinx inventory](https://qualtran.readthedocs.io/en/latest/objects.inv).
It excludes module pages and does not count class methods separately. The
supplement is an initial subset of the Bloqs library. This baseline is a
snapshot of the latest documentation retrieved on 2026-11-03, not a tagged
release. The inventory header does not declare a version.

Inventory SHA-256: `ce0637e400dd9281d4049524f2ac10c58cbda18b75ef364b222dca0d118632a1`

| Status | Count | Meaning |
| --- | ---: | --- |
| Unassessed | 226 | FlagQuantum mapping and evidence need review |
| Planned | 4 | Tracked in the algorithm roadmap |
| In review | 2 | Implementation has a PR |
| Verified | 0 | Merged and supported by parity or equivalent contract evidence |

Update the CSV with each PR. Record the FlagQuantum symbol and evidence link.
Mark a row verified after merge and validation against the agreed contract.
Record intentional semantic differences in the evidence field. Expand Bloq
coverage against a pinned upstream revision before reporting full-library parity.

## SelectedMajoranaFermion

[FlagQuantum #1035](https://github.com/flagos-ai/FlagQuantum/pull/1035) remains
in review. Three parity cases passed against [Qualtran 0.7.0](https://pypi.org/project/qualtran/0.7.0/)
with Cirq 1.4.0 and NumPy 1.26.4. Five local cases also passed.

The parity test uses one control, two index qubits and four targets. A seeded
complex input has amplitudes across both control branches and all four
addresses. It compares every complex output amplitude at an absolute tolerance
of 1e-12 and checks that Qualtran's decomposition returns the ancillary qubits
to zero.

FlagQuantum defines unused binary addresses as identity and verifies that
behavior locally. The upstream comparison covers the declared iteration domain.
Qualtran exposes typed selection registers and uses unary iteration.
FlagQuantum's reference path accepts flat binary bindings and materializes a matrix at
compile or execution time, capped at ten operand qubits. The stacked test
run passed all ten cases, including legacy IR and deferred lowering.

To run the optional test from a FlagQuantum development environment:

```sh
python -m pip install 'qualtran==0.7.0'
python -m pytest -q tests/unit/algorithms/primitives/multiplexers
python -m pytest -q tools/qualtran_parity
```

## SelectPauliLCU

[FlagQuantum #1037](https://github.com/flagos-ai/FlagQuantum/pull/1037) passed
three complex-amplitude comparisons against Qualtran 0.7.0 on reordered
registers with no control, control on zero, and control on one. The cases use
a signed Pauli table. The local round-trip test also verifies identity on an
unused address. The current dense lowering supports ten operand qubits.

### Gate-level Pauli LCU walk

Run the comparison with Qualtran 0.7.0, Cirq Core 1.4.0, and NumPy 1.26.4:

```sh
python -m pytest -qs tools/qualtran_parity/test_pauli_lcu_walk_qualtran.py
```

For H = X − 0.6Y + 0.4Z, both walks use three index qubits, one data
qubit and one clean work qubit. Both reproduce the target block H/2 and
eigenphases ±arccos(λ/2) for the eigenvalues λ of H within 1e-7. The work
qubit returns to zero for all clean-workspace basis inputs within 1e-12.
The Qualtran check also simulates a coherent complex input and compares
it with the linear action reconstructed from the basis outputs.

| Stage | FlagQuantum | Qualtran SELECT reference |
| --- | --- | --- |
| PREPARE | 26 operations | Same circuit |
| PREPARE inverse | 26 operations | Same circuit |
| Index reflection | 13 operations, including one Toffoli | Same circuit |
| SELECT | 30 operations, including ten Toffolis | Cirq decomposition retains no explicit Toffolis and emits one measurement, one classically controlled operation, and one reset |

The reference shares FlagQuantum's PREPARE and reflection to isolate SELECT.
Cirq decomposes the Qualtran SELECT while retaining explicit Toffolis if
present. The script prints each stage's emitted operations. These are
decomposition-level counts. Comparing T costs requires a common synthesis
policy for rotations, Toffolis, and measurement-based uncomputation.

### Native PREPARE and walk comparison

Run `python -m pytest -qs tools/qualtran_parity/test_native_pauli_lcu_walk.py`
with the same pinned reference environment. For the same signed Hamiltonian,
Qualtran's native alias PREPARE uses a two-bit address and coefficient
precision 0.17. Its native LCU block differs from H/2 by 0.04714045 in
operator norm. FlagQuantum's example uses a three-bit address to exercise
its SELECT workspace and reports a maximum element error of 2.9e-8.

With this alias PREPARE, the v0.7.0 QubitizationWalkOperator reflects on
the index alone. Its second spectral moment differs by 0.11111111 from the
moment predicted by its own encoded block. Composing the same native LCU
block with a Qualtran zero reflection on index plus alias junk gives the
predicted first three moments to 1e-10. That reference uses 16 qubits and
emits 5673 Cirq operations, including the decomposition's bookkeeping and
feedback. The FlagQuantum example reports five qubits and eleven explicit
Toffolis. Different index widths, preparation precision, and decomposition
levels make these counts descriptive rather than a common Clifford+T cost.

### SELECT table-size sweep

Run `python -m pytest -qs tools/qualtran_parity/test_select_pauli_lcu_qualtran.py`
in the pinned reference environment. The sweep reuses the existing SELECT
parity test and appends each bound operation with `Circuit.append`. For each
case, the dense executor and gate lowering agree with Qualtran on complex
amplitudes at absolute tolerance 1e-12, with clean workspace on both sides.

Uncontrolled SELECT on two data qubits with a repeated signed Pauli pattern:

| Entries | FlagQuantum work / 1q / 2q / 3q | Qualtran work / 1q / 2q | Qualtran measurements / feedback / resets |
| ---: | --- | --- | --- |
| 4 | 0 / 18 / 1 / 6 | 1 / 97 / 24 | 2 / 2 / 2 |
| 8 | 1 / 44 / 0 / 38 | 2 / 242 / 62 | 6 / 6 / 6 |
| 16 | 2 / 104 / 0 / 132 | 3 / 532 / 138 | 14 / 14 / 14 |

Work counts refer to the extra qubits in the emitted circuits. Qualtran uses
Cirq's greedy qubit reuse. FlagQuantum retains Toffolis, while Cirq lowers
the reference further. These are emitted-operation counts, so the table
does not rank Clifford+T cost. Both control polarities also pass at four
entries.

### Clifford+T counts for SELECT

The same sweep also decomposes both circuits through Cirq Core 1.4.0.
Each odd Pauli quarter turn counts as one T or T inverse after Clifford
conjugation. No cross-operation optimization is applied. Classical
feedback in these cases controls Clifford operations.

| Entries | Control | FlagQuantum T | Qualtran T | Qualtran measurements / feedback / resets |
| ---: | --- | ---: | ---: | --- |
| 4 | None | 42 | 8 | 2 / 2 / 2 |
| 4 | 0 | 133 | 12 | 3 / 3 / 3 |
| 4 | 1 | 133 | 12 | 3 / 3 / 3 |
| 8 | None | 266 | 24 | 6 / 6 / 6 |
| 16 | None | 924 | 56 | 14 / 14 / 14 |

FlagQuantum emits no measurement, feedback or reset in these cases.
The counts compare the current coherent Toffoli ladder against Qualtran's
measurement-assisted unary iteration. They do not estimate physical time,
magic-state factory throughput, or the best possible synthesis.

### FlagQuantum measurement-assisted unary iteration

`lower_select_pauli_lcu_to_unary_iteration` emits a native DynamicCircuit from
the existing structured instruction. It follows the prefix iteration and
temporary logical AND construction used in Qualtran v0.7.0. The coherent
lowering remains available for static callers. Partial tables preserve identity
on unused binary addresses.

| Entries | Control | T / T† on both sides | Measurements / feedback / resets on both sides | Emitted Clifford operations, FlagQuantum / Qualtran |
| ---: | --- | ---: | --- | --- |
| 4 | None | 8 | 2 / 2 / 2 | 35 / 113 |
| 4 | 0 | 12 | 3 / 3 / 3 | 48 / 141 |
| 4 | 1 | 12 | 3 / 3 / 3 | 46 / 139 |
| 8 | None | 24 | 6 / 6 / 6 | 94 / 280 |
| 16 | None | 56 | 14 / 14 / 14 | 212 / 614 |

The emitted Clifford counts use the native gate representation on the
FlagQuantum side and Cirq's expanded representation on the Qualtran side.
Workspace counts match in all five cases. Exact dynamic density-matrix
parity, including coherence and workspace reset, is checked through eight
entries. At sixteen entries the dynamic comparison checks resource counts,
and the coherent comparison still checks complex amplitudes.
