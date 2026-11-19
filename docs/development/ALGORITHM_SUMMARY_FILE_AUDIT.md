# Complete logical algorithm summary source mapping

The complete source is frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Source file | Frozen blob | Native owners |
| --- | --- | --- | --- |
| Q-0901 | surface_code/algorithm_summary.py | a4c71329551fee2c02c24358ff416aec8282172a | Existing resource_model.py, estimation.py and beverland.py |
| Q-0902 | surface_code/algorithm_summary_test.py | 673438bf4aa45d920493efd3d005a7e2ff57ad21 | tests/unit/test_algorithm_summary_file.py |

## Complete definition mapping

The source `AlgorithmSummary` maps to the existing concrete `ModeledResources`.
`n_algo_qubits` maps to `peak_logical_qubits` and `n_logical_gates` to `gates`.
The optional `n_rotation_layers` stays at the existing Beverland consumer's
`rotation_layers` argument with the same None default. It is independent of the
logical gate count and is not consumed by the physical aggregation. No second
summary representation or new public export is introduced.

Source `from_bloq` queries QEC categories and integer logical peak width. The
existing native `query_qec_gate_cost` and `query_resource_logical_width` own these
queries. Concrete models enter through `estimate_ftqc_resources`; that existing
entry estimates components and model-bearing operations without lowering them.
Its additional success-probability field retains its established default.

All seven original from-operation examples are compared through independently
queried native declarations. They include T, Toffoli, Fredkin, temporary And,
Z power, the complete ten-bit phase-gradient unitary and three-control X.
Rotation layer inputs None, zero, one and four agree with the unchanged source
minimum-time consumer. Existing THC end-to-end and before/after-emission evidence
is reused. These are logical models and do not replace emitted primitive counts.

## Direct dependency correction

The source Z-power example exposed a missing logical-width binding for
`RotationCall`. The Compiler-owned width authority now assigns all six rotation
opcodes their original one-qubit signature. The regression compares Rx, Ry, Rz,
X/Y/Z power source widths, graph annotations, explicit graph width overrides,
Power and OnEach. Other unsupported identities keep their original failures.
The superseded rotation-unsupported assertion is removed. The compiler module
has its own independent commit, followed by this whole-file mapping evidence.

No protected compiler API changes. Symbolic resource formulas remain at their
existing symbolic interface. Concrete integer-width admission and the explicit
rotation-depth consumer boundary remain native policy. Full complex amplitudes,
measurement histories and clean workspace belong to the previously accepted
primitive implementations and are not newly claimed by the summary mapping.

All 4668 concentrated resource and width regressions pass. The original seven
source tests pass unchanged. All three changed Python files pass strict mypy,
Black and Ruff. Compiler-team scope, architecture, IR and API checks pass.
Q-0901 and Q-0902 close as complete source/test mappings.
