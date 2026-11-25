# Complete textbook phase-estimation source audit

## Frozen responsibility and ownership

This audit maps complete `phase_estimation/text_book_qpe.py` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Original files remain untouched.
`algorithms/primitives/textbook_phase_estimation.py` owns composition and metadata.
It reuses shared phase windows, packed PREPARE, block validation, named register
partitioning, the existing textbook QFT and the existing resource graph.
No compiler implementation or protected public export changes.
The execution classification is `single_device_fast_path`.

| Source responsibility | Native owner and acceptance |
| --- | --- |
| `unitary`, window and optional inverse QFT | Native signatures and explicit directional child emitters |
| Phase width and target/phase descriptors | Original name, width and side retained, including symbolic declarations |
| Default inverse QFT and width assertion | Shared textbook QFT, native ValueError for resolved width mismatch |
| Prepare, reversed phase powers, inverse transform | `make_on`, ordinary `circuit.append(op)` |
| Power specialization | Callback receives exact positive repetition count and direction |
| Declared call dictionary | `source_graph`, retaining child-key overwrite behavior |
| Concrete composed cost | `resource_model`, caller-owned child costs and shared QFT model |
| All four examples | Numeric width and three symbolic factory configurations checked |

The first phase wire is most significant. Named unitary ports retain signature
order and array flattening follows the existing partition owner. Physical
unitary ports must be through ports. The phase output starts clean. Each child
owns caller-supplied clean work and result slots. Sequential child work may be
reused after its owner restores zero. PREPARE has no logical junk. The explicit
reverse applies forward QFT, reversed inverse powers and inverse PREPARE.
The source adds no final measurement.

## Calls, complex amplitudes and replacement

At widths 1, 2, 3 and 5, source top-level controlled operations have exactly the
reversed-bit `1,2,4,...` schedule. Their operators match each emitted repeated
controlled phase without phase alignment. QFT expansion is compared at its
source CZPow, H and swap boundary, including operands and angles. The frozen
Bloq-wrapped ZPow uses `Power` for higher powers. Its Cirq wrapper does not
perform the fast-forward exponent combination initially considered during the
implementation. The test callback repeats the body and preserves that boundary.

Full output complex tensors match the untouched source on a coherent target
superposition entangled with the phase estimate, with a complex spectator.
Explicit reverse restores the initial state. A two-port CNOT example verifies
named target mapping, repeated controls, direction and full complex tensors.
Replacing the QFT emitter by the captured source CZPow-boundary stream preserves
append instructions on relocated phase wires. Escaping a child partition,
overlapping work, missing PREPARE and mismatched inverse-QFT width are refused.
IR JSON round-trip retains the complete unitary stream.

For measured Luis–Perina preparation at widths 1 and 2, every forward history
matches the source QPE tensor, including absolute complex phase and clean work.
Every independent forward/inverse history restores the initial coherent state,
spectator and all work. There are respectively 2 and 3 measurement events in
each direction, giving respectively 16 and 64 joint histories. Existing complete
LP child stream/feedback/resource evidence is reused.

## Actual primitives and independent declared model

The direct repeated controlled-phase callback emits `2^m-1` cphase gates, the
shared inverse QFT emits `m(m-1)/2` cphase and `floor(m/2)` swap, and preparation
plus QFT emits `2m` H. There are no work qubits or measurements on this direct
rectangular path. At m=3 this is 17 primitives, including 6 H, 10 cphase and
one swap. This is the actual callback implementation inventory.

Independently, the declared source model uses one PREPARE, `2^m-1`
controlled-unitary calls and one inverse-QFT call. The supplied CZPow model
retains one And, Clifford, rotation and measurement plus its three-qubit peak.
Every QEC category and peak width of the composed model matches the original
TextbookQPE at all four tested widths. This measured decomposition model is
separate from the direct cphase execution inventory.

## Original tests and remaining boundaries

The complete untouched original test file reports 10 passed, with its original
unknown slow-marker warning. This includes all six uniform-Pauli-walk cases at
2,3,4 terms and the GQSP decomposition regression. Its existing log
`g2-textbook-qpe-source.log` is reused.

Native composition accepts an explicit controlled-power emitter, including
caller-specific source specialization. The Cirq SDK's default synthesis of a
controlled Bloq can introduce opaque matrix leaves and a different primitive
stream. Full SDK-default controlled-U leaf equality remains unverified by this
batch. Native end-to-end versions of the original uniform-Pauli walk and
HamiltonianSimulationByGQSP fixtures depend on their existing typed admission
and GQSP precision gaps. They remain open under Q-0634 and the corresponding
parent records. Original source success is separate from native acceptance.
The inherited LP generic-adjoint global-minus defect remains recorded in its
own audit. The native explicit mathematical inverse is tested independently.
The TextbookQPE notebook is tracked separately under Q-0632.

Q-0633 closes complete source responsibility mapping. Q-0634 remains open for
these concrete composition boundaries. Concentrated and strict check results
are recorded below after execution.

## Batch verification

The concentrated textbook-QPE, window, LP, QFT-package, phase-gradient and GQSP
file regression reports 103 passed. Strict mypy covers both new Python files.
Black, Ruff, Algorithms scope, architecture, multi-level IR and protected API
checks pass. Evidence logs use `g2-textbook-qpe-` in the reference workspace.

## Original-parameter GQSP update, 2026-11-25

The original time-five, `1e-7`-precision Hubbard GQSP now has an actual native
binding and complete source/native child-channel comparison. A generic precision
or constructor refusal is not the current blocker. The exact degree-127 case
leaves probability `0.3289881219704` in the two outer junk bits that the original
graph frees. Native retains these bits; dropping them fails its contract.
See [the full-degree GQSP proof](HUBBARD_GQSP_FULL_DEGREE.md) for full complex
amplitude, all-history/private-work and separate resource evidence. This source
counterexample is not repaired or reclassified as clean QPE workspace. The
whole-file Q-0634 obligation remains open, with the numerical boundary now
explicit rather than an unverified small-degree extrapolation.
