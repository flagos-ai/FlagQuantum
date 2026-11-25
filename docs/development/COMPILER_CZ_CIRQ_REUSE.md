# Cirq local cleanup in native CZ compilation

This change adapts Cirq's pending-per-qubit single-qubit merging and Pauli
rotation construction into FlagQuantum's existing verified CZ block stage.
It complements the existing Qiskit-derived two-qubit synthesis rather than
introducing an external-circuit round trip or a second compiler IR.

## Sources and ownership

The inspected upstream revisions are Cirq 1.7.0,
`de651a7886d1a1383d595f95f36b93ed3ea91685`, and Qiskit 2.6.2,
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`.

- Cirq's
  [two_qubit_to_cz.py](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/transformers/analytical_decompositions/two_qubit_to_cz.py),
  especially `_merge_single_qubit_gates`, provides the pending-per-wire traversal
  and flushing at entangling operations.
- Cirq's
  [single_qubit_decompositions.py](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/transformers/analytical_decompositions/single_qubit_decompositions.py),
  especially `single_qubit_matrix_to_pauli_rotations`, provides the Pauli-rotation
  construction, adapted to the admitted RX/RZ basis.
- The existing Qiskit-derived Weyl and supercontrolled synthesis remains the
  authoritative two-qubit constructor. Its attribution stays in
  `two_qubit_synthesis.py`.

Cirq and Qiskit use Apache-2.0. The adapted helper preserves the Cirq copyright,
license reference and modification notice. FlagQuantum already uses Apache-2.0;
this change adds no production dependency or vendor object to Core or Compiler.
Compiler owns the implementation and its numerical acceptance. The helper can
be removed or replaced without changing artifact consumers or public contracts.

## Native integration and acceptance

`_cz_blocks._candidate` calls `compact_native_cz_locals` after checking the raw
proposal's growth budget and strict CZ reduction. Local matrices accumulate
independently per wire and flush at each CZ. The helper reuses FlagQuantum's
matrix reader and Euler solver, emits at most three RX/RZ leaves per local run,
and selects a replacement only when it is strictly shorter. It deletes only
exact zero rotations; no new approximation threshold is introduced.

The enclosing stage still independently recomputes the complete two-qubit
operator and requires residual at most `1e-11` up to a recorded global phase.
The local-run helper retains CZ operations and their operand order and does
not move rotations through CZ. The subsequent
[commuting-rotation stage](COMPILER_CZ_COMMUTING_ROTATIONS.md) may combine
generated RZ rotations across CZ before the same whole-block verifier runs.
No final permutation is absorbed and no global phase is written to IR.
Annotated, custom-matrix, symbolic, trainable, batched, out-of-pair and unsupported
instructions remain unchanged. Generic fixed-pulse synthesis is unchanged.

The admitted profile is numeric generated blocks targeting arbitrary-angle RX,
RZ and full CZ, on the existing single-device CPU compilation path. This does
not expand automatic target admission, public fields, serialization, root
`fq.compile`, final layouts or support for dynamic circuits.

## Development evidence, including a rejected adaptation

The following fifty deterministic QR-generated 4x4 matrices are development
inputs, not held-out application tasks. Counts describe emitted instructions,
not depth, duration, fidelity or hardware performance.

| Probe | Existing construction | Cirq-derived proposal | Decision |
| --- | ---: | ---: | --- |
| Full-CZ interaction formulas with fixed RX pulses | 1859 | 2102 | Rejected: every case grew |
| Same interaction comparison with arbitrary RX emission on both sides | 1300 | 1350 | Rejected: every case grew |
| Pending-run cleanup after existing synthesis | 1859 | 1299 | Integrated behind existing verification |

The rejected interaction-kernel ports were exploratory temporary experiments;
their per-seed counts and residuals are retained in
[fixed-pulse evidence](evidence/cirq-cz-fixed-pulse-probe.json) and
[arbitrary-RX evidence](evidence/cirq-cz-native-rx-probe.json). They are not
independently replayable release evidence: their exploratory implementation is
not shipped. They justify investigating local cleanup rather than enabling an
inferior interaction kernel, not a conclusion about upstream compiler quality.

The implemented cleanup has a retained runnable probe and
[per-seed results](evidence/cirq-cz-local-cleanup-probe.json). All fifty cases
reduce the emitted instruction count, retain the CZ count and pass complete
4x4 operator comparison at `1e-11`. To reproduce it:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_cz_local_cleanup > /tmp/cz-cleanup.json
```

This comparison isolates generated local gates; it does not measure compilation
time or total routing cost. The sample is already inspected and must not later
be described as prospective evidence. Unchanged SDKBench and independently
generated application workloads remain necessary before comparative claims.

## User workflow and tests

Existing users continue preparing the same target-specific artifacts. The
following developer example isolates the existing native block stage:

```python
import flagquantum as fq
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks

source = fq.Circuit(n_qubits=2).swap(0, 1).cx(1, 0).to_ir()
native = resynthesize_cz_blocks(source)
assert all(gate.name in {"rx", "rz", "cz"} for gate in native.instructions)
assert sum(gate.name == "cz" for gate in native.instructions) == 2
```

This is an existing private Compiler development entry, not a new stable API.
The artifact integration tests exercise the complete affected target workflow:

```bash
python -m pytest tests/integration/test_native_block_physical_evidence.py -q
python -m pytest tests/integration/test_compiler_application_cz.py -q
```

Expected result: the existing source-to-target workflow preserves numerical
semantics, layouts and artifact replay while eligible accepted CZ rewrites use
shorter local RX/RZ sequences. Unqualified targets retain their prior path.
For isolated changes, run:

```bash
python -m pytest tests/unit/test_cz_local_cleanup.py \
  tests/unit/test_compilation_target_cz_resynthesis.py \
  tests/unit/test_native_block_optimization.py -q
```

The tests cover complete operators, reversed/nonconsecutive wires, singular and
near-singular Euler cases, protected inputs, corrupt-proposal rejection and raw
proposal resource limits before cleanup. They accompany the existing artifact
and application conformance tests; coverage does not replace those checks.


## Integration status

PR #1229 integrated this cleanup at
`f9487973b653015f4ab5842df39ade21497c74cf`. Its exact head
`ff0728bf4899651c4d61afd257e850436211b4b7` passed 239 checks, with three
optional checks skipped. The PR retains earlier runner permission failures,
dependency-download timeouts and one resource-pressure timeout, together with
the repairs and successful bounded retry. No tolerance, selected test or
timeout budget was relaxed. These CI results support integration; the
development probes above still do not establish comparative performance.
