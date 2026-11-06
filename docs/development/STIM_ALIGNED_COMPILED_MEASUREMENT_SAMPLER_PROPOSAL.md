# Prepared measurement sampling aligned with Stim

Status: approved API contract and implementation candidate.

Proposed title: `feat(circuit): compile Clifford measurement samplers`.
Primary owner: Simulation, with Integration review of the Circuit entry and
protected API. Classification: `single_device_fast_path`.

## First delivery

Add one method to the existing `Circuit`, inherited by `DynamicCircuit`:

```python
Circuit.compile_sampler(*, seed: int | None = None) -> CompiledMeasurementSampler
CompiledMeasurementSampler.sample(shots: int, *, bit_packed: bool = False) -> torch.Tensor
CompiledMeasurementSampler.sample_write(
    shots: int, *, filepath: str | os.PathLike[str], format: Literal["b8"]
) -> None
```

The handle is created by compilation and is not serializable. Its canonical
type is in `flagquantum.simulation.stabilizer.measurement_sampler`. Add no root
export, backend flag, result schema or registry. The handle owns one compiled
sampler and stream. It snapshots the circuit at preparation, returns separate
owned output buffers, and retains no completed batch. Use it from its creating
thread and process; independent workers create independent samplers. If the
SDK fails after stream execution starts, invalidate the handle because its
RNG position may have advanced. Refusal during argument or resource validation
leaves the stream untouched. Seed reproducibility is scoped to the same SDK,
machine instruction set and ordered calls. Cross-method or differently
partitioned calls do not promise equal bit streams.

`sample` returns CPU bool rows of shape `(shots, M)`, or CPU uint8 rows of
shape `(shots, ceil(M / 8))` with `bit_packed=True`. Here M counts explicit
measurement occurrences in source order. Repeated qubits and classical slots
keep distinct columns. Reset and implicit terminal readout add no columns.
Use the owned storage from Stim's fresh array through Torch without creating a
second full payload. The file operation requires explicit `format="b8"`;
positive shots and b8 are the first admitted slice of the Stim interface.
`fq.run(..., outputs=fq.samples())` keeps its existing final-readout semantics.

## Source reuse and supported program domain

Use the existing optional Stim 1.16.0 SDK for this exact first slice. The
audited upstream commit is `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`. The
installed package, source distribution, and upstream tag identities were
verified in the existing parity audit. Missing Stim raises the existing
dependency error. Other SDK versions on this new route require separate
admission; existing routes and dependency declarations remain unchanged.

Reuse the Simulation-owned dependency loader, seed and shot validation,
Clifford gate and expansion mapping, and Stim engine circuit builder. The
prepared object calls the SDK's `compile_sampler`, `sample` and
`sample_write`. Reference sampling, Pauli frames, RNG, byte packing and
streaming writes stay in the upstream implementation. Core owns the IR and
existing classical def-use validation. No SDK object enters the public return
or a Core/Compiler contract. Circuit's method delegates locally to Simulation.

Admit a bound CPU circuit with the default initial state, one circuit batch,
exact schema-defined Clifford gates without custom matrices or parameters,
single-qubit explicit Z measurements and resets, and at least one measurement.
A terminal explicit measurement is legal here and occupies a normal record
column. This slice supports one classical test on X with equals zero or one.
Refuse conjunctions of multiple bits, disjunctions, conditional reset or
measurement, unbound or non-Clifford gates, noise, custom initial states,
routed/placed artifacts, conflicting output measurement requests, and
unsupported backends or devices before preparation. No fallback or rewrites
of unsupported instructions.

The lowering tracks each slot's latest written measurement ordinal. At
feedback, the Stim lookback equals that ordinal minus the number of
measurements already emitted. Unwritten slots refuse. For equals one, emit
record-controlled X. For equals zero, emit X followed by record-controlled X.
The pair applies X exactly when the record is zero. The existing explicit
measurement instructions emit once in their original order. The mapping is
private to this path and does not alter source IR or the ordinary execution
planner. Existing DynamicCircuit construction is enough to deliver the slice.
The separate native-builder PR enables the same example with `fq.Circuit(3)`
without changing the sampler method.

## Limits and file behavior

The first release uses the following fixed, proposed bounds:

| Item | Admission |
| --- | --- |
| Source | At most 4,096 qubits, 65,536 explicit measurements and 100,000 lowered operations. Count lowered operations including Clifford expansions and inverted feedback. |
| `sample()` returned tensor | At most 64 MiB. Compute shots times M for bool or shots times ceil(M / 8) for uint8 before SDK entry. |
| `sample()` frame estimate | At most 64 MiB for the screening estimate `ceil(shots / 1024) * 1024 * (2*n_qubits + M) / 8`, rounded up to whole bytes. This estimates the principal packed frame and record tables and uses a conservative shot alignment. It is not a process RSS ceiling. |
| `sample_write()` output | At most 1 GiB, exact bytes `shots * ceil(M / 8)`. Check the destination and available space before SDK entry. The space check does not reserve disk blocks. |
| `sample_write()` buffering | No full array in Python. Make one call to the pinned SDK's b8 writer, which chooses its own batches and streaming path. At the admitted source maxima, the principal table estimate at 1,024 shots is 9 MiB. SDK scratch, reference state, allocator overhead and file caches are outside that estimate. |

No unbounded Python conversion, Python packing loop, hidden truncation or
fallback. Test oversize requests with a fake engine so no large allocation
or I/O is required. Use checked integer arithmetic and expose requested
versus permitted bytes on refusal. These checks bound requested payload and
file size; certifying an absolute resident-memory limit needs separately
reviewed measurement or containment.

Use a private staging file in the destination directory because the upstream
SDK opens its path in write/truncate mode. Publish only after the SDK returns
and the file size is exactly expected. Use an atomic no-replace publication
operation on a supported local filesystem, with a no-overwrite preflight.
Reject existing files and symlinks without opening them for writing. If
publication cannot guarantee no replacement, fail closed. Remove only staging
files created by this call on ordinary errors. No remote storage, multi-file
manifest, power-loss durability or resumable output is part of this slice.
A b8 file contains raw shot-major bytes. Column count comes from the circuit,
not a header. Pack columns little-endian within each byte and zero unused bits.

## Runnable workflow after implementation

Python 3.12, CPU PyTorch and the existing Stim extra with version 1.16.0.
Use a fresh local destination for write verification.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from flagquantum.dynamic import DynamicCircuit
import stim

circuit = (
    DynamicCircuit(3)
    .h(0)
    .measure(0, classical_bit=0)
    .conditional("x", 1, classical_bit=0)
    .reset(0)
    .conditional("x", 0, classical_bit=0, equals=0)
    .measure(0, classical_bit=0)
    .conditional("x", 2, classical_bit=0)
)
sampler = circuit.compile_sampler(seed=7)
records = sampler.sample(shots=32)
assert tuple(records.shape) == (32, 2)
assert bool((records[:, 0] ^ records[:, 1]).all())
packed = sampler.sample(shots=32, bit_packed=True)
assert tuple(packed.shape) == (32, 1)
assert set(packed[:, 0].tolist()) <= {1, 2}
with TemporaryDirectory() as directory:
    path = Path(directory) / "measurements.b8"
    sampler.sample_write(shots=32, filepath=path, format="b8")
    readback = stim.read_shot_data_file(
        path=path, format="b8", num_measurements=2
    )
    assert readback.shape == (32, 2)
    assert (readback[:, 0] ^ readback[:, 1]).all()
    assert path.read_bytes() == bytes(a + 2 * b for a, b in readback.tolist())
    print("records", tuple(records.shape))
    print("packed", tuple(packed.shape))
    print("b8 bytes", path.stat().st_size)
```

Expected stdout:

```text
records (32, 2)
packed (32, 1)
b8 bytes 32
```

An independent, handwritten Stim 1.16.0 circuit already passed upstream
preflight. Both outcomes occurred in the first and second column. The packed
bytes were 1 or 2, and Stim read back all 32 rows. Two separate deterministic
fixtures verified bytes `01` for two measurements and `ff 01` for nine
measurements, including zero padding. This is upstream evidence. The native
lowering, record budget and publication checks are pending implementation.

## Exact proposed paths

1. `flagquantum/circuit.py`
2. `flagquantum/simulation/stabilizer/measurement_sampler.py` (new)
3. `flagquantum/simulation/stabilizer/README.md`
4. `contracts/compiled-measurement-sampler-v1.json` (new)
5. `docs/development/STIM_ALIGNED_COMPILED_MEASUREMENT_SAMPLER_PROPOSAL.md` (new)
6. `docs/reference/API.md`
7. `docs/reference/KNOWN_LIMITATIONS.md`
8. `docs/guides/LOCAL_WORKFLOWS.md`
9. `tests/unit/test_compiled_measurement_sampler_contract.py` (new)
10. `tests/team/simulation/test_compiled_measurement_sampler_conformance.py` (new)
11. `tests/api_contract/test_public_docstring_examples.py`
12. `tools/qec_installed_smoke.py`
13. `tests/unit/test_qec_installed_profiles.py`
14. `examples/qec/compiled_measurement_sampler.py` (new)
15. `examples/README.md`

No changes to compiler IR, the DynamicCircuit builder or engine, existing
execution semantics, dependencies, ABI, CI or frozen contracts. If reusable
engine functionality cannot be called within this scope, stop and propose
the smallest amendment. Commit the approved contract, fake and conformance
checks first. Run path ownership before editing.

## Acceptance commands

```bash
python tools/check_team_scope.py --team integration --files <the-15-paths-above>
python -m pytest -q \
  tests/unit/test_compiled_measurement_sampler_contract.py \
  tests/team/simulation/test_compiled_measurement_sampler_conformance.py \
  tests/unit/test_qec_installed_profiles.py \
  tests/api_contract/test_public_docstring_examples.py \
  tests/test_dynamic_circuit.py
python -m examples.qec.compiled_measurement_sampler
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/check_docs_links.py
python tools/docs_source_of_truth.py --check
python tools/check_qec_installed_profiles.py --dist-dir dist --report-dir /tmp/qec-installed-reports
```

Use a core-only fake to verify one preparation, one SDK call per sample or
write, no implicit final readout, untouched destination on failure, buffer
ownership and early admission limits. With Stim 1.16.0, compare the emitted
operation and record-target order against independently written Stim text.
Use a deterministic same-slot overwrite fixture to catch column reversal
without a distribution comparison. Use an additional feedback gate after
the second write to make a stale-slot mapping change a later measured bit.
For a deterministic witness, prepare q0 in one, measure into slot zero,
reset q0, measure into slot zero again, condition X on q2, then measure q2.
The three bits must be (1, 0, 0), while a stale first-read control gives
(1, 0, 1). The exact b8 bytes are 01 and 05 respectively.
Check b8 raw bytes for two and nine columns and read both back using Stim.
Seeded comparisons use identical upstream call patterns; deterministic fixtures
supply cross-method byte equality without assuming seeded equivalence.
Verify exact-cap and one-over refusal using fakes, not a large live run.
The original dynamic and terminal sampling contracts remain green. Register
the new public docstring entry and run it in both dependency environments.
Use minimum Ruff, Black, focused strict Mypy, hygiene and language checks.
Run the user example from outside the checkout at the final PR head and
include the executable code, prerequisites and observed stdout in the PR.
No CI dispatch, hardware run or research is authorized by this proposal.

## Follow-on work

Dynamic detector conversion, prepared detector sampling and batch decoding,
noise, additional feedback forms, further output formats and general
`fq.run` measurement history each require their own admission evidence.
The embedded C++ route requires its own source and build proposal.
This PR advances bounded FTS-04 record and feedback conformance. It does not
close FTS-04 or claim full Stim parity, benchmark performance or capacity.
