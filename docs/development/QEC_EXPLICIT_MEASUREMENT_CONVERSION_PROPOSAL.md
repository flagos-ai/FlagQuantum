# Native dynamic measurement records to QEC events

Status: review-ready proposal; no repository implementation or build performed.
Inspected main: 75b29158db87afecd1b975bf7d8f174afe3bba3d.
Proposed title: `feat(qec): convert native dynamic measurement records with embedded Stim`.

## Product outcome and verified gap

Reuse #1102's embedded prepared measurement sampler. Add native explicit-record
conversion so a user can sample the circuit and convert its complete record
to detectors and observables without installing the Stim Python package.

#1096 supplies external text record conversion. #1100 supplies a reusable
terminal-only converter which rejects reset, measurement and feedback in its
source. #1102 emits every explicit measurement column and resolves feedback
against the latest write to each classical slot. #1105 provides mixed dynamic
SV correctness references. None of these deliveries alone supplies the proposed
native explicit-record converter.

Existing DynamicCircuit builders remain the way to write measurement,
reset and feedback. The method is defined on the existing Circuit class
and inherited by DynamicCircuit. Circuit construction convergence is a
separate API decision.

## Public API delta for approval

Add exactly this method, using the existing QEC-owned MeasurementMap and
existing compiled converter result type:

```python
def compile_m2d_converter(
    self: Circuit,
    *,
    detector_map: MeasurementMap,
    observable_map: MeasurementMap,
) -> CompiledMeasurementsToDetectionEventsConverter:
    ...
```

Retain the existing namespace function and its terminal-only signature,
behavior, refusal domain and layout handling. The new method consumes
explicit source measurements in instruction order. It appends no terminal
measurements. In this mode the existing result's measurement_qubits property
reports the source qubit for each measurement column, with repetitions where
a qubit is measured more than once. Its conversion method keeps the existing
bool-only signature and separate owned detector/observable outputs.

The new method is an approved-API prerequisite under PUBLIC_API_PROTECTION.md.
Land its reviewed candidate contract, core fake and conformance cases before
implementing the domain-owned bridge in the same PR. Do not modify unrelated
frozen API snapshots to obtain green tests.

## Runtime and architectural boundary

Circuit delegates to QEC for map validation and result composition. QEC passes
validated parity rows to Simulation. Simulation reuses the sampler's exact
lowering, record ordinals and native source bridge; backend selection is the
same wheel-owned decision as sampling. External Stim types do not cross the
Simulation boundary. The compiled converter prepares and validates its
reference once, independently of sampler RNG state.

The native method initially shares the sampler's exact unplaced CPU PyTorch,
batch-one, default-state, no-noise gate domain. It accepts explicit Z
measurement, reset and single-slot X feedback for equals 0 or 1. T and
conditional H are refused before backend loading. A classical slot may be
overwritten, while every measurement keeps its independent input column.
Use the existing declaration validator to reject reads before a slot is
defined. No dynamic-SV record-log schema is added here.

The source profile must operate without an installed Stim distribution.
The SDK profile remains an independently tested implementation of the same
new contract. Missing or broken native code in a source profile fails.
The broad existing Stim-text conversion path continues to have its existing
SDK support and rejection rules.

## Source reuse

Use exactly the previously reviewed Stim 1.16.0 PyPI archive:
SHA-256 9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52.
Upstream commit: e2fc1eca7fd21684d433aa5f10f4504ea4860d07.
The retained tag archive has SHA-256
f8b2b2e77adf91027583ef17fc44facd776694df0f45c4d8b990580bc68fb410.

The chosen static candidate reuses all 78 vendored files and adds exactly
16, for 94 total. The list and every hash are in
`conversion-and-determinism-static-candidate.json`. All selected bytes
match the two retained archives. There are 36 candidate translation units.

Six additions enable upstream measurements_to_detection_events helpers.
Ten additional files provide error analysis and its internal DEM
representation. These retain the existing rule that every declared detector
and observable parity must be deterministic in the ideal circuit. Use
the upstream error analyzer with gauge detectors disallowed before preparing
conversion. This internally needed DEM code does not expose a new DEM API.

The existing upstream conversion computes reference-normalized flips and
propagates the effect of Pauli feedback on measurement flips. A direct XOR
of raw columns with one reference row would not replace these semantics.
Reuse the native conversion helper with full measurement-major bit tables,
reference sampling and zero sweep inputs.

No upstream files are patched. Preserve LICENSE and the narrow EOF exception.
This closure was obtained by literal include, companion translation-unit,
and gate-registration inspection. A successful native link and dependency
audit remain required to establish completeness. If additional files are
required, stop with the link log and revise the inventory before execution.

## Exact proposed integration paths

1. `flagquantum/circuit.py`
2. `flagquantum/qec/execution/measurement_conversion.py`
3. `flagquantum/simulation/stabilizer/_record_conversion.py`
4. `flagquantum/simulation/stabilizer/_embedded_sampler.py`
5. `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp`
6. `flagquantum/simulation/stabilizer/csrc/stim-sampler-source-lock.json`
7. `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md`
8. `flagquantum/simulation/stabilizer/README.md`
9. `contracts/qec-explicit-measurement-converter-v1.json` (new)
10. `docs/development/QEC_EXPLICIT_MEASUREMENT_CONVERSION_PROPOSAL.md` (new)
11. `tests/unit/test_compiled_measurement_sampler_contract.py`
12. `tests/qec/test_explicit_measurement_conversion.py` (new)
13. `tests/team/simulation/test_explicit_record_conversion_conformance.py` (new)
14. `examples/qec/dynamic_measurement_conversion.py` (new)
15. `examples/README.md`
16. `docs/reference/API.md`
17. `docs/reference/KNOWN_LIMITATIONS.md`
18. `tools/verify_embedded_stabilizer_install.py`

This means 18 integration paths plus 16 new upstream paths.
The original 78 upstream files retain their bytes.
setup.py already reads the source lock and its translation-unit list;
MANIFEST.in already carries the required extensions. Neither is proposed to
change. Stop and disclose if integration requires another path.

## Supported data and budgets

First version accepts a dense CPU bool record with shape (shots, M).
The source, qubit and lowered-operation bounds from #1102 apply unchanged.
Additionally cap map rows D+O at 65,536 and total parity-column references
at 100,000 before rendering detector annotations. Each row indexes
measurement ordinals; zero observables are allowed and at least one
detector is required. Validate map width against M before loading the engine.

Cap the contiguous input payload and combined owned output payload at
64 MiB each before native entry. Account for required contiguous copies in
the estimate. Explicitly estimate the input and output packed scratch tables
with the upstream shot alignment. This describes the allocated tables,
not a hard process RSS guarantee. No output array is borrowed from a
reused native scratch buffer. Zero-shot conversion returns the exact
empty shapes after validating the declared circuit.

Packed converter input, streaming conversion, file-format conversion,
sweep controls, noise, layout/routing and unrestricted Clifford feedback
are outside this first native-record method. Existing sampler packed
sample and b8 behavior remains available.

## Proposed public example

The APIs below describe the proposed feature. Do not run as current software.
Use the source-backed macOS arm64 wheel and CPU PyTorch without the external
Stim SDK. The same example is compared against the SDK profile in a separate
installed environment.

```python
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap

circuit = (
    DynamicCircuit(3)
    .h(0)
    .measure(0, classical_bit=0)
    .conditional("x", 1, classical_bit=0)
    .reset(0)
    .conditional("x", 0, classical_bit=0, equals=0)
    .measure(0, classical_bit=0)
    .conditional("x", 2, classical_bit=0)
    .measure(1, classical_bit=1)
    .measure(2, classical_bit=2)
)
converter = circuit.compile_m2d_converter(
    detector_map=MeasurementMap(4, ((0, 1), (0, 2))),
    observable_map=MeasurementMap(4, ((1, 3),)),
)
records = circuit.compile_sampler(seed=7).sample(shots=32)
assert converter.measurement_qubits == (0, 0, 1, 2)
detectors, observables = converter.convert(
    measurements=records, separate_observables=True
)
assert detectors.shape == (32, 2) and not bool(detectors.any())
assert observables.shape == (32, 1) and not bool(observables.any())

injected = torch.tensor(
    [[True, False, False, False], [True, False, True, True]], dtype=torch.bool
)
detectors, observables = converter.convert(
    measurements=injected, separate_observables=True
)
assert detectors.tolist() == [[False, True], [False, False]]
assert observables.tolist() == [[False], [True]]
print("native QEC records: 32 shots, 2 detectors, 1 observable")
print("injected detector and observable flips agree")
```

The first detector uses a parity whose ideal value is one. The all-zero flip
output therefore explicitly tests reference normalization. The second
detector checks the first feedback target. The observable checks the second
feedback target. The injection table uses faults after the controlled source
measurement, so its row parities have an independent handwritten oracle.
Add a separate upstream comparison for a flipped control record to verify
feedback propagation instead of relying on that simple oracle.

## Upstream reference checked during preparation

The handwritten circuit passed the existing Stim 1.16.0 SDK's ideal
determinism check and converter on four fixed records: two ideal rows,
one target-measurement flip, and one observable-measurement flip.
The raw JSON is retained in `upstream-reference.stdout`. This exercises
the upstream converter only. The proposed native API and C++ binding
were not executed and no new sampling run was performed.

## Planned verification commands and gates

```sh
python -m pytest -q tests/unit/test_compiled_measurement_sampler_contract.py tests/qec/test_explicit_measurement_conversion.py tests/team/simulation/test_explicit_record_conversion_conformance.py
python -m examples.qec.dynamic_measurement_conversion
python -I tools/verify_embedded_stabilizer_install.py
```

Core fake tests must run without Stim or a compiler. Exercise both supported
condition polarities, same-slot overwrite, nonzero ideal parity, controlled
record flips with downstream Pauli feedback, zero shots, empty observables,
wrong widths and types, nonzero row limits, nondeterministic declared parity,
invalid native outputs, and missing/broken extension. Verify T and conditional
H refusal before backend loading.

Run independent upstream Stim 1.16.0 comparisons with a handwritten circuit.
Reuse #1102's exact hash-locked macOS arm64 build procedure and wheelhouses.
Build from sdist, compare all package and vendor bytes, then run the example
and verifier outside the source checkout without Stim Python distribution
metadata or an importable stim module. Record native preparation and convert
calls, source commit, extension path and digest, and dependencies. Check
sample and converter lifetime independence.

Run the existing terminal and Stim-text conversion tests unchanged in their
supported profiles. Preserve refusal, exception and serialization semantics.
Do not infer successful decoding, full Stim parity, performance or full
FTS-04/R08 completion from conversion correctness alone.

## Delivery and plan accounting

One direct-main PR, with contract/fake/conformance before implementation
commits inside that PR. Keep its final description centered on the complete
workflow and final-head installed evidence. Continue local work without waiting
for CI, as instructed. Preserve original CI failure logs and follow required
merge gates when delivery is requested. No research probe, numerical campaign,
or method-selection benchmark is authorized by preparing this proposal.

This extends the native record conversion slice of FTS-04/R08 and the R03
replacement boundary. FTS-10/12 mixed-method selection and optimized execution,
full W5 and R04 remain open. The previously merged #1105 covers dynamic SV
reference behavior and does not expand the Stim Clifford/feedback admission set.
