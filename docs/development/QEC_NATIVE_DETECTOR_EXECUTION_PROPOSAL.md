# Native detector sampling and exact DEM capability proposal

Status: explicitly approved implementation contract. The original reviewed
proposal SHA-256 is 8c81cf405b61e59088d791f710fb9bdcade5b959e1a28f9cee3d3b4a437b1fe3.
Approval covers these 27 paths and publication; research execution is excluded.

## Product outcome and existing baseline

Use the existing FlagQuantum circuit builder to place physical Pauli noise and
measurements. Prepare detector sampling and derive its exact decoder model from
the same circuit and maps, then use the existing PyMatching consumer to count
logical failures. No new circuit class, root export, hardware-profile system,
Runtime output selector or public provenance object is introduced.

PR1106 supplies native explicit-record conversion. PR1087 is merged at
1004335d287c58952205829cef5a49636dbdfad2; its source/smoke/core installation passed,
while full main CI/package admission remains separate. PR1092's weighted-graph
candidate is not required by this proposal: the established probability-DEM
consumer is the boundary. Re-audit actual main and source ownership before
cutting an implementation branch. Do not stack PR branches or reuse stale greens.

Existing DynamicCircuit remains the builder for explicit intermediate records;
it is a Circuit subclass. This proposal does not claim that the base Circuit
builder has gained measure/reset/feedback construction. Unifying those builders
and fq.run planning remains a separate approved-contract task.

## Proposed public API, not implemented

Add only these two Circuit methods, inherited by DynamicCircuit:

```python
circuit.compile_detector_sampler(
    *, detector_map: MeasurementMap, observable_map: MeasurementMap,
    seed: int | None = None,
) -> CompiledDetectorSampler

circuit.detector_error_model(
    *, detector_map: MeasurementMap, observable_map: MeasurementMap,
) -> DetectorErrorModel
```

The new compiled handle lives in the QEC execution module, without a root
re-export. Its supported first method is:

```python
sampler.sample(
    shots: int, *, separate_observables: bool = False,
    append_observables: bool = False, bit_packed: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

The names follow familiar Stim usage, while records, maps, returned tensors and
DetectorErrorModel are FlagQuantum-owned. Direct handle construction and
serialization are refused. Input validation does not advance RNG. Zero shots
return exact empty shapes. A backend failure poisons the new handle instead of
silently continuing an unknown RNG state. Existing sampler/converter signatures,
refusals and serialization behavior remain unchanged.

Default sample(shots) returns detector bits, following familiar Stim usage.
separate_observables=True returns (detectors, observables); append_observables=True
returns one detector-then-observable tensor. The two flags are mutually exclusive
and are validated before RNG work. Dense output is owned shot-major CPU bool.
Packed output is owned LSB-first CPU uint8. Separate outputs are individually
padded; appended output is packed across the detector/observable boundary as one
bit sequence, not concatenated padded byte arrays. All spare bits are zero. Seed validation follows the existing prepared sampler. Reproducibility is
for the pinned implementation and identical call schedule; no cross-version,
cross-architecture or batch-size-invariance claim is made.

## Proposed end-to-end user code

This example is proposed software, not an executed result:

```python
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec import PyMatchingDecoder
from flagquantum.qec.execution.context import MeasurementMap

circuit = (
    DynamicCircuit(3)
    .x(0).x(1).x(2)
    .bit_flip(0, 0.1).bit_flip(1, 0.1).bit_flip(2, 0.1)
    .measure(0, classical_bit=0)
    .measure(1, classical_bit=1)
    .measure(2, classical_bit=2)
)
detector_map = MeasurementMap(3, ((0, 1), (1, 2)))
observable_map = MeasurementMap(3, ((0,),))
sampler = circuit.compile_detector_sampler(
    detector_map=detector_map, observable_map=observable_map, seed=7,
)
model = circuit.detector_error_model(
    detector_map=detector_map, observable_map=observable_map,
)
decoder = PyMatchingDecoder.from_detector_error_model(model)
failures = 0
for batch_size in (16, 16, 3):
    detectors, observables = sampler.sample(
        batch_size, separate_observables=True,
    )
    predicted = decoder.decode_batch(detectors)
    failures += int((predicted ^ observables).any(dim=1).sum())
print("shots", 35)
print("logical_failures", failures)
print("logical_error_rate", failures / 35)
```

This is a three-data-qubit code-capacity illustration, not multi-round fault
tolerance or hardware feedback. The logical-one preparation makes raw readout
one distinct from a logical flip. Expected invariants: model has two detectors
and one observable; each batch has (S,2)/(S,1) output shapes; all 35 shots are
retained and 0 <= failures <= 35. Do not invent a golden stochastic failure count.
Capture real final-head stdout during approved implementation. Existing optional
PyMatching is still required for decoding; this is not native PyMatching source
integration. The embedded sampler/DEM producer must work without Stim Python.

## Source reuse and ownership

Reuse exactly the current 96 unchanged Stim 1.16.0 files and 37 translation units.
Upstream commit e2fc1eca7fd21684d433aa5f10f4504ea4860d07; PyPI source SHA-256
9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52.
The retained tag SHA-256 is
f8b2b2e77adf91027583ef17fc44facd776694df0f45c4d8b990580bc68fb410.

The current vendored frame_simulator_util.h declares
sample_batch_detection_events. The existing error_analyzer.h and compiled source
already provide circuit_to_detector_error_model. Reuse their implementations
through the private limited-API C++ bridge, rather than importing or copying
upstream Python bindings, NumPy or pybind11 into the embedded build.

The sampling helper recreates FrameSimulator storage on each draw. Compilation
therefore promises reused lowering, parsed circuit and RNG, not persistent
scratch amortization or a measured speed benefit. Do not advertise performance.
A successful link, no-SDK installation and conformance still have to prove the
closure. Stop before adding any further upstream source or build/dependency path.

Circuit contains thin lazy entry methods. QEC owns map semantics, public handle
validation and conversion to the established DetectorErrorModel. Simulation owns
Clifford/noise admission and numerical calls. SDK objects stop at Simulation.
Contract, fake and conformance tests must precede implementation commits in the
same approved direct-main capability PR.

## Admission and exactness

One default-initial-state CPU PyTorch source circuit, batch size one, explicit
Z records in source order. Preserve repeated classical-slot records and latest
slot feedback using the existing lowerer. Fixed named Clifford gates, resets
and single-slot X feedback for equals 0/1 retain the existing subset. T,
conditional H, multiple-control feedback, custom matrices, unbound/trainable
parameters, placed/routed circuits, GPU and amplitude/thermal decay fail before
engine preparation. No SV fallback or approximation.

For new methods only, accept inline one-/two-qubit stochastic Pauli channels via
the existing _frame_weights/_frame_operations classifier and unchanged numerical
tolerance. Existing compile_sampler and record-conversion noise refusals remain.
Do not replace the authoritative classifier or copy a second channel taxonomy.
Measurements with readout-error parameters and classical record corruption are
not introduced by placing a quantum bit-flip channel before a measurement.
General NoiseModel attachment, multi-qubit correlated chains beyond arity two,
MemoryCircuit backend replacement, detector sample_write and file-format expansion are explicit later
gaps, not claims of this first native source route.

Sampling and DEM admission are distinct. A Pauli channel that can be sampled
need not have a supported exact independent-mechanism DEM. Analyze with gauge
acceptance disabled, decompose_errors=False, approximation threshold zero and
ignore_decomposition_failures=False. Refuse unsupported exact analysis; never
silently split a correlated mechanism, approximate disjoint alternatives or clip
probabilities. Use DetectorErrorModel.from_stim_text with its existing default
use_decomp_suggestions=False. Mirror the existing exact SDK helper's missing
shape-declaration normalization using actual circuit/DEM dimensions, including
zero-error and unused-observable cases. Do not introduce another text parser.

## Proposed finite bounds, internal constants

Retain current 4096 source-qubit, 65536 measurement, 100000 source/lowered-op,
65536 total map-row and 100000 parity-reference limits. Keep input validation
before import/preparation. Each sampled output payload and conservative padded
frame/table estimate is capped at the existing 64 MiB boundary before drawing.
No clipping, renormalization, threshold relaxation or reduced-repeat study.

Before DEM analysis additionally bound both (n+M)*(D+O) and
sum(4**channel_arity-1)*(D+O) at 1000000, and conservatively bound encoded DEM
text at 16 MiB, with an output-size guard before returning Python data. These are
proposed new-route resource guards, not changes to existing APIs or process RSS
guarantees. Arrays, analysis work and allocator/RSS are different quantities.
No new live RSS monitor/probe is part of this implementation authorization.

## Acceptance without a scientific campaign

1. Core fakes: signatures, map/source validation, no backend work on refusal,
   ownership, padding, zero shots, batch totals, RNG non-advance on validation,
   failure poisoning and missing/broken extension without silent SDK fallback.
2. Preserve current sampler/converter refusal tests and the historical exports,
   class-pickle, registry, fixtures, thresholds and old CI logs.
3. Independently handwrite the three-qubit Stim reference. Check exact DEM fault
   signatures and probabilities, including declared dimensions. Enumerate all
   eight forced X-fault patterns with 0/1 channels against a fixed nominal-p
   decoder. Its exact independent oracle is 3*p*p - 2*p*p*p (0.028 at p=0.1),
   not the exact failure rate of a finite sampled run.
4. Check repeated-slot conditional-X programs against SDK reference; retain the
   nonzero ideal-parity tests. T and conditional H must refuse before loading.
5. Exercise accepted and refused Pauli-analysis cases separately. Test a channel
   that sampling supports but exact DEM analysis refuses. No swallowed refusal.
6. Check default/separate/appended dense and packed results, conflicting flags,
   zero observables, a 7-detector/2-observable byte boundary and 16/16/3 chunks. Do not
   require separate tied decoder calls to return identical solutions.
7. Hash-locked offline macOS arm64 sdist/wheel/core acceptance. Without an installed
   or importable Stim package, instrument actual native sample/DEM calls and log
   extension path/hash/dependencies. Separately exercise the SDK implementation
   and the existing four installed profiles. Record actual final-head user code.
8. Run actual CI tools as well as supported minimums: Ruff0.15/0.16.10,
   Black24.1/26.10.1 (same config, AST safety) and Mypy2.0/2.4.1 strict Python3.12.
   These version pairs are observed tools, not a universal future-version claim.
9. Register the new public module's examples in the actual doctest ENTRIES from
   the start. Update authoritative capability/parity sources and regenerate their
   documentation while retaining partial global statuses. Never promote full
   parity or performance on functional evidence alone.

Planned commands after approval:

```sh
python -m pytest -q tests/qec/test_native_detector_execution.py tests/unit/test_compiled_measurement_sampler_contract.py
python -m pytest -q tests/team/simulation/test_native_detector_execution_conformance.py tests/api_contract/test_public_docstring_examples.py tests/unit/test_qec_installed_profiles.py
python -m examples.qec.native_detector_sampling
python -I tools/verify_embedded_stabilizer_install.py
```

No command above was executed for the proposed feature. A separately approved
study/budget is still required for performance, capacity, threshold or mixed-route
experiments. This advances FTS-04/R03/R08; it does not close FTS-02, R04 or M2.

## Exact proposed repository scope

1. `flagquantum/circuit.py`
2. `flagquantum/qec/execution/detector_sampling.py`
3. `flagquantum/simulation/stabilizer/_detector_sampling.py`
4. `flagquantum/simulation/stabilizer/measurement_sampler.py`
5. `flagquantum/simulation/stabilizer/_embedded_sampler.py`
6. `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp`
7. `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md`
8. `flagquantum/simulation/stabilizer/README.md`
9. `flagquantum/qec/execution/README.md`
10. `contracts/qec-native-detector-execution-v1.json`
11. `docs/development/QEC_NATIVE_DETECTOR_EXECUTION_PROPOSAL.md`
12. `tests/unit/test_compiled_measurement_sampler_contract.py`
13. `tests/qec/test_native_detector_execution.py`
14. `tests/team/simulation/test_native_detector_execution_conformance.py`
15. `tests/api_contract/test_public_docstring_examples.py`
16. `tests/unit/test_qec_installed_profiles.py`
17. `tools/qec_installed_smoke.py`
18. `tools/verify_embedded_stabilizer_install.py`
19. `examples/qec/native_detector_sampling.py`
20. `examples/README.md`
21. `docs/reference/API.md`
22. `docs/reference/KNOWN_LIMITATIONS.md`
23. `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md`
24. `capability-maturity.toml`
25. `docs/generated/CAPABILITIES.md`
26. `contracts/cudaq-parity-matrix.toml`
27. `docs/reference/CUDAQ_PARITY_MATRIX.md`

No setup.py, MANIFEST.in, dependency, runner or CI workflow changes are included.
The source lock and all 96 upstream bytes stay immutable. Stop and disclose any
required expansion before implementation. Main admission of existing PRs remains
a separate gate; this proposal is not a delivery receipt.
