# API Change Proposal 048: Code-independent QEC records, detector error models, and distance-scaling evidence

## Status

**Implementation authorized; additive to the frozen repetition-code surface.** The
user authorized five staged slices that turn `flagquantum/qec` from one pinned
three-qubit reference profile into a code-independent QEC surface with a
detector error model, a replaceable decoder, and statistically supported
distance-scaling evidence. Each stage lands as its own reviewed change. A later
stage does not begin before the earlier stage's acceptance criteria hold.

## Problem

`flagquantum/qec/` ships exactly one reference profile: a three-data-qubit
repetition code, a two-bit-syndrome lookup decoder, a two-round temporal rule,
and a fixed-round memory experiment. The profile is pinned to three data wires
at the type level. `ErrorEvent.wire` and `Correction.wire` accept only `{0, 1, 2}`,
`PauliFrame.apply` requires exactly three data bits, `ErrorEvent.pauli` and
`Correction.pauli` accept only `"x"`, and `RepetitionMemoryShot.__post_init__`
requires the logical bit to be the majority of three decoded bits. The code
itself is not a value: it lives inside the source-string template
`_memory_source`, and the wire layout is implied by that string.

Four consequences follow.

1. **No distance scaling.** The reference profile cannot express `d = 5` or
   `d = 7`, so it cannot answer the question QEC exists to answer: how the
   logical error rate falls as code distance grows. `capability-maturity.toml`
   accordingly records logical suppression and thresholds as unsupported.
2. **No code abstraction.** Nothing can describe a code's checks, stabilizers,
   logical observables, or wire layout. A code is a string, so a second code
   cannot be added without a second string template.
3. **No detector or observable annotation.** `DetectionEvent` records exist, but
   nothing declares which measurement parities are deterministic in the
   noiseless circuit. Without that declaration no external decoder can be
   configured, because a decoder is defined by the detection events it consumes.
4. **No detector error model.** This is the missing abstraction. The current
   CUDA-Q QEC library treats the DEM as its central object — `DetectorErrorModel`,
   `dem_from_memory_circuit()`, `dem_from_stim_text()`, `dem_sampling()`,
   `detector_error_matrix`, `observables_flips_matrix` — and then attaches
   decoders rather than implementing them: its decoder list includes `pymatching`
   and `chromobius` alongside its own `nv_qldpc_decoder`. Without a DEM there is
   nothing to attach a decoder to, no way to state a code's error mechanism
   structure, and no basis for a logical-suppression claim that a reviewer can
   check independently.

**Parity target.** `qiskit-qec` is archived, read-only, and self-describes as
early-stage with breaking API changes. It is not a viable parity target and is
not used as one. The target is the CUDA-Q QEC abstraction surface plus
stim-compatible detector error model interchange, because "define the DEM, plug
in the decoder" is the shape both the CUDA-Q and the stim/pymatching ecosystems
converged on.

**Convention finding.** The frozen profile defines logical failure as the
majority of the frame-corrected data bits. Majority is a profile-specific
decision rule, not a linear logical observable: no operator's eigenvalue is
being measured, and majority and the code's declared logical operator can
disagree on the same syndrome history. A code-independent layer must define
failure as the parity of a declared logical observable, because only a linear
observable composes with distance scaling. This proposal therefore does not
claim that general-layer `d = 3` failure counts equal frozen-profile `d = 3`
failure counts. The frozen profile's own arithmetic is unchanged.

## Decision

### Layer placement and additive discipline

Add a code-independent layer beside the frozen repetition profile. Do not
generalize the frozen types. `contracts/hybrid-compilation-private-v0-candidate.json`
pins seven names as contractual surfaces:

```
phase16.decoder_contract   flagquantum.qec.Decoder
phase16.reference_decoder  flagquantum.qec.RepetitionLookupDecoder
phase16.workflow           flagquantum.qec.run_repetition_memory_experiment
phase18.finite_shot_sweep  flagquantum.qec.run_repetition_memory_noise_sweep
phase19.decoder_contract   flagquantum.qec.StreamingDecoder
phase19.reference_decoder  flagquantum.qec.RepetitionStreamingLookupDecoder
phase20.decoder            flagquantum.qec.RepetitionTemporalDecoder
```

Changing them would be a breaking change requiring contract re-review, and the
existing records describe a different concept — repetition-profile records —
from the code-independent records this proposal needs. One existing type is
already general: `DetectionEvent` is reused directly, because a detector is a
detector at any distance.

Everything below is new. The seven pinned names keep their current signatures and
semantics, and `run_repetition_memory_experiment` keeps its current behavior.

Modules stay flat under `flagquantum/qec/`, because the coverage policy keys in
`contracts/coverage-policy.toml` are non-recursive and a new subpackage would
need its own entry.

### Stage 1 — Code, Pauli, and detector records

- `pauli.py`: `Pauli`, a sparse Pauli string over arbitrary wire indices, with
  `support`, `weight`, `commutes_with`, composition, and a deterministic text
  form. This is the general replacement for the frozen `str`-valued `pauli`
  field, which only ever holds `"x"`.
- `codes.py`: the `StabilizerCode` protocol and `RepetitionCode(distance)`.
  The protocol exposes `distance`, `num_data_qubits`, `num_ancilla_qubits`,
  `data_wires`, `ancilla_wires`, `checks`, `stabilizers`, and
  `logical_observables`; the last is a tuple of `Pauli`, so failure is defined by
  a declared operator rather than by a counting rule.
- `circuit.py`: `build_memory_circuit(code, *, rounds)` returns the circuit
  source together with a `DetectorLayout` and an `ObservableLayout`, so detector
  and observable identity is derived from the code rather than asserted by the
  caller. A `DetectorLayout` is a dense, ordered tuple of detectors, each
  identified by the round and code check it belongs to and by the measurement
  parities that compose it. An `ObservableLayout` is a dense, ordered tuple of
  logical observables, each a `Pauli` plus the measurement parity representative
  that realizes it at readout.

Detector semantics are pinned as follows. A detector is a measurement parity
that is deterministic in the noiseless circuit. A `rounds`-round memory
experiment on a distance-`d` repetition code has `(d - 1) * (rounds + 1)`
detectors: one boundary per syndrome round comparing that round against its
predecessor, with the first round compared against the prepared all-zero prior,
plus one final boundary comparing the last syndrome round against the final data
measurements. The round-zero comparison against a known prior is the same rule
`SyndromeRound` detection events already apply.

`RepetitionCode(distance=3)` must reproduce the circuit currently produced by
`_memory_source(compiled_feedback=False)` instruction for instruction. That
equality is an acceptance test, not a comment.

### Stage 2 — Detector error model

- `dem.py`: `DetectorErrorModel`, `DemError` (a probability, a detector
  signature, and an observable-flip signature), `DetectorErrorModel.from_memory_circuit(...)`,
  `detector_error_matrix()`, `observables_flips_matrix()`, `dem_sampling()`,
  `to_stim_text()`, and `from_stim_text()`.

Construction is exact and does not sample, which is possible because the
reference gate set is Clifford and every configured channel is a Pauli channel.
Propagating one Pauli mechanism through a Clifford circuit yields a deterministic
detector and observable signature, so a single forced-error execution determines
that mechanism's signature exactly. Error probabilities come from the Pauli
decomposition of each channel — `two_qubit_depolarizing_channel` already builds
its Kraus operators explicitly from the Pauli basis in
`flagquantum/noise/channels.py`, so decomposition is a match against those basis
operators, not a numerical fit.

Two boundaries are explicit. The model is defined for Pauli noise only;
`coherent_overrotation_channel` and `thermal_relaxation_channel` have no Pauli
decomposition and fail closed rather than being approximated. And the model is
built on the memory circuit *without* in-circuit feedback, because the DEM
describes the physical noise-to-detection mapping that a decoder inverts;
`compiled_lookup` feedback is the layer the decoder replaces.

Cross-check acceptance: the detector rates the DEM predicts must agree, inside a
stated confidence interval, with rates sampled directly from the existing noisy
trajectory simulator. This is the strongest correctness evidence available for
the DEM and is required, not optional.

### Stage 3 — Decoder

- `decoding_graph.py`: `DecodingGraph` and `build_decoding_graph(dem)`, with edge
  weights `log((1 - p) / p)` and observable-flip labels on edges.
- `matching.py`: `MatchingDecoder`, a self-implemented minimum-weight
  perfect-matching decoder over graphlike detector error models. At `d <= 7`,
  `rounds <= 7` a detector graph has at most 42 detector nodes, so any correct
  minimum-weight algorithm is effectively instantaneous; no performance-grade
  matcher is required and none is attempted.
- `adapters.py`: a `pymatching` adapter behind a new `pymatching` optional extra,
  used as a cross-check rather than as the authority, plus a stim-text round-trip
  path.

Self-implementation is the authority for three reasons: it adds no runtime
dependency, it runs in the `cpu-core` lane that installs only `.[dev]`, and
bit-for-bit agreement between the self-implemented matcher and `pymatching` on
identical DEMs is itself the correctness evidence. Hyperedge and non-graphlike
detector error models fail closed rather than being approximated by graphlike
edges.

### Stage 4 — Statistics and evidence

- `statistics.py`: `wilson_interval` (preferred over the normal approximation
  because logical error rates at useful distances are small), `DistanceScalingPoint`,
  and `estimate_crossing(...)` returning a crossing estimate with a confidence
  interval and an explicit goodness-of-fit result.

Evidence is a provenance-bearing artifact, following the existing
`benchmarks/results/` pattern, not a CI test. Two measured constraints shape how
it is produced; both were measured against this repository before the proposal
was written.

First, the private hybrid dynamic simulator costs roughly linear time per shot
and does not amortize across a larger batch. At `d = 7, rounds = 7` it measures
about 9.6 ms per shot at 256 shots and about 11.9 ms per shot at 1024 shots.
Sampling a full curve from the circuit would therefore cost hours per
probability point. Second, the `batched` dynamic strategy fails closed with
`batched dynamic execution unavailable: memory budget exceeded` at `shots = 4096`
for `d = 7, rounds = 7`, so shots must be chunked regardless.

The artifact is therefore produced in two explicitly labeled layers:

- The DEM is validated against the **circuit simulator** at `d = 3` and `d = 5`,
  comparing DEM-predicted detector rates against circuit-sampled detector rates
  inside a stated confidence interval within an explicit shot budget.
- The distance-scaling curve is then sampled from the validated DEM through
  `dem_sampling(...)`, the same construction stim and CUDA-Q use for
  `dem_sampling()`.

Every artifact records which layer produced which number, and no artifact
presents a DEM-sampled rate as a circuit-sampled rate.

CI runs a reduced reproduction contract at `d = 3` and `d = 5` with a few
hundred shots that checks wiring and monotonicity only, because a threshold
crossing cannot be established inside a useful CI time budget and a CI test that
claimed one would be dishonest.

### Stage 5 — Documentation and contract close-out

`flagquantum/qec/README.md`, `IMPLEMENTATION.md`, `capability-maturity.toml`, the
generated capability catalog, and `contracts/coverage-policy.toml` are updated
together. The capability entry is added new; `repetition_code_memory` is not
edited.

### Noise scope

Phenomenological noise is the primary evidence: an independent flip on every data
qubit per round and an independent flip on every syndrome measurement. Literature
values exist for this combination, so the measured crossing can be compared
against a published number, which is the most direct form of the parity goal.
This adds a new `PhenomenologicalNoise` to `flagquantum/qec/noise.py` that
generalizes over `n` wires. The frozen `RepetitionNoiseProfile` is left exactly
as it is, and the top-level `flagquantum/noise` package is not modified.

Circuit-level noise is secondary supporting evidence: depolarizing and
measurement faults at realistic locations, using channels that already exist
(`two_qubit_depolarizing_channel`, `depolarizing_channel`). It demonstrates that
one DEM-and-matching implementation covers both noise scopes. No circuit-level
threshold is claimed from it.

## Public API

```python
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.matching import MatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.statistics import estimate_crossing

code = RepetitionCode(distance=5)
built = build_memory_circuit(code, rounds=5)
noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)

dem = DetectorErrorModel.from_memory_circuit(built, noise=noise)
print(dem.detector_error_matrix().shape)
print(dem.observables_flips_matrix().shape)
print(dem.to_stim_text()[:80])

crossing = estimate_crossing(
    probabilities=(0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
    distances=(3, 5, 7),
    rounds=5,
    noise=noise,
    decoder_factory=MatchingDecoder,
    shots=200_000,
    seed=0,
)
print(crossing.estimate, crossing.confidence_interval)
print(crossing.goodness_of_fit)
```

`to_stim_text()` and `from_stim_text()` interchange the same model with the stim
ecosystem without adding stim as a dependency. Detector-rate comparison between
the DEM and the circuit simulator is a test and an evidence step, not public API.

## Interpretation

A DEM means that every independent physical error mechanism has been identified
with the detectors and logical observables it flips, and that the model is
exact for Pauli noise in the reference gate set. It does not mean the noise
model matches any real device. A crossing estimate with a confidence interval
means that the measured finite-shot logical error rates for the stated distances
intersect within that interval under the stated noise model. It does not mean
real hardware has that threshold, and no fault-tolerance claim follows.

Real-time decoding, hardware feedback, surface codes, correlated and non-Pauli
noise, and LDPC codes remain out of scope.

## Stage 1 landed state and known debt

Stage 1 is implemented on branch `feat/qec-stage1-code-records`. The layer exists
as `flagquantum/qec/{pauli,codes,circuit}.py`, publishes eleven names from
`flagquantum.qec`, documents its boundary in `flagquantum/qec/IMPLEMENTATION.md`,
and is covered by `tests/qec`, which grew from 82 tests to 201.

This section exists because a later stage must not build on the following
limitations without knowing they are there. The items below were found during
review, judged minor, and deliberately not fixed in Stage 1.

**Constraints a later stage will hit:**

- `CodeCheck` can describe only Z-type checks. Every CNOT must control a data wire
  and target the check's ancilla, so only a Z-basis ancilla measurement is
  reachable, and a stabilizer with an X component is rejected by validation. A
  code family needing X-type checks requires this record to grow first.
- A detector's parity is validated for wire and round *membership* only. Nothing
  checks that a detector names the same check across rounds, that a terminal
  detector's data wires are the support of the check it belongs to, or that the
  `source` text is the program the layouts describe. A hand-built layout can
  therefore be semantically wrong while passing every validator.
- `CodeCheck.index` is not consumed by the library. The emitted measurement order
  and the detector stride key on tuple position, so a code may declare indices that
  disagree with position with no error. The DEM must key on position, not on
  `index`.
- The `StabilizerCode` protocol is enforced by `isinstance` only, which for a
  `runtime_checkable` protocol tests attribute presence. Nothing cross-checks
  `num_data_qubits`, `num_ancilla_qubits`, or `stabilizers` against
  `data_wires`, `ancilla_wires`, or `checks`, and `stabilizers` is fully derivable
  from `checks`.
- `runtime_checkable` protocols with property-only members raise on `issubclass()`
  before Python 3.12, while `pyproject.toml` declares `>=3.10,<3.13`. Only
  `isinstance` is used today, which is safe on all supported versions.

**Smaller items, safe to carry:**

- `Pauli.from_label` skips wire validation on its identity path, and `from_text`
  accepts leading zeros and Unicode digits and reports a duplicated term as a
  field-ordering error. `to_text`'s exact format is pinned only by round-trip
  tests. No library consumer uses these yet; the DEM is the first likely one.
- `Detector.index` and `LogicalObservable.index` accept non-integers (including
  `bool`) while `MeasurementRef`, `MemoryCircuit.rounds`, and `RepetitionCode`
  apply `Integral` guards. `MemoryCircuit.source` is validated only for
  non-emptiness.
- `build_memory_circuit` rejects a code whose declared logical leaves its declared
  `data_wires`, reporting it as an observable-readout error. That is code-level
  incoherence rather than a builder defect, and it has no test.
- The observable-count validator has no test: removing it leaves the whole suite
  green.
- `checks` and `stabilizers` rebuild and re-validate their tuples on every
  attribute access.
- `flagquantum/qec/README.md` is the user-facing entry point and still lists only
  `repetition.py`, `decoders.py`, `noise.py`, and `types.py`. Stage 5 owns the QEC
  documentation sweep that will bring it up to date.

## Compatibility

The change is additive. The seven pinned names, `repetition_code_memory`
capability semantics, all existing `flagquantum.qec` exports, and all existing
`tests/qec` behavior are unchanged. The `pymatching` adapter is behind an
optional extra and is never imported at module import time. `flagquantum.noise`
is not modified.

## Acceptance

- `RepetitionCode(distance=3)` reproduces the circuit currently produced by
  `_memory_source(compiled_feedback=False)` instruction for instruction, and
  `RepetitionCode(distance=5)` and `(distance=7)` build, lower, and execute
  through the private hybrid compiler without compiler or runtime changes.
- The detector count for a `rounds`-round distance-`d` memory experiment is
  `(d - 1) * (rounds + 1)`.
- Every DEM detector and observable flip signature is reproduced by a
  single-forced-error execution, and no mechanism is missing or duplicated.
- DEM-predicted detector rates agree with rates sampled from the noisy
  trajectory simulator inside a stated confidence interval, at `d = 3` and
  `d = 5`, within an explicit shot budget.
- Non-Pauli channels, hyperedge detector error models, and unsupported gates fail
  closed with a stated reason rather than being approximated.
- The self-implemented matcher and `pymatching` agree bit for bit on identical
  graphlike DEMs, and the agreement is a test.
- Logical failure is the parity of a declared logical observable, and the
  frozen profile's majority convention is documented as a deliberate,
  non-equivalent convention rather than silently unified.
- Every measured logical error rate is reported with a confidence interval, and
  every crossing estimate reports goodness of fit or an explicit failure.
- No document, docstring, or capability row claims a threshold, logical
  suppression, real-time decoding, or hardware feedback beyond the evidence
  actually produced.
- Every DEM-sampled logical error rate is labeled as DEM-sampled, and no
  artifact presents one as a circuit-sampled rate.
- Shot batches are chunked so that the `batched` dynamic strategy's memory budget
  is never exceeded.
- Report-only artifacts carry provenance, and CI covers wiring and monotonicity
  only.
- API contract, guide, limitations, release notes, tests, and executable example
  describe the same behavior.
