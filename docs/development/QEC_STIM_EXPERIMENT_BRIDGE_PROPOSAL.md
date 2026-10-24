# QEC annotated Stim experiment bridge proposal

## Status and decision

Proposed for roadmap E; not approved and not implemented. Approval of the
complete compatibility delivery sequence authorizes this reviewable design,
not the new public entry points below. Core owns QEC composition; Simulation
owns engine execution and all imports of Stim. Integration reviews the shared
boundary and protected additive public surface before implementation.

This proposal adds no root `flagquantum.qec` export, result class, registry,
portable artifact schema or mandatory dependency. The proposed opt-in module is
`flagquantum.qec.adapters.stim`. Existing imports, constructors, method defaults,
results, failure contracts and frozen restructuring artifacts remain unchanged.

## Existing authorities and demonstrated gaps

- `MemoryCircuit` owns declaration order, detector parities and logical
  measurement parities; `execution/sampling.py` owns lowering, placement and the
  mapping from those references to recorded columns.
- `simulation/stabilizer/engine.py` already translates the positioned CircuitIR
  and terminal-qubit list into Stim. It remains the sole production Stim import.
- `MemorySampler` already provides owner-bound, bounded, repeated draws returning
  the existing CPU int8 `DemSample`; reuse its lifecycle and logical meaning.
- `DetectorErrorModel.from_stim_text` and the existing PyMatching adapter already
  ingest an external model and decode its events. Neither reads an annotated
  external circuit into a reusable owned sampler. DEM text is not the original
  syndrome-extraction circuit and cannot recover it.
- The current text reader requires explicit, contiguous detector declarations.
  Stim may declare them only implicitly through error targets when coordinates
  are absent. The bridge must supply explicit shape without weakening or
  changing that existing reader. Missing intermediate logical declarations also
  need normalization when referenced and explicit logical indices are mixed.
- A deterministic measurement can be one in the absence of noise. Raw record
  parity is therefore not a detection-event/observable-flip sample for arbitrary
  external input. Use the engine's reference-normalized detector sampler.

## Proposed public entry points

All three functions live only in `flagquantum.qec.adapters.stim`:

```python
def memory_to_stim_text(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
) -> str: ...

def compile_stim_sampler(
    circuit_text: str,
    *,
    seed: int | None = None,
) -> MemorySampler: ...

def stim_circuit_to_dem_text(
    circuit_text: str,
    *,
    decompose_errors: bool = False,
) -> str: ...
```

`memory_to_stim_text` exports the positioned program used by actual native
sampling, appends terminal data measurements in the same order, then emits
`DETECTOR rec[...]` and `OBSERVABLE_INCLUDE(index) rec[...]` from the existing
column maps. Measurement order and detector/logical numbering are unchanged.
Do not reconstruct channels from names or nominal scalar probabilities: the
Kraus operators in the positioned program define its executed distribution.
A generic native declaration carries no geometric coordinates, so none are
invented or inferred from wire numbering.

`compile_stim_sampler` snapshots immutable source text and prepares a single
reference-normalized stream. It requires at least one detector, allows zero or
multiple observables, and returns the existing `MemorySampler`. Each draw
returns `(B, D)` detector bits and `(B, L)` logical flips, never measurement bits
or SDK objects. Reuse creator process/thread checks, positive-shot validation
before RNG advancement, owner-idempotent close, context cleanup, failure
poisoning, no copy/pickle and caller-owned CPU buffers. A fixed seed repeats an
identical call schedule within the recorded engine/version/architecture;
different batching need not produce identical bits.

Validate noiseless detector and logical determinism before returning a handle,
using the noise-free circuit and disallowing gauge detectors. Noise removal is
only a determinism check: the prepared stream always executes the original
noisy source. Do not require noisy DEM analysis to succeed in order to sample
an otherwise supported circuit; a DEM representation refusal is separate.

`stim_circuit_to_dem_text` analyzes the original circuit with
`allow_gauge_detectors=False`, `approximate_disjoint_errors=False`,
`ignore_decomposition_failures=False` and the caller's explicit decomposition
choice. Unsupported exact conversion raises a capability error rather than
turning on disjoint-error approximation or dropping mechanisms. Return plain
text with explicit complete detector/logical declarations. Flatten bounded DEM
loops when normalizing index offsets, preserve effective coordinates and all
existing coordinate declarations, and add only missing shape declarations.
The exact text is not a stable serialization schema or a byte-for-byte round
trip promise.

The existing owned DEM remains a numerical detector/logical model without new
coordinate fields. Keep source and canonical DEM text as the coordinate-bearing
carriers. `DetectorErrorModel.from_stim_text(text)` reads the raw law; an explicit
matching-only conversion may use existing `use_decomp_suggestions=True`.
Decomposition suggestions must never replace the circuit sampled for outcomes.

## Supported scope, bounds and errors

The external entrance accepts the installed supported Stim version's annotated
Clifford circuits with finite repeats, supported resets/measurements/Pauli noise,
record-controlled operations and coordinate shifts. It promises no arbitrary
Stim-to-native IR reverse compilation, QPU execution, real-time feedback,
non-Clifford approximation or external device support. Preserve instruction
ordering and actual exclusivity of channel branches.

Proposed initial validation bounds are 1 MiB UTF-8 source, 4,096 qubits,
1,000,000 expanded instructions and 1,000,000 total recorded
detector/observable columns. Check declared wire and logical indices before
compiling a sampler, including sparsely numbered very large indices. Count
repeat bodies symbolically before materializing any expansion; reject overflow
or an excess with `CapabilityError`. These are disclosed initial support bounds,
not throughput or capacity claims. Sampling storage remains batch-bounded; no
past sample is retained. Do not add a new task scheduler or generic budget API.

- Invalid original argument types, including a non-string source or boolean seed,
  raise `TypeError`; empty/malformed source or an invalid scalar value raises
  `ValueError` with the underlying syntax cause retained where relevant.
- Supported syntax with nondeterministic declared parities, excessive declared
  expansion, or unrepresentable exact DEM analysis raises `CapabilityError`.
- Selecting the bridge without Stim raises the existing
  `StabilizerDependencyError` with installation guidance. Imports and decoder
  registry inspection still succeed without Stim/PyMatching. Do not install or
  select a fallback automatically.
- Failures preparing a handle release any partial SDK state; draw failures retain
  their original error and the existing poisoned-handle lifecycle. No partial
  success report, retries or silently dropped rows are introduced.

## Internal owner boundary

The separate QEC-FUNCTIONAL-005 sparse batch contract remains unapproved and
is not required or activated here. Use the already approved logical-only
PyMatching Tensor batch operation for the first shared-evaluation example.

QEC supplies an already positioned CircuitIR, terminal-qubit sequence and plain
immutable parity-column tuples for native export. Simulation privately owns
serialization of its engine circuit and its annotations. For external input,
Simulation parses text, computes bounded shape/determinism and retains the SDK
sampler behind the existing prepared-measurement object. It returns plain counts
and prepared owned bit records to the QEC composition layer. An identity column
selection over concatenated detector/observable flips lets the existing
`MemorySampler` provide the public lifecycle without a second sampler manager.

Any minimal private extension of `_PreparedMeasurements` must retain every
existing native path's behavior. Add no public Simulation boundary or engine
protocol just to support these three adapter functions. Keep existing registry,
root exports and historical fixtures read-only; review a narrow candidate
contract for the new opt-in namespace before activation.

## Existing runnable user example

The following uses current public APIs; it is not the proposed bridge API:

```bash
pip install -e '.[stim,pymatching]'
```

```python
import stim
import torch
import flagquantum.qec as qec

# An external annotated experiment, kept as a circuit rather than DEM samples.
circuit = stim.Circuit('''
    R 0
    X_ERROR(0.1) 0
    M 0
    DETECTOR(0) rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
''')
model = qec.DetectorErrorModel.from_stim_text(str(circuit.detector_error_model()))
decoder = qec.PyMatchingDecoder.from_detector_error_model(
    model.merge_duplicate_mechanisms()
)
detectors, observables = circuit.compile_detector_sampler(seed=53).sample(
    shots=35, separate_observables=True
)
sample = qec.DemSample(
    detectors=torch.from_numpy(detectors), observables=torch.from_numpy(observables)
)
predicted = decoder.decode_batch(sample.detectors)
failures = int((predicted != sample.observables).any(dim=1).sum())
print(sample.shots, model.num_detectors, model.num_observables, failures)
```

Observed output on Stim 1.16.0 / Torch 2.13.0 / macOS arm64:

```text
35 1 1 0
```

This one-bit example makes the detector and logical flip coincide, so it
illustrates the external-circuit data path, not a logical-suppression result.
The bridge replaces manual SDK preparation/conversion with owned bounded
sampling; existing native-memory examples continue to work unchanged.

## Acceptance before implementation delivery

1. Native Repetition and Surface X/Z d=3/5/7, one/three rounds: exported
   measurement order, map identities, detector/logical counts and normalized
   parity agree on the same raw measurement batch using an independent Stim
   m2d converter. Cover initial/final boundaries and forced faults.
2. External sources: nonzero noiseless reference, zero/multiple/gapped logical
   indices, unused detectors/logicals, finite nested repeats, coordinate shifts,
   syntax failures, nondeterministic parities and expansion-limit refusals.
3. Compare raw and canonicalized DEM shapes, coordinates and probabilities;
   missing declarations are supplied without changing the existing text
   reader's refusal contract. Exact disjoint-error refusal stays a refusal.
4. Two entrances feed the same existing shot-level comparison and fixed-shot
   statistics. Sample the original circuit, never the decomposed matching model;
   compare native/PyMatching predictions on the same samples with explicit
   handling of degenerate matching and unsupported graphs.
5. Existing native sampler lifecycle/RNG/shape tests remain unchanged and pass;
   exercise the same owner, close, failure and invalid-request cases for the
   external factory. Run real optional-dependency profiles and installed-package
   examples; no stubs may stand in for real SDK presence/absence evidence.
6. User-facing Python examples, strict type/format/architecture/API gates and
   required current-head CI/GPU pass before the normal delivery merge.

## Read-only feasibility evidence and an open precision issue

A private developer probe on the pre-bridge implementation used four cells,
Surface d=3/5 x X/Z, three rounds, 35 shots, seed 53 and the fixed supported
phenomenological noise. Native sampling and independently converted exported
measurements agreed exactly for every sampled detector/observable bit. Shapes
were 33/97 measurements, 24/72 detectors and one observable. A noiseless
nonzero-measurement witness normalized to zero flips, a coordinate/repeat source
round trip retained geometry, and a nondeterministic detector was refused.
This is feasibility evidence, not a public adapter implementation or complete
E acceptance.

The initial raw DEM import failed because its detector shape was implicit;
adding missing explicit declarations solved that gap without editing the
reader. A separate configured-probability comparison at absolute tolerance
1e-12 failed: actual default Kraus probabilities differ from configured decimal
rates through existing finite tensor precision. Detector-rate differences were
about 4.65e-9 and logical marginal differences up to 7.53e-9. Using the actual
Kraus law as the independent input gave zero difference in the four cells.
Keep both original failed criteria and raw evidence. Export must preserve the
executed law; do not silently rewrite operators from nominal metadata, relax an
existing scientific gate or claim configured/native/exported distributions are
identical. F/G/H must resolve and validate the configured-versus-executed
numerical precision contract before whole-route acceptance I.

Primary semantics reference: [Stim 1.16 source API reference](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/python_api_reference_vDev.md),
verified against the installed 1.16.0 SDK for the probe. Its strict DEM conversion
separates decomposition suggestions from disjoint-channel approximation. The
[DEM format](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/file_format_dem_detector_error_model.md)
defines coordinates, detector/logical targets, shifts and repeat semantics.
