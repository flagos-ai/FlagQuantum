# Migrating a Stim workflow to FlagQuantum

FlagQuantum supports both Stim interoperability and bounded native QEC circuit
execution. Keep Stim circuit generation when needed, or use native Circuit IR
with measurement maps. Compiled samplers and converters use an embedded pinned
engine in qualified wheels; SDK-backed profiles use the optional Stim adapter.
Vendor objects remain behind the numerical boundary.

The model sampler draws declared fault mechanisms; circuit detector sampling
executes the original noisy circuit. These are distinct distributions and
validation routes, not interchangeable performance claims. The guide's original
model-reading examples remain supported below.

| Stim | FlagQuantum |
| --- | --- |
| `stim.Circuit.generated` | keep it: generate circuits with Stim |
| `circuit.detector_error_model(...)` | keep it: write the model text with Stim |
| `dem_from_stim_text(text)` | `DetectorErrorModel.from_stim_text(text)` |
| `dem.compile_sampler(...).sample(...)` | `DetectorErrorModel.dem_sampling(...)` |
| `circuit.compile_detector_sampler(...)` | Native `circuit.compile_detector_sampler(detector_map=..., observable_map=...)` for admitted circuits; DEM sampling remains separate |
| `pymatching.Matching.from_detector_error_model(...)` | `get_decoder("minimum_weight_matching", model)` |

## Read the model text

Start from a circuit exactly as you would without FlagQuantum.

```python
import stim

from flagquantum.qec import DetectorErrorModel

circuit = stim.Circuit.generated(
    "repetition_code:memory",
    distance=3,
    rounds=20,
    before_round_data_depolarization=0.05,
    before_measure_flip_probability=0.02,
    after_reset_flip_probability=0.01,
)
detector_error_model = circuit.detector_error_model()
raw_text = str(detector_error_model)
print(len(raw_text.splitlines()), sum(1 for line in raw_text.splitlines() if line.strip().startswith("repeat")))
# 35 1  -- Stim's writer uses `repeat` as soon as a block is worth repeating
parsed = DetectorErrorModel.from_stim_text(raw_text)
print(
    parsed.num_detectors,
    parsed.num_observables,
    len(parsed.errors),
    parsed.mechanisms_are_unique(),
)
# 42 1 141 False  -- read as written, and 141 mechanisms of which two share a signature
```

`from_stim_text` reads the block rather than refusing it: a `repeat <count> { ... }`
block is interpreted by running the block's instructions the number of times its
count states, so the detectors, the mechanisms and the accumulated
`shift_detectors` offsets are the ones the instructions produce in sequence.
Blocks nest, `repeat 0` contributes nothing, and `stim`'s `flattened()` form is
one route to the same model rather than a prerequisite. A `#` comment is cut at
the `#` wherever it sits, so a commented line states no instruction.

What the reader still refuses is the text its own record cannot hold. Every
refusal names what it could not read rather than dropping the instruction, so a
text carrying information this model cannot state fails closed instead of being
read as a shorter model:

| Text | Refusal | Repair |
| --- | --- | --- |
| `detector D1` with no `detector D0` | `detector declarations must be consecutive from D0` | declare every index, or use `shift_detectors` |
| `error(0.01) D0 D0` | `an error mechanism must flip at least one detector or observable` | name the mechanism's targets, at least one of which survives |
| `repeat 3 { ...` with no closing brace | `a repeat block must close with a brace on a line of its own` | close the block, or drop the `repeat` and write the instructions out |
| `X_ERROR(0.01) 0` | `unsupported stim instruction 'X_ERROR(0.01) 0'` | keep circuit instructions in Stim; this reader takes a detector error model |

## One fault stated twice

`mechanisms_are_unique()` is `False` above, and that is not a defect in the
reading: two different lines can erase to the same detectors *and* the same
observables. A matcher that sees only parity cannot weight two parallel edges
for one signature, so it refuses the model and names the merge:

```python
from flagquantum.qec import get_decoder

try:
    get_decoder("minimum_weight_matching", parsed)
except ValueError as error:
    print(error)
    # mechanisms 7 and 8 both flip D2 D4; merge_duplicate_mechanisms() states one prior for a shared signature and is what a decoder needs  -- the matcher's refusal
model = parsed.merge_duplicate_mechanisms()
print(len(model.errors), model.mechanisms_are_unique())
# 103 True  -- every shared signature now carries one prior
```

The default `DemMergeRule.INDEPENDENT_PARITY` gives a shared signature the
probability that an odd number of its members fires, which is the prior a
decoder needs; the alternative rule and the closed-form formulas are in
[`flagquantum/qec/README.md`](../../flagquantum/qec/README.md).

## Sample the model without a circuit

```python
sample = model.dem_sampling(shots=3, seed=1)
print(type(sample).__name__, tuple(sample.detectors.shape), sample.detectors.dtype)
# DemSample (3, 42) torch.int8  -- one row per shot, one column per detector
print(tuple(sample.observables.shape), sample.observables.dtype)
# (3, 1) torch.int8  -- one column per logical observable
```

The rows are bit **indicators** and not detector **indices**, which is the one
shape change to make when moving from Stim's sampler to `decode`:

```python
import numpy as np

shots = 4000
sample = model.dem_sampling(shots=shots, seed=11)
matcher = get_decoder("minimum_weight_matching", model)
predictions = np.array(
    [
        bool(matcher.decode(np.flatnonzero(row)).observables)
        for row in sample.detectors.numpy().astype(bool)
    ]
)
answers = sample.observables.numpy()[:, 0].astype(bool)
print(shots, round(float(np.mean(predictions != answers)), 4))
# 4000 0.1565  -- the logical error rate of this model at this distance
```

`decode` takes the flat indices of the detectors a shot flipped, so a bit row
goes through `np.flatnonzero` first; passing the row itself raises "a syndrome
cannot name the same detector twice". `MatchingDecodeResult.observables` is a
tuple and is empty when no observable flips, so read it as a truth value rather
than indexing its first element.

## The `^` separator has two readings, and they are not equivalent

A depolarizing fault on a check can flip three or four detectors at once. Stim's
decomposed output splits each such mechanism with `^` into graphlike pieces, and
FlagQuantum offers both readings of that separator under the same flag name
Stim uses.

```python
surface = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
    before_measure_flip_probability=0.01,
    after_reset_flip_probability=0.01,
)
combined = DetectorErrorModel.from_stim_text(str(surface.detector_error_model()))
expanded = DetectorErrorModel.from_stim_text(
    str(surface.detector_error_model(decompose_errors=True)),
    use_decomp_suggestions=True,
)
print(len(combined.errors), sum(1 for error in combined.errors if len(error.detectors) > 2))
# 219 113  -- the line as written, and 113 mechanisms that are hyperedges
print(len(expanded.errors), expanded.mechanisms_are_unique())
# 556 False  -- one mechanism per component, and the split repeats signatures
graphlike = expanded.merge_duplicate_mechanisms()
print(len(graphlike.errors), graphlike.mechanisms_are_unique())
# 78 True  -- what a matcher will accept
```

The default reading states the line Stim wrote: the signature is the symmetric
difference of the line's targets. It is faithful and it is **not graphlike**,
which is why the matcher above refuses it. `use_decomp_suggestions=True` gives
one mechanism per component at the line's own probability, which is the
graphlike decomposition a matcher consumes, and its cost is measurable on the
number a threshold estimate reads:

```python
reference = surface.detector_error_model().compile_sampler(seed=4).sample(shots=200000)[1]
print(round(float(combined.observable_rates()[0]), 3))
# 0.146  -- the default reading against Stim's own sampler
print(round(float(graphlike.observable_rates()[0]), 3))
# 0.189  -- the expanded reading, 29% above it
print(round(float(reference[:, 0].mean()), 3))
# 0.146  -- Stim's sampler of the same detector error model text
```

Three decimals, and not the six an earlier draft printed, because one of the three
numbers is Stim's and not this package's: a sampled mean over 200000 shots is a
draw from Stim's random stream, whose exact value belongs to the Stim build in use.
It read `0.146475` here and `0.146075` on the CI runner, four hundredths of a
percentage point apart and both well under one standard error of the estimate
(`sqrt(0.146 * 0.854 / 200000)` is `7.9e-4`). What the block claims is which of the
two readings agrees with Stim, and that survives the third decimal: the expanded
reading sits ten times further from Stim's rate than the default one does. The
first two lines are rounded to match so the three are compared like for like.

Stim's sampler of the *decomposed* text reports the same rate, so the disagreement
is in the reading and not in the sampling. The two readings are therefore a choice
rather than a preference:

- **Decoding a syndrome.** Decompose and merge. A minimum-weight matcher needs a
  graphlike model, `get_decoder("minimum_weight_matching", graphlike)` or
  `get_decoder("pymatching", graphlike)` both accept it, and they agree shot for
  shot.
- **A threshold, a marginal, or any rate.** Use the default reading, which is the
  model Stim wrote, and decode it with a decoder that carries hyperedges.

```python
events, observables = surface.compile_detector_sampler(seed=9).sample(
    shots=2000, separate_observables=True
)
syndromes = events.astype(bool)
hyperedge_decoder = get_decoder("belief_propagation_osd", combined)
matcher = get_decoder("minimum_weight_matching", graphlike)
cross_check = get_decoder("pymatching", graphlike)


def predicted(decoder):
    return np.array(
        [bool(decoder.decode(np.flatnonzero(row)).observables) for row in syndromes]
    )


matching = predicted(matcher)
crossed = predicted(cross_check)
hyperedges = predicted(hyperedge_decoder)
print(matching.tolist() == crossed.tolist())
# True  -- the crossed routes agree on every shot
print(float(np.mean(matching == crossed)) > float(np.mean(matching == hyperedges)))
# True  -- and neither agrees with the hyperedge route as often
```

The block prints those two comparisons instead of the two rates, because the rate
is Stim's draw and the comparison is not. Both readings decode every shot to the
same logical class, so their rates are equal by construction -- 0.0375 here and
0.047 on the CI runner over the same 2000 shots and the same seed, a difference of
nineteen shots and about two standard errors of an estimate at this size -- which
is exactly why a rate cannot choose between them, and also why quoting it would
pin the draw rather than the claim.

The per-shot agreement survives both builds: the in-tree matcher and the PyMatching
cross-check return the same class on every shot, while belief propagation with
ordered statistics agrees with the matcher on all but about two percent of them
(34 of the 2000 here). What actually separates the readings is the marginal in the
table above, which is why a migrated script's correctness check belongs on the
model and not only on the decoded rate.

## Going back to Stim

A model built in FlagQuantum writes Stim's format, so a route that ends in Stim
or PyMatching does not need a translation layer:

```python
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
)

built = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=3),
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
)
text = built.to_stim_text()
print(text.splitlines()[0])
# error(0.02) D0 L0  -- FlagQuantum writes the format Stim reads
round_trip = stim.DetectorErrorModel(text)
print(len(built.errors), round_trip.num_detectors, round_trip.num_observables, round_trip.num_errors)
# 15 8 1 15  -- every mechanism survives the round trip
```

The writer prints a probability with `repr` — the shortest decimal that reads
back as the identical double — where Stim's own printer uses the platform's
`long double` width. On arm64 macOS the two differ in the sixteenth significant
digit; the writer's own precision check is in
[`flagquantum/qec/IMPLEMENTATION.md`](../../flagquantum/qec/IMPLEMENTATION.md).

## Read a model back into a circuit

Stim's `compile_detector_sampler` turns a detector error model into something
that draws shots from it, and the same route exists on this side. A model built
here is not a dead end: `circuit_from_detector_error_model` writes it as the
circuit it already is — one wire per detector, one per observable, one
Pauli-frame channel per fault — and `detector_error_model_from_circuit` reads it
back.

```python
from flagquantum.qec import (
    circuit_from_detector_error_model,
    detector_error_model_from_circuit,
)

realization = circuit_from_detector_error_model(built)
print(realization.n_wires, len(realization.instructions))
# 9 24  -- one wire per detector and observable, one frame per mechanism
same = detector_error_model_from_circuit(realization, num_detectors=built.num_detectors)
print(same.num_detectors, same.num_observables, same.num_errors, same == built)
# 8 1 15 True  -- the model this began from rather than one like it
```

What comes back is the model the trip started from, error ids included, because
the two halves of a frame are read from different parts of the instruction: the
wire list and the operators say which targets a fault flips, and the mass of each
branch is declared rather than recovered by squaring a square root. The
realization is a Clifford program, so it samples on the stabilizer engine and no
statevector enters it: a distance-7 rotated surface patch's model is 49 wires and
43 mechanisms, and it draws 20000 shots in 0.02 seconds.

## What this migration does not cover

These are stated as boundaries rather than left for a caller to discover.

- **No circuit-level reader.** `from_stim_text` reads detector error models only.
  Every circuit instruction, `X_ERROR` included, is refused by name. Stim stays
  the circuit reader and the circuit sampler.
- **A realization is not the experiment the model came from.** A detector error
  model states which detector parities a fault flips and never which measurements
  compose a detector, so the syndrome-extraction circuit, its gate sequence, its
  depth and its ancilla layout are outside it: two different memory experiments of
  the same distance under the same noise record define the same model, and the
  reader returns the canonical detector-level circuit rather than either
  experiment. The detector count is the caller's to state, because a wire index
  does not say which side of the split it falls on.
- **No block expansion.** A `repeat <count> { ... }` block is interpreted rather
  than kept, so the reader pays one expansion Stim's own reader does not: the
  detectors, the mechanisms and the accumulated `shift_detectors` offsets are
  the ones the block's instructions produce in sequence. The model either
  reader produces is the same, and `flattened()` on the Stim side remains an
  equivalent route rather than a prerequisite.
- **Comments are cut, not refused.** A `#` is cut wherever it sits and runs to
  the end of its line, so a commented line states no instruction. Stim's own
  writer does not emit comments into a detector error model, so this only
  matters to a text a caller edited.
- **The registry's promise is a decoded syndrome, and the pair graph is a
  narrower one.** `decoder_names()` names five decoders: the in-tree matcher,
  the windowed reading of it, the PyMatching cross-check, and two
  belief-propagation readers, the sum-product `belief_propagation` and the
  ordered-statistics `belief_propagation_osd` that answer the model as written.
  What no name promises is `graph`: `GraphlikeDetectorErrorModelDecoder` states
  that on its own, the three matcher names satisfy it, and asking either
  belief-propagation name for a `DecodingGraph` is refused with a reason rather
  than failing inside a constructor, because a model with a three-detector
  mechanism has no pair graph. So `get_decoder` takes the five names and three
  carriers -- a model, Stim's text for one, a decoding graph -- and a
  parity-check matrix is not among them, since lifting one would mean choosing
  the noise model and round count as well.
- **The registry's third name is the matcher, banded.** `get_decoder` reaches
  `MinimumWeightMatchingDecoder` by name, and it reaches the same matcher over a
  window by name too: `sliding_window_matching` cuts a history into bands of
  detectors and matches one window at a time, which is how a syndrome longer than
  the exact matcher's defect budget is answered at all. A history that fits the
  budget needs no window and the matcher's name is the one to ask for; a history
  that does not is the case the windowed name exists for, and its answer is an
  answer about a window rather than about the whole history.
- **Model sampling is not circuit sampling.** `dem_sampling` samples the detector
  error model, which is the object a decoder consumes. A study that needs
  circuit-level samples keeps `compile_detector_sampler`, and a study that needs
  FlagQuantum's own noisy circuit execution uses the noise models in
  [`docs/guides/NOISY_SIMULATION.md`](NOISY_SIMULATION.md).
- **Neither the construction route, the reading route nor the realization is
  bounded by the patch's width.** Reading a large patch's text costs one parse,
  building a model from a memory circuit derives each signature from the circuit's
  own layouts, and the realization is a Clifford program on one wire per detector
  and observable, so a distance-7 rotated surface patch is 97 wires and one
  observable and reaches a model in under a quarter of a second. The scope that
  does remain is recorded in
  [`docs/reference/KNOWN_LIMITATIONS.md`](../reference/KNOWN_LIMITATIONS.md).

The runnable, gate-checked version of every block above is
[`tests/qec/test_stim_user_migration.py`](../../tests/qec/test_stim_user_migration.py),
which executes each block in one namespace and compares its output against the
values quoted here.


## Fixed-shot reports from your Stim circuit

Use `run_stim_experiment` when you already hold supported Stim circuit text and
want complete logical-failure counts and a confidence interval. This entry uses
the original circuit as its sampling source, strict DEM analysis and the selected
decoder's existing default configuration. Install the `stim` extra; the
`pymatching` selection additionally requires that extra.

```python
from flagquantum.qec.execution.experiments import run_stim_experiment

text = (
    "R 0 1\nX_ERROR(0.125) 0\nX_ERROR(0.25) 1\nM 0 1\n"
    "DETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2]\n"
)
result = run_stim_experiment(
    text, decoder="minimum_weight_matching", shots=35, batch_size=16, seed=53
)
print(result.shots, result.logical_failures, result.observable_failures)
# 35 0 (0, 0) -- exact decoding of all four independent-fault outcomes
print(result.sampling_source)
# stim_circuit -- original external circuit sampling source
```

Expected: `35 0 (0, 0)` followed by `stim_circuit`. The swapped logical columns
are deliberate. This small independent-fault example has an exact logical
prediction for each syndrome; it makes no physical suppression or speed claim.
The runnable script is
[`stim_memory_experiment.py`](../../examples/qec/stim_memory_experiment.py).

`StimExperimentResult` is separate from the existing `MemoryExperimentResult`.
For FlagQuantum memory declarations continue using `run_memory_experiment`.
Keep using the lower-level bridge when you need direct samples or manual decoder
control. Unsupported hyperedges, isolated-detector/parallel-edge translation
cases and other existing refusals remain explicit; this entry does not enable
approximate graph conversion, implicit decoder fallback or adaptive stopping.
See the [approved contract](../development/QEC_STIM_TEXT_EXPERIMENT_CONTRACT.md).


## Write detector samples without retaining all shots

The detector-file candidate adds the familiar `sampler.sample_write` method;
see the [approved contract and executable example](../development/QEC_DETECTOR_FILE_OUTPUT_PROPOSAL.md).
Run `python examples/qec/native_detector_sampling.py` in a qualified profile.
The example retains the original decoder workflow and adds a deterministic
repeated-slot witness, expected to print `detector_bytes 64`.

Use `format="b8", append_observables=True` for detector-then-observable bits.
Alternatively set `obs_out_filepath=...` and `obs_out_format="b8"` to write
same-shot logical flips separately. Six formats are supported: 01, b8, r8,
hits, dets and ptb64. Dets files name detectors D and logicals L; measurement
files name occurrences M. These are reference-normalized flips, not raw records.

Select at most one observable layout. Ptb64 requires shots divisible by 64.
Existing destinations are refused. Final files total at most 1 GiB, with
conservative sparse bounds and a 64 MiB principal-table estimate. These guards
are not total disk or process RSS limits. Both streams validate before sequential
no-replace publication; two-file atomic visibility is not promised. Failures
invalidate the handle and retain errors. T gates, conditional H, general controls
and non-Pauli noise remain outside this admitted path. Public file reading and
streaming decoder composition remain future work.

## Keep the familiar converter file method

Use the existing native Circuit converter, then call
`converter.convert_file(measurements_filepath=..., measurements_format="b8",
detection_events_filepath=..., detection_events_format="b8")`.
`python -m examples.qec.measurement_file_conversion` is a complete runnable
repeated-classical-slot example: three writes retain three occurrence columns,
and fixed records 010 produce detectors 10 and observable 1.

Stim users keep the keyword names and defaults. Six input formats are accepted;
selected PTB64 outputs and sweep files are refused in this bounded native route.
Unlike detector `sample_write()`, this converter permits appended observables
and a separate observable file together. `Path` inputs normalize to strings at
the SDK boundary. No complete file is turned into a Python shot tensor.
See the [API](../reference/API.md#convert-a-measurement-file-to-detector-and-logical-files)
and [approved scope](../development/QEC_MEASUREMENT_FILE_CONVERSION_PROPOSAL.md).

## Captured records and batch decoding candidate

Use the familiar `read_shot_data_file` name from
`flagquantum.qec.execution.shot_data`. Results are owned Torch bool/uint8 tensors
instead of NumPy arrays. The new `iter_shot_data_file` is a FlagQuantum extension
for the embedded build, not an upstream Stim API. Its independently packed
detectors can feed the existing `PyMatchingDecoder.decode_packed_batch`.
See [shot-file decoding](../../examples/qec/shot_file_decoding.py) and the
[approved contract](../development/QEC_SHOT_DATA_READING_PROPOSAL.md). Standard
SDK builds support eager reads only. Zero-column records and arbitrary-data
write_shot_data_file remain explicit gaps. This candidate is not full parity.
