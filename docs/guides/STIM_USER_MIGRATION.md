# Migrating a Stim workflow to FlagQuantum

FlagQuantum replaces the half of a Stim-based QEC study that sits **after**
circuit generation: reading a detector error model, sampling it, and decoding a
syndrome. It does not replace Stim's circuit layer, and it does not ask you to
stop using it.

The interface between the two halves is the detector error model text, which
Stim already writes and FlagQuantum reads. No Stim object crosses the boundary,
no circuit is re-simulated, and nothing under `flagquantum/` imports `stim` — the
reader, the model sampler, and both in-tree decoders are Stim-free, so the
`stim` extra is only needed by the half that generates circuits.

| Stim | FlagQuantum |
| --- | --- |
| `stim.Circuit.generated` | keep it: generate circuits with Stim |
| `circuit.detector_error_model(...)` | keep it: write the model text with Stim |
| `dem_from_stim_text(text)` | `DetectorErrorModel.from_stim_text(text)` |
| `dem.compile_sampler(...).sample(...)` | `DetectorErrorModel.dem_sampling(...)` |
| `circuit.compile_detector_sampler(...)` | `DetectorErrorModel.dem_sampling(...)` for a model, or keep circuit sampling |
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
try:
    DetectorErrorModel.from_stim_text(raw_text)
except ValueError as error:
    print(error)
    # repeat blocks are not supported: expand the block into the instructions it repeats  -- the reader's refusal
```

`from_stim_text` refuses what its own record cannot hold rather than dropping
it, and a `repeat` block is the first thing a multi-round circuit hits. Stim
already owns the expansion, so the caller spends one call on it:

```python
flattened = DetectorErrorModel.from_stim_text(
    str(detector_error_model.flattened())
)
print(
    flattened.num_detectors,
    flattened.num_observables,
    len(flattened.errors),
    flattened.mechanisms_are_unique(),
)
# 42 1 141 False  -- flattened, read, and 141 mechanisms of which two share a signature
```

The four refusals a migrating script meets are worth knowing by name, because
each one is a repair rather than a lost feature:

| Text | Refusal | Repair |
| --- | --- | --- |
| `repeat 3 { ... }` | `repeat blocks are not supported: ...` | `stim.DetectorErrorModel.flattened()` |
| `error(0.01) D0  # note` | `error targets must be D or L indices, not '#'` | strip comments before reading |
| `detector D1` with no `detector D0` | `detector declarations must be consecutive from D0` | declare every index, or use `shift_detectors` |
| `X_ERROR(0.01) 0` | `unsupported stim instruction 'X_ERROR(0.01) 0'` | keep circuit instructions in Stim; this reader takes a detector error model |

## One fault stated twice

`mechanisms_are_unique()` is `False` above, and that is not a defect in the
reading: two different lines can erase to the same detectors *and* the same
observables. A matcher that sees only parity cannot weight two parallel edges
for one signature, so it refuses the model and names the merge:

```python
from flagquantum.qec import get_decoder

try:
    get_decoder("minimum_weight_matching", flattened)
except ValueError as error:
    print(error)
    # mechanisms 7 and 8 both flip D2 D4; merge_duplicate_mechanisms() states one prior for a shared signature and is what a decoder needs  -- the matcher's refusal
model = flattened.merge_duplicate_mechanisms()
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
print(round(float(combined.observable_rates()[0]), 6))
# 0.146392  -- the default reading against Stim's own sampler
print(round(float(graphlike.observable_rates()[0]), 6))
# 0.189091  -- the expanded reading, 29% above it
print(round(float(reference[:, 0].mean()), 6))
# 0.146475  -- Stim's sampler of the same detector error model text
```

Stim's sampler of the *decomposed* text reports the same 0.146475, so the
disagreement is in the reading and not in the sampling. The two readings are
therefore a choice rather than a preference:

- **Decoding a syndrome.** Decompose and merge. A minimum-weight matcher needs a
  graphlike model, `get_decoder("minimum_weight_matching", graphlike)` or
  `get_decoder("pymatching", graphlike)` both accept it, and they agree shot for
  shot.
- **A threshold, a marginal, or any rate.** Use the default reading, which is the
  model Stim wrote, and decode it with a decoder that carries hyperedges.

```python
from flagquantum.qec import BeliefPropagationOsdDecoder

events, observables = surface.compile_detector_sampler(seed=9).sample(
    shots=2000, separate_observables=True
)
syndromes = events.astype(bool)
answers = observables[:, 0].astype(bool)
hyperedge_decoder = BeliefPropagationOsdDecoder(combined)
matcher = get_decoder("minimum_weight_matching", graphlike)
cross_check = get_decoder("pymatching", graphlike)


def predicted(decoder):
    return np.array(
        [bool(decoder.decode(np.flatnonzero(row)).observables) for row in syndromes]
    )


matching = predicted(matcher)
crossed = predicted(cross_check)
hyperedges = predicted(hyperedge_decoder)
print(
    round(float(np.mean(matching != answers)), 4),
    round(float(np.mean(crossed != answers)), 4),
)
# 0.0375 0.0375  -- the rate alone does not separate these routes
print(float(np.mean(matching == crossed)), float(np.mean(matching == hyperedges)))
# 1.0 0.983  -- the per-shot agreement does
```

At this shot count all three decoders report the same logical error rate to four
decimals, so a rate comparison cannot choose between the two readings. The
per-shot agreement can: the in-tree matcher and the PyMatching cross-check pick
the same logical class on every shot, while belief propagation with ordered
statistics differs from the matcher on 1.7% of shots. What actually separates the
readings is the marginal in the table above, which is why a migrated script's
correctness check belongs on the model and not only on the decoded rate.

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
- **No block expansion.** A `repeat` block is refused, not expanded; the repair
  is `flattened()` on the Stim side.
- **No comment handling.** Comments are not part of the format this reader
  accepts, though Stim's own writer does not emit them into a detector error
  model.
- **Belief propagation with ordered statistics is not in the decoder registry.**
  `BeliefPropagationOsdDecoder` is constructed directly as above, because
  `register_decoder` requires a `from_detector_error_model` classmethod the class
  does not carry, so `decoder_names()` names the in-tree matcher and the
  PyMatching cross-check only. A hyperedge model therefore has no registry name
  today.
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
