# Quantum error correction

Build experiments that connect syndrome extraction, decoding, correction, and
logical-result analysis. The long-term direction is a complete QEC workflow
for fault-tolerant quantum computing research, including logical operations
and hardware feedback.

QEC owns codes, decoder semantics, detection events, and Pauli frames. It
composes Compiler control flow, Runtime feedback, Simulation kernels, Noise
models, and Remote hardware interfaces.

## Start with a memory experiment

This experimental local example uses the three-data-qubit repetition-code
profile and injects one X error before syndrome extraction:

```python
from flagquantum.qec import ErrorEvent, ErrorSchedule, run_repetition_memory_experiment

result = run_repetition_memory_experiment(
    error_schedule=ErrorSchedule((ErrorEvent(round_index=0, wire=1),)),
    rounds=3,
    shots=16,
    seed=0,
)
print(result.logical_error_rate)
```

## Declare a code and its detection layout

`codes.py` holds the code records and `circuit.py` turns one into a memory
experiment. This is the code-independent layer beside the frozen repetition
profile:

```python
from flagquantum.qec import RotatedSurfaceCode, build_memory_circuit

memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
print(len(memory.detectors), len(memory.observables))
```

A detector is a measurement parity that is deterministic in the noiseless
circuit. Because both the initial state and the terminal data readout are in the
Z basis, a Z-type check declares a detector in every round plus one terminal
detector, while an X-type check declares one for every round after the first. A
code whose declared logical observable is not Z-type is refused rather than
measured under premises that do not hold for it.

## Decode a detection-event syndrome

`decoding_graph.py` turns a detector error model into a weighted graph and
`matching.py` decodes a syndrome on it. This is the code-independent decoder: it
reads the model's mechanisms and nothing about the code that produced them.

```python
from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
)

model = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=3),
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
)
decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
sample = model.dem_sampling(shots=1, seed=0)
syndrome = tuple(int(i) for i in sample.detectors[0].nonzero().flatten())
print(decoder.decode(syndrome).observables)
```

An edge weight is `log((1 - p) / p)`, a mechanism that flips one detector becomes
an edge to a boundary node, and the decoder's prediction is the exclusive-or of
the selected mechanisms' observable labels. A mechanism that flips three or more
detectors is a hyperedge and is refused with a stated reason rather than
projected onto a pair. The matcher enumerates the ways to pair the defective
detectors, so it refuses a syndrome larger than its defect budget instead of
returning a pairing that only looks cheapest.

One fault stated twice is a second refusal, and merging is the repair. A model
whose mechanisms repeat a signature — the same detectors *and* the same
observables — is one a matcher must not weight, because the graph would hold two
parallel edges and take the cheaper, so the decoder refuses it and names the
merge that resolves it:

```python
from flagquantum.qec import DetectorErrorModel

split = DetectorErrorModel.from_stim_text(
    "error(0.01) D0 L0\n"
    "error(0.01) D0 L0\n"
    "detector D0\n"
    "logical_observable L0\n"
)
print(split.mechanisms_are_unique())  # False
whole = split.merge_duplicate_mechanisms()  # DemMergeRule.INDEPENDENT_PARITY
print(whole.num_errors, whole.errors[0].probability)  # 1 0.0198
```

`merge_duplicate_mechanisms(rule=...)` gives every shared signature one prior.
`DemMergeRule.INDEPENDENT_PARITY`, the default, is the probability that an odd
number of the group fires; `DemMergeRule.CLAMPED_LINEAR_SUM` adds them and clamps
at one, for a caller whose mechanisms exclude each other. The parity rule is
exact rather than tidy: a detector's rate is a product of `1 - 2p` factors over
the mechanisms that touch it, so the combined prior of a group is the single `p`
whose factor is that group's product and every rate is unchanged.
`mechanisms_are_unique()` and `require_unique_mechanisms()` are the predicate and
the refusal.

## Cross-check the matcher against PyMatching

The matcher is the authority, and its evidence cannot come from itself. Brute
force over enumerable syndromes is independent of the implementation but not of
the detector error model both read, so `adapters.py` adds the one check that is:
a second decoder, over the same decoding graph, behind the optional `pymatching`
extra.

```bash
pip install 'flagquantum[pymatching]'
```

```python
from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RepetitionCode,
    build_memory_circuit,
)

model = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=3),
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
)
ours = MinimumWeightMatchingDecoder.from_detector_error_model(model)
theirs = PyMatchingDecoder.from_detector_error_model(model)
print(ours.decode([0, 1]).weight == theirs.decode([0, 1]).weight)
```

The two agree on the cheapest weight of every syndrome and on the observables
wherever the cheapest explanation is unique, and a tie is uncomparable rather
than a disagreement: two explanations of equal weight can flip different
observables, and no tie-breaking rule is more correct than another. The
comparison is also limited by PyMatching's arithmetic, which is narrower than
this package's — it reports `3.9020747171643912` for a mechanism stated here as
`3.9020746947749574` — so a caller compares weights with a tolerance of about one
single-precision rounding per selected mechanism. Ties and tolerance are stated
in the module docstring rather than hidden in a test.

Translation is refused with a stated reason in two places, because a translation
that is not faithful would make the comparison measure the translation. A
detector pair carrying two mechanisms cannot be merged without losing a logical
label, so it is refused rather than handed to PyMatching's `independent`
strategy; and a detector that no mechanism flips is refused because PyMatching
infers its detector count from its edges, so the syndrome vector would be
renumbered. Both graphs are ones the matcher itself answers, so the cross-check's
domain is narrower than the authority's and never the reverse.

## Build a model from matrices, without a circuit

A code whose checks and logical operators are known as matrices needs no gadget
and no wire layout. `css_code_matrices` reads a code record into one
`CssCodeMatrices` record carrying all four CSS blocks, and `from_code_matrices`
derives the mechanisms from it:

```python
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    css_code_matrices,
)

matrices = css_code_matrices(RepetitionCode(distance=3))
model = DetectorErrorModel.from_code_matrices(
    matrices,
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    num_rounds=3,
)
print(model.num_detectors, model.num_errors)  # 6 15
```

`hz[k, q]` is one when a Z-type check `k` sees data qubit `q`, `hx` uses the
same convention for X-type checks, and `lz` and `lx` use it for the logical
operators of each type. The record reads all four, so a code with both readouts
reaches a model with an X detector band and an `lx` observable block as well as a
Z one, and each of the three single-qubit Pauli fault families contributes its
own row. A block may be omitted, and then it declares nothing. A matrix that is
not binary, is not two-dimensional, or does not index the same data qubits as
another non-empty block is refused rather than coerced, because a real-valued
matrix is a rate description and rounding it would decide which checks a fault
triggers. A logical observable that is neither pure X nor pure Z is refused with
its index named rather than read as one of the two, so a code the CSS record
cannot describe says so instead of silently losing a fault family.

This route is the code-capacity experiment, so its detector geometry is not the
memory circuit's. A fault in round `r` reaches the detector band of round `r` and
the band of round `r + 1`, and the final round has no band after it, so the
detector count is `num_rounds * num_checks` with no terminal readout — where a
memory circuit gains a terminal detector per Z-type check because it measures its
data qubits. Both geometries are pinned to their own route and neither stands for
the other. The rates are read against these matrices: the per-qubit vectors are
indexed by column, in the code's own `data_wires` order, and the per-check vector
by row, with the Z-type checks first.

## Give one location its own rate

`PhenomenologicalNoise` states a rate per fault family and, optionally, a rate per
element. The four scalars are `data_flip` (an X fault), `phase_flip` (a Z fault),
`both_flip` (a Y fault) and `measurement_flip`, which are upstream `CssNoise`'s
`px`, `pz`, `py` and `pm`. The four vectors are `data_flip_per_qubit`,
`phase_flip_per_qubit`, `both_flip_per_qubit` and `measurement_flip_per_check`,
which are upstream's `px_per_qubit`, `pz_per_qubit`, `py_per_qubit` and
`pm_per_check`:

```python
from flagquantum.qec import DetectorErrorModel, PhenomenologicalNoise, RepetitionCode

noise = PhenomenologicalNoise(
    data_flip=0.02,
    data_flip_per_qubit=(0.0, 0.05, 0.02),
)
model = DetectorErrorModel.from_code(
    RepetitionCode(distance=3), noise=noise, num_rounds=2
)
print(sorted({error.probability for error in model.errors}))
```

The override is wholesale: a stated vector replaces its scalar for **every**
element rather than mixing with it, which is why a vector has to name every data
qubit (or every check) and a short one is refused naming both counts instead of
being partially applied. An element whose effective rate is zero states a
location that cannot fire, so it enumerates no mechanism at all: in the example
above the first data qubit contributes nothing while its neighbours contribute
their own rates, and the sampler places no channel for it. A per-check vector is
indexed by matrix row — Z-type checks first, then X-type checks — while a code is
free to declare its checks in any order, so the declaration order and the vector's
order are related by one named translation rather than by a convention each route
re-states.

This is a description of noise locations, not a `NoiseModel`, and the sampler
places only the two families the engine has a channel for: a data flip at a round
boundary and a measurement flip at a check's readout. The phase and Y data rates
are read by the construction routes, which read matrices and supports rather than
executing a program.

## Say that two mechanisms are alternatives

Every mechanism in a model is an independent fault unless it carries an
`error_id`, and mechanisms sharing one are mutually exclusive: at most one of
them fires in a shot. This is the one statement the parity matrices cannot carry,
because two columns of a parity matrix are independent by construction.
`dem_alternatives.py` owns its arithmetic:

```python
from flagquantum.qec import DemError, DetectorErrorModel

model = DetectorErrorModel(
    num_detectors=3,
    num_observables=1,
    errors=(
        DemError(probability=0.1, detectors=(0, 1), error_id=0),
        DemError(probability=0.2, detectors=(0, 1), error_id=0),
    ),
)
print(model.detector_rates()[0])  # 0.30000000000000004, not the 0.26 of two
                                  # independent mechanisms
print(model.error_ids)            # (0, 0): upstream's parallel id vector
```

The members of a group are disjoint pieces of one shot. Each keeps the
probability it states, the left-over mass is the group firing none of them, a
target's rate over the group is the sum of the members touching it rather than
their parity, and `dem_sampling` draws once per group and lets at most one member
fire. A group summing to exactly one is admitted — it fires every shot — and a
lone id excludes nothing, so a model with no ids is this arithmetic's empty case
and behaves exactly as it did before the field existed. A group whose
probabilities sum *above* one is refused rather than renormalized, because
renormalizing would change every rate the caller read.

The statement is about the distribution and not about the signature, so the three
operations that assume independence refuse such a model and name the ids instead
of dropping the structure: `to_stim_text()` (the format reads every error
instruction as independent), `merge_duplicate_mechanisms()` (both rules combine a
group by assuming independence), and `MinimumWeightMatchingDecoder` (one weight
per mechanism is the weight of a fault that fires alone, while a group is one
fault whose weight is the group's own negative log-likelihood). Folding a group
under its exclusivity into the single mechanism a matcher can weigh is not
implemented here; that gap is recorded against the `canonicalize_for_rounds`
family in `contracts/qec-cudaq-alignment-checklist.toml`.

## Sample the circuit itself, not the model

`sampling.py` samples the same experiment from the circuit rather than from the
model, so a logical-failure rate no longer depends on the statevector ceiling that
bounds construction. It shares the code record and the noise record with the
model above and nothing else: the model derives each mechanism's signature from
the source, this module derives each mechanism's instruction position from the
lowered program.

```python
from flagquantum.qec import (
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    sample_memory_circuit,
)

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
sample = sample_memory_circuit(
    memory,
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
    shots=1000,
    seed=0,
)
print(sample.detectors.shape, sample.observables.shape)
```

A data flip is placed at the round boundary *before* the round's first gate, so it
opens the frame that round's detectors compare against, and a measurement flip is
placed immediately *before* the readout of the check it corrupts. Neither position
exists in the source program, whose bounded hybrid capture refuses a channel call
outright, so both are derived from the lowered program — and the program must
lower to `rounds` identical blocks measuring each check once in the code's
declared order. A program that lowers to anything else, a channel that is not the
bit-flip pair, and a lowered measurement node are each refused with a stated
reason rather than sampled under an attribution that may be wrong.

Because the two samplers are separate code paths, a rate from this one is
circuit-sampled and a rate from `dem_sampling` is model-sampled; the tests pin the
two to each other rather than letting either stand for the other.

## Hand a decoder both halves

A model says which mechanisms a decoder can see. It does not say where in a shot
each detector's parity is read, and a matcher is handed a syndrome over raw
measurements, so it needs both. `decoder_context_from_memory_circuit` builds the
model once and returns a context whose components pair it with the maps:

```python
from flagquantum.qec import (
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    decoder_context_from_memory_circuit,
)

memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
context = decoder_context_from_memory_circuit(
    memory, noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
)
inputs = context.z_component()
print(context.num_measurements(), inputs.measurement_to_detectors.rows[0])
print(inputs.measurement_to_detectors.dense().shape)
```

`full_component`, `x_component` and `z_component` each return a `DecoderInputs`:
the model read over the detectors of one basis, plus the measurement-to-detector
and measurement-to-observable maps for it. A `MeasurementMap` is one row per
detector or observable holding the measurements whose parity it is, and it
projects to both forms a caller may want — `dense()` for the
`(rows, measurements)` orientation upstream stores, and `flattened()` for the
`-1`-terminated sparse vector a realtime decoder configuration takes, where the
terminator is what keeps a row that reads nothing visible.

Two things are refused rather than approximated. A row may not name one
measurement twice, because two reads of one measurement cancel rather than add. A
component of a model that states error ids is refused, because projecting a group
of alternatives onto one basis would either drop the correlation or merge two of
its members. A basis the experiment declares no detector for is refused rather
than returned empty.

The split is a reading of the model, not a second construction: every detector is
carried by the check whose ancilla it reads, so the Z component carries the
terminal detectors and the union of the two components is the model as built.
That is also why the numbering the maps use is pinned to the sampler's by a test
rather than by a shared constant — the sampler derives its record columns from the
lowered program, this module derives them from the circuit's own declaration, and
a decoder fed a sampled syndrome has to be matching detectors against the
measurements that actually compose them.

## Come from Stim

A Stim user already has the circuit layer. What this package offers is the half
that sits after it, and the interface between the two is the detector error model
text: nothing under `flagquantum/` imports `stim`, so the `stim` extra is needed
only by the code that generates circuits.

```python
import stim
from flagquantum.qec import DetectorErrorModel, get_decoder

circuit = stim.Circuit.generated(
    "repetition_code:memory",
    distance=3,
    rounds=3,
    before_round_data_depolarization=0.05,
)
text = str(circuit.detector_error_model().flattened())
model = DetectorErrorModel.from_stim_text(text)
decoder = get_decoder("minimum_weight_matching", model.merge_duplicate_mechanisms())
print(model.num_detectors, len(model.errors), decoder is not None)
```

`str(circuit.detector_error_model())` on a multi-round circuit carries a `repeat`
block that `from_stim_text` refuses by name instead of expanding, so
`.flattened()` is the whole repair. The other refusals, the two readings of the
`^` separator and the observable marginal each one implies, and what this route
does not cover are in
[Migrating a Stim workflow](../../docs/guides/STIM_USER_MIGRATION.md); that guide
is executed fence by fence by `tests/qec/test_stim_user_migration.py`, and
[`examples/qec/stim_user_migration.py`](../../examples/qec/stim_user_migration.py)
is the runnable form of the same route.

## Change and verify

Use [repetition.py](repetition.py) for experiment composition,
[decoders.py](decoders.py) for decoding, [decoding_graph.py](decoding_graph.py)
and [matching.py](matching.py) for the detector-error-model decoder,
[adapters.py](adapters.py) for the PyMatching cross-check,
[sampling.py](sampling.py) for sampling detection events from a memory circuit,
[noise.py](noise.py) for code-specific noise profiles, [codes.py](codes.py) for
code records, [circuit.py](circuit.py) for detector and observable layouts, and
[types.py](types.py) for records. Run from the repository root:

```bash
python -m pytest tests/qec -q
python tools/check_architecture.py
```

Check syndrome histories, correction actions, and final logical outcomes for
known injected errors. Decoder changes must also cover readout faults and
errors near the final round. Logical suppression or threshold claims require
separate statistical and scaling evidence.

[Feedback modes and model boundaries](IMPLEMENTATION.md) explain the supported
profiles. This reference experiment does not establish general FTQC or
hard-real-time hardware feedback.
