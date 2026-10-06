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
    error_schedule=ErrorSchedule((ErrorEvent(round_index=0, qubit=1),)),
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

A record can also be reached by name rather than by class, which is the shape
CUDA-Q QEC's own `get_code(name, options)` has. `get_code` builds the record,
`code_names` lists the names, and `register_code` puts one there:

```python
from flagquantum.qec import code_names, get_code

print(code_names())
patch = get_code("rotated_surface", distance=3)
patch.num_ancilla_qubits, patch.num_ancilla_x_qubits, patch.num_ancilla_z_qubits
patch.num_x_stabilizers, patch.num_z_stabilizers
```

A record reports its ancillas as a total and as two bands, and the bands are
derived from its checks by `ancilla_bands` rather than declared beside them, so
the two cannot drift: a check's ancilla measures exactly the basis its
stabilizer's type fixes. An ancilla measuring neither basis -- a flag, or an idle
ancilla -- is in neither band, so the bands need not cover the total, and
`build_memory_circuit` refuses a record whose stated counts disagree with the
bands its own checks define.

The two stabilizer counts are the two further counts CUDA-Q QEC's `code` record
declares as pure virtuals, which makes five counts a caller reads. In each of the
three code classes upstream ships one ancilla measures one stabilizer, so a
stabilizer count and the band count of the same basis are one quantity; here
those two accessors read the bands rather than deriving the same number a second
time.

A detector is a measurement parity that is deterministic in the noiseless
circuit, and which parity that is is stated relative to the experiment's readout
basis. Under the default Z basis a Z-type check declares a detector in every round
plus one terminal detector, while an X-type check declares one for every round
after the first; under an X basis the two classes swap, because the preparation
and the terminal readout are then in that basis:

```python
from flagquantum.qec import SteaneCode, build_memory_circuit

z_memory = build_memory_circuit(SteaneCode(), rounds=3)
x_memory = build_memory_circuit(SteaneCode(), rounds=3, readout_basis="x")
print(z_memory.readout_basis, x_memory.readout_basis)
print(x_memory.observables.observables[0].pauli)
```

The basis is a field of the record rather than a second entry point, and it is
checked against the observables the code declares rather than assumed from them: a
code that declares no logical observable in the requested basis is refused, rather
than measured under premises that do not hold for it. CUDA-Q QEC derives the same
basis from the preparation kernel it is handed and offers `x_` and `z_` model
entry points beside the full one; here one record states it and one builder reads
it.

## Decode a detection-event syndrome

`decoding_graph.py` turns a detector error model into a weighted graph and
`matching.py` decodes a syndrome on it. This is the code-independent decoder: it
reads the model's mechanisms and nothing about the code that produced them.
`belief_propagation.py` is the second decoder over the same model, for the
mechanisms a matcher has no edge for.

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

## Decode a model the matcher refuses

A matcher needs one edge per mechanism, so a mechanism that flips three or more
detectors is a hyperedge and `matching.py` refuses it. `belief_propagation.py`
reads exactly that model: a mechanism is one variable, a detector is one check,
and a mechanism touching three detectors joins three checks, so the graph is the
model as written rather than a projection of it.

```python
from flagquantum.qec import BeliefPropagationDecoder, DetectorErrorModel

model = DetectorErrorModel.from_stim_text(
    "error(0.01) D0 D1 D2\n"
    "error(0.01) D0 D1\n"
    "error(0.01) D1 D2\n"
    "detector D0\n"
    "detector D1\n"
    "detector D2\n"
)
result = BeliefPropagationDecoder.from_detector_error_model(model).decode((0, 1, 2))
print(result.converged, result.mechanisms, result.observables, result.weight)
```

`converged` is false exactly when the exchange did not settle and an
ordered-statistics solve over a greedily chosen information set answered instead,
so the flag is a fact about the path taken rather than a quality score. Belief
propagation is exact on a factor graph that is a tree and approximate off one, so
a loopy model can settle on an explanation that is heavier than the cheapest; a
caller who needs the cheapest explanation on a graphlike model wants the matcher.
The decoder refuses a model that states its mechanisms are alternatives, a
mechanism of rate zero, and a syndrome that no set of mechanisms can produce,
rather than approximating any of the three.

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
and no qubit layout. `css_code_matrices` reads a code record into one
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
memory circuit gains a terminal detector per check of the readout basis it is
built in, because it measures its data qubits. Both geometries are pinned to their own route and neither stands for
the other. The rates are read against these matrices: the per-qubit vectors are
indexed by column, in the code's own `data_wires` order, and the per-check vector
by row, with the Z-type checks first.

## Build a code record from matrices

The records above are declared: a class states its layout and builds its checks
from it. A caller who already holds the four CSS blocks does not need a class for
them. `CssCode` takes a `CssCodeMatrices` and a distance, and answers the same
members `StabilizerCode` declares, so everything downstream -- the ancilla bands,
the memory-circuit source, the detector and observable layouts, `css_code_matrices`
itself -- takes a matrix-built record on the same terms as a declared one:

```python
import torch
from flagquantum.qec import CssCode, CssCodeMatrices, build_memory_circuit

matrices = CssCodeMatrices(
    hz=torch.tensor([[1, 1, 0], [0, 1, 1]]),
    lz=torch.tensor([[1, 1, 1]]),
)
code = CssCode(matrices=matrices, distance=3)
memory = build_memory_circuit(code, rounds=3)
print(len(code.checks), code.num_ancilla_z_qubits, len(memory.detectors))  # 2 2 8
```

Column `q` of every block is data qubit `q`, the Z-type checks take the ancillas
in `hz`'s row order and the X-type checks the ones after them, and the CNOT
direction of each check is the one its type fixes, exactly as a declared check
states it. The Shor code is the case that shows this is not a second way to write
the three declared families: it is nine data qubits with six Z-type checks, two
X-type checks and a weight-three logical Z, no class here declares it, and it
reaches a memory circuit and a detector error model from four blocks alone.

The distance is stated rather than derived, which is CUDA-Q QEC's own division:
its code record declares no distance accessor, its own factories read one out of
the options they are built with and throw when it is absent, and its matrix record
carries no distance to read. A minimum-weight-codeword search is exponential in
general, so deriving the distance here would refuse exactly the large matrices
this route exists for. Instead the matrices are held to the distance they are
given in the one direction a stated logical operator can prove: an operator of
weight *w* bounds the distance above by *w*, so a record claiming a distance
larger than the lightest operator it states is refused and one that understates
its distance is accepted.

The four blocks are also held to the code algebra a matrix record normally leaves
unchecked. Two check blocks that do not commute, a logical operator meeting the
opposite basis' checks on an odd number of qubits, a logical operator that is a
product of its own type's checks, two dependent rows in one logical block, and a
check or logical row with no support are each refused with the row and the reason
named. CUDA-Q QEC validates a common column count, the rate-vector lengths and the
probability range, and none of this, so the checks here are a deliberate
strengthening rather than a match.

Two things the route does not carry. It is not registered by name -- the identity
of a matrix-built record is the matrix, so `get_code` still reaches the three
declared records -- and it is CSS-shaped, so an arbitrary non-CSS stabilizer list
still has no route in.

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

This is a description of noise locations stated at round boundaries and check
readouts, and every family it states is a location on **both** routes. The
sampler executes one channel — the engine's single-wire bit flip — so a Z or Y
location is that channel conjugated by the Clifford that turns a bit flip into
the Pauli the family names: `h` around it for `phase_flip`, `sdg`/`s` around it
for `both_flip`. That is one draw at that family's rate, not a pair of
independent bit flips, and the noiseless circuit is untouched because the pair
cancels when the channel does not fire. A measurement flip is the bit flip
itself, since a readout is flipped in the basis it is read in.

## Bind a channel to a gate

`flagquantum.noise.NoiseModel` is the second record the three entry points accept,
and which record you state is what selects the grammar. It names gates rather than
round boundaries, and its fault follows the gate its rule matched, once per round
that gate appears in. That is where upstream CUDA-Q places a channel bound to a
named gate: the channel acts on the state the gate leaves behind.

```python
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec import DetectorErrorModel, RotatedSurfaceCode, build_memory_circuit

circuit = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
noise = NoiseModel().add("h", bit_flip_channel(0.01)).add("cx", bit_flip_channel(0.01))
model = DetectorErrorModel.from_memory_circuit(circuit, noise=noise)
print(model.num_detectors, model.num_errors)
```

`sample_memory_circuit(circuit, noise=noise, shots=..., seed=...)` samples the
same record, and the model and the sample read one mechanism list and one
lowering, so they cannot disagree about which instruction a fault sits at. A rule
whose gate the circuit does not execute contributes no location: a noise model is
stated over a gate set, and a gate outside it is not part of the experiment. A
rule that names a gate the circuit executes many times contributes one location
per occurrence, so the round is read off the lowered program rather than stated in
the record.

This grammar places a single-qubit Pauli fault and nothing else, so four things
are refused by name rather than approximated, on the model route and the sampling
route alike. A channel that is not one Pauli fault — a depolarizing or damping
channel, whose Kraus set is a mixture rather than an identity and a flip. A rule
naming `measure` or `reset`, because a fault is attached to the gate a rule names
and a readout fault's position is the check it corrupts, which is the
`measurement_flip` field of a `PhenomenologicalNoise`. A one-qubit channel bound
to two wires, which is a correlated fault rather than two independent ones. And a
fault that would follow the program's last instruction, which is stated rather
than dropped so the model cannot be silently thinner than the record.

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
print(model.error_rates)          # (0.1, 0.2): one entry per column, each
                                  # mechanism's own rate, not the 0.3 marginal
```

The members of a group are disjoint pieces of one shot. Each keeps the
probability it states, the left-over mass is the group firing none of them, a
target's rate over the group is the sum of the members touching it rather than
their parity, and `dem_sampling` draws once per group and lets at most one member
fire. A group summing to exactly one is admitted — it fires every shot — and a
lone id excludes nothing, so a model with no ids is this arithmetic's empty case
and behaves exactly as it did before the field existed. A group whose
probabilities sum *above* one is refused rather than renormalized, because
renormalizing would change every rate the caller read. `error_rates` is the
column view beside the matrices rather than the folded one: entry `i` and column
`i` of `detector_error_matrix` are one mechanism because both are read from the
model's own normalized mechanism order, and each entry is the rate its own
mechanism states, so the folding stays in `detector_rates()` and
`observable_rates()`.

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
placed immediately *before* the readout of the check it corrupts. A Z or Y data
fault is placed at that same boundary, as the one channel the engine has wrapped
in the conjugation that names the Pauli. Neither position exists in the source
program, whose bounded hybrid capture refuses a channel call outright, so both are
derived from the lowered program — and the program must lower to the experiment's
readout rotation around `rounds` identical blocks measuring each check once in the
code's declared order. The rotation belongs to the experiment rather than to a
round, so the plan states its length as one offset beside the round block instead
of folding the two together; it is one gate per data qubit at each end, not a
block measuring every check. A program that lowers to anything else, a channel
that is not the bit-flip pair, and a lowered measurement node are each refused
with a stated reason rather than sampled under an attribution that may be wrong.

The same sampler reads either basis of a code that declares a logical observable
of each type, because the basis is a field of the record rather than a second
entry point:

```python
from flagquantum.qec import (
    PhenomenologicalNoise,
    SteaneCode,
    build_memory_circuit,
    sample_memory_circuit,
)

defect = PhenomenologicalNoise(data_flip=0.01, phase_flip=0.02, measurement_flip=0.03)
x_memory = build_memory_circuit(SteaneCode(), rounds=3, readout_basis="x")
x_sample = sample_memory_circuit(x_memory, noise=defect, shots=1000, seed=0)
print(x_memory.readout_basis, x_sample.detectors.shape)
```

Because the two samplers are separate code paths, a rate from this one is
circuit-sampled and a rate from `dem_sampling` is model-sampled; the tests pin the
two to each other rather than letting either stand for the other.

## Read a recorded bit, not just a parity

A detection event is a parity of recorded bits. Sometimes the bit itself is what a
caller wants, and there is exactly one kind of bit that no layout names: round zero
of a patch measures the check class its preparation does not stabilize — an X-type
check under the all-zero preparation, a Z-type check under the `|+>` one — whose
outcome is not deterministic, so no detector may name it, and it is still measured
and still recorded. `measurement_refs` names every recorded bit and
`sample_memory_measurements` returns them:

```python
from flagquantum.qec import (
    MeasurementRef,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    build_memory_circuit,
    sample_memory_measurements,
)

memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=2)
samples = sample_memory_measurements(
    memory,
    noise=PhenomenologicalNoise(measurement_flip=0.02),
    shots=1000,
    seed=0,
)
Z_ANC = 9  # the code's first Z-type check; see RotatedSurfaceCode(...).checks
print(samples.outcome(MeasurementRef(1, Z_ANC)).shape)
print(samples.vector([MeasurementRef(0, Z_ANC), MeasurementRef(1, Z_ANC)]).shape)
print(samples.integer([MeasurementRef(0, Z_ANC), MeasurementRef(1, Z_ANC)]).max())
```

A handle is a value in the same sense CUDA-Q's is: `outcome` reads one handle,
`vector` reads a chosen sub-vector as booleans in the order asked for, and
`integer` packs that same order into one integer per shot with the first handle
least significant. Reading a gap in a bit stream does not need a parity and does
not need a column index, and a detector's parity recomputed from the handles it
names is the detector's parity — a test asserts that per detector and per
observable, so the two readings cannot drift apart.

What this layer is not is a statement about which handles are deterministic. A
noiseless run reads a Z-type check's ancilla as 0 in every round, an X-type
check's ancilla as unbiased, and an individual terminal data wire as unbiased, while
every detector and every observable reads 0. So a rate read off a single handle is
a rate read off the state and not a channel rate, and a detector is a parity and not
a measurement. The absence this leaves standing is the kernel annotation form
itself: `detector`, `detectors` and `logical_observable` as kernel calls are not
implemented here, and identity is set in a layout record beside the source rather
than in the kernel body.

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
That is also why the numbering the maps use is checked against the sampler's
rather than shared with it — the sampler derives its record columns from the
lowered program, this module derives them from the circuit's own declaration, and
`measurement_refs` states the same layout a third time so that a handle names a bit
whichever of the two a caller came from. A decoder fed a sampled syndrome has to be
matching detectors against the measurements that actually compose them, and that is
now something a test can recompute from handles rather than only assume.

## Cut a model into rounds

A long experiment's model is one object, but a decoder that reads it a few rounds
at a time needs those rounds to be nameable. `ChunkLayout` states how a model's
detectors divide into layers, `DemChunksSpec` asks for windows over them, and the
windows close back into the model they were cut from:

```python
from flagquantum.qec import (
    ChunkLayout,
    DemChunksSpec,
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    dem_chunks_from_spec,
    dem_close_all,
)

noise = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
layout = ChunkLayout.from_memory_circuit(model, memory)
chunks = dem_chunks_from_spec(DemChunksSpec(layout=layout, window=2))

for chunk in chunks:
    print(chunk.first_layer, chunk.last_layer, len(chunk.model.errors))
print(dem_close_all(chunks) == model)
```

A *layer* is the set of detectors with one round index, so `layout.widths` is the
round size in detector order and sums to the model's detector count. A *window*
spans a run of layers and shares exactly one layer with each neighbour: the
window's own bands are its leading boundary layer, its interior, and its trailing
boundary layer, and the stride is the window width minus one, which is what makes
the windows tile rather than overlap. `SeamId.prev_round` and `SeamId.next_round`
name the two boundaries a window carries, and a window carries only the ones it
has a neighbour for — the first window has no `prev_round`, the last no
`next_round`.

Ownership is what makes the windows a partition rather than a cover. A mechanism
spans two adjacent layers, so the first window holding it whole owns it, every
mechanism is owned once, and `dem_close_all(chunks) == model` is the invariant a
test asserts rather than assumes. Three things are refused instead of
approximated: a window of one layer, a window wider than the layout, and a layer
count that does not tile at the stride. A mechanism that no window holds whole is
refused rather than split — splitting it would state two weaker faults where the
model stated one — and a mechanism flipping no detector is refused because an
observable-only fault has no round to be placed in.

`dem_stitch` contracts the shared layer of two adjacent windows and lays it out
once, between their interiors, so a stitched pair spans one layer fewer than the
sum of its parts; `dem_stitch_all` folds that over a sequence and
`dem_stitch_merged` follows the stitch with `merge_duplicate_mechanisms` under a
stated rule. `dem_close` puts one window back at its own place in the model's own
numbering without renumbering from zero, so a window's detectors are the model's
detectors. A seam's rows carry the global detector index rather than a position
within the seam, which is why a stitch refuses two boundary bands that merely
have the same width.

## Slide a window over those rounds

Decoding the whole model at once needs the whole syndrome, which a realtime
decoder does not have. A sliding window decodes a few rounds at a time, commits
the part the next window cannot revise, and folds the committed corrections into
a logical-frame prediction. `SlidingWindowDecoder` is that decoder over the
windows above, and it is built from the decomposition rather than from the model,
because the windows are what it slides over:

```python
from flagquantum.qec import (
    BELIEF_PROPAGATION_NAME,
    SlidingWindowDecoder,
    dem_chunks_to_o_sparse,
    dem_chunks_to_pcm,
)

spec = DemChunksSpec(layout=layout, window=2)
decoder = SlidingWindowDecoder(spec, BELIEF_PROPAGATION_NAME)

print(decoder.num_windows)             # windows one block is decoded in
print(decoder.decode([]).observables)  # a whole block, in the model's numbering

# Or one round at a time, in that round's own numbering: a record comes back
# whenever a window closes and None while the window is still open.
for layer in range(layout.num_layers):
    record = decoder.decode_round([])
```

A *fault column* is a column of the model the windows decompose. Both sparse
projections index that one column space in the order `dem_close_all` states, so
`dem_chunks_to_pcm(chunks)` is the closed model's `detector_error_matrix()` read
one detector row at a time and `dem_chunks_to_o_sparse(chunks)` is its
`observables_flips_matrix()`, each in sparse form and neither built dense.

Each window gets its own inner decoder, built from that window's own model and
from nothing else, so no window is ever handed a row or a mechanism that is not
its own. Every window that closes returns a record; `complete` says whether the
block is finished and `converged` is the inner decoders' flags anded, or `None`
when none of them reports one. `faults` is the committed part of the correction
and `observables` is that same vector projected through the model's observable
matrix, so the two are one statement read twice rather than two answers free to
disagree.

An inner decoder must state its selected mechanisms as `mechanisms` in that
window's own column order, which is what `BeliefPropagationDecoder` reports,
because the next window's syndrome has their support removed and a record that
states only observables is an answer rather than a correction. The authority
matcher's record states its correction as graph edges instead, so it is not yet
usable as an inner decoder and a window built with it refuses at the first commit
rather than silently dropping the correction. A whole arbitrary detector subset is
also not a syndrome every window can explain — a window's residue is a subsystem
the window need not span — so a window that cannot explain its residue refuses,
the stream resets, and the next round starts a new block.

`dem_chunks_to_d_sparse(chunks)` is a third numbering again: it maps each
detector row onto the *measurement bits* of a flat `rounds * d` buffer, which is
what a memory experiment's raw measurements occupy. That map holds when every
round is the same `d` detectors wide, which is the precondition its arithmetic
states; a surface code's boundary rounds are half that wide, so the map refuses
that geometry by name and the circuit-derived `MeasurementMap` is the map that
covers it instead.

## Change and verify

Use [repetition.py](repetition.py) for experiment composition,
[decoders.py](decoders.py) for decoding, [decoding_graph.py](decoding_graph.py)
and [matching.py](matching.py) for the detector-error-model matcher,
[belief_propagation.py](belief_propagation.py) for the decoder that reads the
hyperedges the matcher refuses, [chunks.py](chunks.py) for the round-window
decomposition, [sliding_window.py](sliding_window.py) for decoding those windows
and the projections they are read through, [registry.py](registry.py) for
reaching either by name,
[adapters.py](adapters.py) for the PyMatching cross-check,
[sampling.py](sampling.py) for sampling detection events from a memory circuit,
[noise.py](noise.py) for code-specific noise profiles, [codes.py](codes.py) for
code records, [css_code.py](css_code.py) for a code record built from
parity-check matrices, [circuit.py](circuit.py) for detector and observable
layouts, and [types.py](types.py) for records. Run from the repository root:

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
