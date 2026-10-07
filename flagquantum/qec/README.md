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

A syndrome the budget refuses is the case `sliding_window.py` exists for. A
memory experiment's defect count grows with its rounds, so a long history is
exactly where the exact matcher stops answering; the windowed decoder answers it
by deciding a band of detectors at a time against a window of them, with the
budget bounding one window's syndrome rather than the whole history's. It returns
the same result record as the matcher, and with one window over the whole graph it
selects the same mechanisms. A narrow window is a decision about a window, so
observable agreement is bought with the width and not guaranteed, and no threshold
or logical error rate is estimated from it:

```python
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    SlidingWindowMatchingDecoder,
    build_memory_circuit,
)

long_model = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=60),
    noise=PhenomenologicalNoise(data_flip=0.04, measurement_flip=0.04),
).merge_duplicate_mechanisms()
bands = SlidingWindowMatchingDecoder.from_detector_error_model(long_model)
print(bands.commit, bands.window, bands.mechanism_span)
```

`get_decoder("sliding_window_matching", model, commit=..., window=...)` reaches
the same decoder by name, and `get_decoder("minimum_weight_matching", model)`
reaches the exact matcher.

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

## Decode a model the matcher refuses

A code whose mechanisms flip three detectors has no pair-graph, so the matcher
declines it with a stated reason rather than projecting it onto an edge.
`BeliefPropagationOsdDecoder` answers that model instead of declining it:

```python
from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    BeliefPropagationOsdDecoder,
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    SteaneCode,
    build_memory_circuit,
)

memory = build_memory_circuit(SteaneCode(), rounds=1)
model = DetectorErrorModel.from_memory_circuit(
    memory, noise=PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
)
print(model.num_detectors, model.num_errors)
try:
    MinimumWeightMatchingDecoder.from_detector_error_model(model)
except CapabilityError as error:
    print(str(error).split(",", 1)[0])

decoder = BeliefPropagationOsdDecoder(model)
result = decoder.decode([0, 1, 2])
print(result.observables, round(result.weight, 6), result.converged, result.iterations)
```

The decoder iterates min-sum messages between detectors and mechanisms, then
solves the syndrome exactly over the mechanisms that estimate favours most, and
the result says whether the iteration converged and how many passes it spent.
`result.mechanisms` names the mechanisms it selected, so the correction can be
read rather than only applied, and `result.observables` is the prediction.

It is a decoder and not an optimal one, and the boundary is measured rather than
implied. On the one-round Steane model it reaches the least weight and the most
likely observable for all 64 syndromes; on the one-round rotated surface code it
carries more weight than the least-weight explanation on five syndromes of 256
and differs from the most likely observable on three of them. A syndrome outside
the span of the model's mechanisms is refused rather than answered, and this
decoder is deliberately not in `decoder_names()`: the registry's protocol
promises the decoding graph it built, and such a model has none.

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

## Declare a code this package does not ship

The other direction of the same record: `CssCode` takes the four blocks a caller
writes and returns the same kind of code record the six declared families
return, so a code no record here declares reaches the memory circuit, the model,
the sampler and the decoder through the same `StabilizerCode` protocol.

```python
from flagquantum.qec import CssCode

checks = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
]
steane = CssCode(
    hz=checks, hx=checks, lz=[[1, 1, 1, 0, 0, 0, 0]], lx=[[1, 1, 1, 0, 0, 0, 0]]
)
print(steane.num_data_qubits, steane.num_ancilla_qubits, steane.distance)  # 7 6 3
```

The blocks are plain sequences of `0`/`1` rows, so this module needs no array
dependency; a tensor block is refused with the `.tolist()` that reads it named.
Everything is checked rather than assumed: the two check families must commute,
each declared logical operator must commute with the opposite check family while
not lying in the span of its own, the declared operators of a family must pair up
non-degenerately, the blocks must agree on a width that leaves at least one
logical qubit, and an all-zero row is refused because it acts on no data qubit.

The distance is the part worth stating carefully. It is **computed by searching
the code**, not read off the operator the caller wrote down, so the same Steane
code reports three whether its logical operator is declared at weight three or at
weight seven. The search is bounded by `distance_search_weight`, which defaults to
three; a code whose lightest logical operator is heavier than the bound is refused
with the bound named rather than answered with the best weight it happened to
reach. Raise the bound for a code that needs it, and expect the cost to grow with
the data-qubit count.

A check states one ancilla and one coupling order -- the Z-factor pairs first,
then the X-factor pairs -- so a mixed X-and-Z stabilizer is measurable by one
ancilla rather than needing a second, and `CodeCheck` states that rule once.
What a *matrix* route can state is narrower than what the check protocol admits:
this one takes a Z-type block and an X-type block, so a mixed stabilizer has no
row here, because writing it as a row of each would describe two checks that do
not commute. A family whose checks are all mixed is declared as a record instead,
and `ZxxzSurfaceCode` is the first one.

## Derive a family from its lattice

The two routes meet in one place: `triangular_colour_code` derives a family's
matrices from a rule about its lattice and returns the same `CssCode` a caller
would otherwise have written out, so the family is stated once rather than
transcribed.

```python
from flagquantum.qec import triangular_colour_code

colour = triangular_colour_code(5)
print(colour.num_data_qubits, colour.num_ancilla_qubits, colour.distance)  # 19 18 5
weights = {
    max(len(check.stabilizer.z_wires), len(check.stabilizer.x_wires))
    for check in colour.checks
}
print(sorted(weights))  # [4, 6]
```

Each check appears once with one of its two types populated, so a check's weight
is the larger of its two supports rather than either one alone.

Two things about it are worth knowing before reading its checks. Every face
carries one X-type and one Z-type stabilizer, so the two check blocks are the same
matrix and the patch is read out in either basis; and a face on the triangle's
edge spans four qubits while a face in its bulk spans six, so one data fault
lights three detectors and the model is **not** graphlike — the matcher refuses it
and the belief-propagation decoder answers it. The distance is searched for over
the derived matrices exactly as it is for a code a caller writes, so the reported
distance is a property of the patch and not of the logical string the record
declares. At distance three that patch is the Steane code under a relabelling of
its wires; the family grows away from it, and either way nothing downstream can
tell which route produced the record.

## Derive a family from a torus

The same route reaches a family whose lattice is periodic. `toric_code` identifies
the opposite sides of a square grid, so no check sits on a boundary and every check
has the same weight at every linear size.

```python
from flagquantum.qec import build_memory_circuit, toric_code

torus = toric_code(3)
print(torus.num_data_qubits, torus.num_ancilla_qubits, torus.distance)  # 18 18 3
memory = build_memory_circuit(torus, rounds=3)
print(len(memory.observables), [len(o.pauli.support) for o in memory.observables.observables])
# 2 [3, 3]
```

This is the first record here that leaves **two** logical qubits, so the default
memory experiment reads out two observables rather than one: a logical operator is
a cycle that wraps the torus, and the two directions are separate operators whose
representatives must be written down one per logical qubit. The linear size is the
distance, so `toric_code(3)` and `toric_code(5)` are different codes rather than
one code asked for at two distances, and the distance is searched for over the
derived matrices with the linear size as the bound. That bound is the whole cost of
the build, which is why the family is affordable at the small tori an experiment
is built from and is not a way to write down a large one.

The torus is also the clearest case of a model whose graphlikeness is a property of
the **noise declaration** rather than of the code. A Z-type check has a detector at
every round boundary and an X-type check only in the interior rounds, so a
single-round model is graphlike under any noise: one round has no interior for an X
fault to separate into. Add a Y fault and run at least two rounds, and that fault's
X half and Z half land on different round boundaries, so one data fault lights four
detectors, the model stops being graphlike, and the matcher refuses a model it
accepted at one round. The belief-propagation decoder answers that model, and
nothing about the code changed.

## Derive a record from any check matrices

The lattice routes above are one family each. `qldpc_code` is the general case
behind them: give it the two parity-check blocks and it returns the same `CssCode`
a caller would otherwise have written out, with the logical operators derived
rather than asked for.

```python
from flagquantum.qec import qldpc_code

# The Steane code, stated as its three checks and nothing else.
checks = [
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1],
]
steane = qldpc_code(hz=checks, hx=checks)
print(steane.num_data_qubits, steane.num_ancilla_qubits, steane.distance)  # 7 6 3
print([operator.to_text() for operator in steane.logical_observables])
# ['Z1*Z4*Z5', 'X1*X4*X5']
```

The width is the number of columns, so the checks state how many data qubits the
code has and a logical operator is read out of them. `CssCode` admits the same
pair, but it also requires the logical operators, and the derivation here is what
supplies them: it takes the null space of one check family modulo the span of the
other, and then pairs the two families by symplectic elimination so that
`lz[i]` and `lx[i]` are partners rather than merely both present.

Two consequences are worth knowing before reading the output. The number of
logical qubits is the **rank** of the matrices and not their row count, so a check
written twice adds an ancilla and no logical qubit. And the basis is not unique: a
complement of the stabilizer span has many representatives, and which one comes
back is an implementation choice. The derived operator is the same *class* as the
one a declared record states, not necessarily the same bits, which is why
`bivariate_bicycle_code` is built on this route and still reports the parameters
the literature states for its polynomials. The distance is searched over the code
rather than read off the derived operator, so a representative far heavier than the
code's cheapest logical operator still reports the cheaper number.

A pair that does not commute, a pair whose checks leave no logical qubit, a row
that acts on no data qubit, and a distance bound the search does not reach are
each refused with the reason named rather than answered. One of the two blocks may
be empty — a code with only Z-type checks still states its width — but not both,
because then nothing states how many data qubits there are.

## The one family written on the Boolean cube

Every route above is a lattice patch or a pair of polynomials over a group.
`reed_muller_code` is the first family stated on the cube instead: `RM(m, m)` is the
span of the monomials of degree at most `m` evaluated at all `2**m` points, the two
blocks two degrees apart are punctured where every variable is one, and their duals
state the two check families.

```python
from flagquantum.qec import reed_muller_code

code = reed_muller_code(4)
print(code.num_data_qubits, code.num_ancilla_qubits, code.distance)  # 15 14 3
print(code.x_distance, code.z_distance)  # 7 3
```

Those two numbers are why the record states them separately. The least X-type
logical operator has weight seven and the least Z-type one has weight three, so the
code's distance is the smaller of the two, and a bound that reaches the lighter
family does not reach the other. **The default bound of three therefore refuses this
code rather than reporting three as its distance**, because a record reports a
searched number and three was never a search over the X-type family:

```python
from flagquantum.qec import reed_muller_code

reed_muller_code(4)                        # ValueError: no X-type logical operator
reed_muller_code(4, distance_search_weight=7)  # (7, 3), distance 3
```

Which block states which family is a convention rather than a fact about the code:
the two blocks exchanged is the same code with the two Pauli families renamed, and
both readings were measured before the rule was written. What that means for a
caller is that a code whose literature states `(3, 7)` is this one, not a different
one — pass the pair through `qldpc_code` and the two numbers come back exchanged.

The rule is stated for every cube width at or above four, and this route builds the
two smallest. Four is the smallest width whose punctured pair states a code at all:
at width three the two blocks are one and zero degrees apart, their duals carry
three and six rows over seven points, and ten of those eighteen pairs of rows
anticommute, so there is no stabilizer group to return. Five is the widest the
distance search is paid for, and the refusal above five says so in the open rather
than waiting: the search has to exhaust every subset of weight up to six, which is
942648 subsets over the thirty-one points of width five, 75611760 over the
sixty-three of width six, and 5434287328 over the hundred and twenty-seven of width
seven. The width five instance is the one worth knowing about — thirty-one data
qubits carrying **eleven** logical qubits three errors apart, where a lattice patch
of the same length carries one.

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
places both kinds of location: a data fault at a round boundary and a measurement
flip at a check's readout. The data kind is the one that has three fault families,
and they are three channels at the one location rather than three locations: an X
fault, a Z fault and a Y fault. The Y fault is one channel carrying the Y operator
on its single non-identity branch, which is not the same mechanism as an X fault
beside a Z fault — the two would fire independently, so the pair rates between the
two detector bands would differ — and the sampler reproduces the model's pair
rates, not a two-fault reading of them.

A fault placed after a named gate has no location here, and neither does a
depolarizing or a damping channel: the sampler places Pauli faults at the two
positions the record describes and nothing else.

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
model, so a logical-failure rate comes from the circuit's own shots rather than
from the model's sampled ones. It shares the code record and the noise record with the
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

## Move between a model and a realization

`dem_circuit.py` goes the other way from construction: a model is written as the
circuit it already is — one wire per detector, one per observable, one
Pauli-frame channel per fault — and read back. That makes a model something a
simulator can draw from, on the stabilizer engine, without a statevector.

```python
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    circuit_from_detector_error_model,
    detector_error_model_from_circuit,
)
from flagquantum.simulation.stabilizer import sample_noisy_measurements

model = DetectorErrorModel.from_memory_circuit(
    build_memory_circuit(RepetitionCode(distance=3), rounds=3),
    noise=PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.02),
)
realization = circuit_from_detector_error_model(model)
shots = sample_noisy_measurements(realization, shots=100, seed=0)
back = detector_error_model_from_circuit(
    realization, num_detectors=model.num_detectors
)
print(shots.shape, back == model)
```

The two readings are exact in both directions, error ids included, because a
frame's operators say which targets a fault flips and its declared masses say how
often, so the mass is never recovered by squaring a square root. What the reader
returns is not the circuit the model was built from and cannot be: a model states
which detector parities a fault flips and never which measurements compose a
detector, so the syndrome-extraction circuit, its gate sequence and its depth are
outside it, and two different memory experiments of the same distance under the
same noise record define the same model. The realization is the canonical
detector-level circuit the model does determine, the detector count is the
caller's because a wire index does not say which side of the split it is on, and a
mechanism at probability zero is refused rather than dropped, because dropping it
would leave a realization standing for a model with fewer mechanisms than the one
it was read from.

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
[sliding_window.py](sliding_window.py) for the same decoder over a band of
detectors at a time, [bposd.py](bposd.py) for the hyperedge decoder,
[registry.py](registry.py) for reaching a decoder by name,
[adapters.py](adapters.py) for the PyMatching cross-check,
[sampling.py](sampling.py) for sampling detection events from a memory circuit,
[dem_circuit.py](dem_circuit.py) for writing a model as a circuit and reading it
back, [noise.py](noise.py) for code-specific noise profiles, [codes.py](codes.py)
for code records, [circuit.py](circuit.py) for detector and observable layouts,
and [types.py](types.py) for records. Run from the repository root:

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
