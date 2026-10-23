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

The `codes` package holds the code records and `circuit.py` turns one into a memory
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

## Decode a model the matcher refuses by name

A code whose mechanisms flip three detectors has no pair-graph, so the matcher
declines it with a stated reason rather than projecting it onto an edge.
`get_decoder("belief_propagation_osd", model)` answers that model instead of
declining it:

```python
from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    SteaneCode,
    build_memory_circuit,
    get_decoder,
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

decoder = get_decoder("belief_propagation_osd", model)
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
the span of the model's mechanisms is refused rather than answered. The
registry reaches it by name -- `decoder_names()` holds this decoder beside the
matcher, the windowed reading of it and the PyMatching cross-check -- and the
reason the name was absent for a while is now the reason the registry states
two protocols rather than one: what every name promises is the decoded
syndrome, `DetectorErrorModelDecodeResult`, and the decoding graph the
graphlike names build is `GraphlikeDetectorErrorModelDecoder`, which a model
with a three-detector mechanism cannot satisfy. Asking this name for a
`DecodingGraph` is refused with that reason.

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

## Read the text stim writes

`DetectorErrorModel.from_stim_text(text)` reads the stim detector error model text
without importing stim, and `DetectorErrorModel.to_stim_text()` writes it. The text
stim prints for a memory experiment at five rounds or more states its repeated
middles as a `repeat` block, and the reader interprets the block rather than
refusing it:

```python
from flagquantum.qec import DetectorErrorModel

model = DetectorErrorModel.from_stim_text(
    "detector D0\n"
    "detector D1\n"
    "detector D2\n"
    "detector D3\n"
    "detector D4\n"
    "repeat 3 {\n"
    "    error(0.1) D0\n"
    "    shift_detectors 2\n"
    "}\n"
)
print([e.detectors for e in model.errors])  # [(0,), (2,), (4,)]
```

The block's `shift_detectors` advances once per iteration, which is what places the
three copies at successive detectors, and `repeat 0` is legal and contributes
nothing. Reading a block is an **expansion**, so a block costs what its count
states — stim keeps the block and expands it on demand, and this reader does not.
Whatever the block, this reader states no model it cannot hold: a declaration that
skips an index, an error mechanism that flips nothing, a malformed line, a stray
closing brace and a block that never closes are each refused with the reason
named. One narrowing is deliberate. The detector count comes from the `detector`
declarations, so a text that states it only through its error targets —
`error(0.1) D0`, which is what stim prints for a flat one-detector circuit — is
read as a model with no detector and refused, where stim would infer the count.

## Declare a code this package does not ship

The other direction of the same record: `CssCode` takes the four blocks a caller
writes and returns the same kind of code record the eight declared families
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

Two classes rather than one, and the split is the correction this section needs:
the four blocks a caller writes, and the code algebra they are held to, belong to
`CssCodeMatrices` in [dem_construction.py](dem_construction.py), while the record
is [CssCode](css_code.py), which wraps a `CssCodeMatrices` with a **stated**
distance -- the shape the section above uses. The distance search described earlier
in this section is what the declared and derived families go through in
[codes/__init__.py](codes/__init__.py) rather than what this route does, so the paragraphs above
stand as written about the families and about the bound, and what changes is which
class the four blocks are handed to. The eight declared records are the set this
section names; `get_code` reaches three of them by name.

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
nothing about the code changed. `tests/qec/test_toric_memory_execution.py`
decodes three hundred draws of that model, requires every correction to
reproduce its syndrome, and compares the observables it predicts against the
model's exact raw rate -- 0.0033 and 0.0100 against 0.1000 and 0.0933 at a
one-percent fault rate. At five percent the same measurement is 0.2667 against
0.3588, which is recorded rather than asserted: this decoder is a route to an
answer, not a threshold claim.

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

The same placement stated by the other branch, kept in full, followed by the two
corrections that reading needs:

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

Two sentences in the paragraphs above are earlier readings, corrected here rather
than left standing, and both corrections were read off the tree. The engine does
execute one channel, but the merged sampler does not conjugate it:
[channels.py](../noise/channels.py) declares `y_flip_channel`, and
[sampling.py](sampling.py) places each of the three data families as its own channel
at the one data location, which is the reading the paragraph before this one states
and the reason a pair rate is one draw at one family's rate rather than a
conjugation around the engine's flip. And a fault placed after a named gate does
have a location: `flagquantum.noise.NoiseModel` is a record the sampler accepts
beside `PhenomenologicalNoise`, and a gate-bound fault is placed immediately after
the gate its rule matched -- the grammar the section below states. What the
paragraph before this one keeps is the part that is still refused by name on both
routes, and the section below lists it: a channel that is not a single-qubit Pauli
fault, a rule naming `measure` or `reset`, a one-qubit channel bound to two wires,
and a fault that would follow the program's last instruction.

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
block, and `from_stim_text` interprets the block rather than refusing it, so the
unflattened text is read directly and `.flattened()` is the form the guide's own
fences use rather than the only form this reader accepts. An earlier revision of
this section recorded the block as refused by name, which was true until
[dem_text.py](dem_text.py) read blocks; the section above states what the reader
does with one. The other refusals, the two readings of the `^` separator and the
observable marginal each one implies, and what this route does not cover are in
[Migrating a Stim workflow](../../docs/guides/STIM_USER_MIGRATION.md); that guide
is executed fence by fence by `tests/qec/test_stim_user_migration.py`, and
[`examples/qec/stim_user_migration.py`](../../examples/qec/stim_user_migration.py)
is the runnable form of the same route.

## Change and verify

Use [repetition.py](repetition.py) for experiment composition,
[decoders.py](decoders.py) for decoding, [decoding_graph.py](decoding_graph.py)
and [matching.py](matching.py) for the detector-error-model matcher, which is the
authority decoder this layer ships, [belief_propagation.py](belief_propagation.py)
for the decoder that reads the hyperedges the matcher refuses,
[bposd.py](bposd.py) for the belief-propagation-and-ordered-statistics decoder
reached by name, [chunks.py](chunks.py) for the round-window decomposition,
[sliding_window.py](sliding_window.py) for the same decoder over a band of detectors
at a time and for the projections those windows are read through,
[registry.py](registry.py) for reaching a decoder by name,
[adapters.py](adapters.py) for the PyMatching cross-check,
[sampling.py](sampling.py) for sampling detection events from a memory circuit,
[dem_circuit.py](dem_circuit.py) for writing a model as a circuit and reading it
back, [noise.py](noise.py) for code-specific noise profiles, [codes/__init__.py](codes/__init__.py)
for code records, [css_code.py](css_code.py) for a code record built from
parity-check matrices, [dem_text.py](dem_text.py) for the stim text reader,
[circuit.py](circuit.py) for detector and observable layouts, and
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
