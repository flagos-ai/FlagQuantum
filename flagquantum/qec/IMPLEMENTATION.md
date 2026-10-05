# Quantum error correction

This domain owns quantum-error-correction concepts and workflows: code
definitions, syndrome and detection-event records, decoder contracts,
correction decisions, and logical-result analysis.

It does not define generic circuit, compiler, runtime, simulation, noise, or
provider semantics. Dynamic control remains in Compiler and Runtime; QEC
workflows compose those capabilities. Numerical kernels remain in Simulation,
and hardware feedback contracts remain in Remote or a future realtime Runtime
domain.

The first experimental profile contains a three-data-qubit repetition code, a
bounded deterministic X-error schedule, a two-bit-syndrome lookup decoder, and
a fixed-round memory experiment using the private bounded hybrid compiler.
`compiled_lookup` performs immediate reference feedback inside the lowered
circuit; the separate `offline_pauli_frame` mode leaves the data uncorrected
until a decoder-produced frame is applied after execution. Two local Runtime
modes call a replaceable `StreamingDecoder` after every syndrome round and
either apply a physical X (`runtime_decoder`) or update an X Pauli frame
(`runtime_pauli_frame`). The frame adjusts later syndrome interpretation and
final readout without changing the quantum state. Shot traces retain true and
observed bits, actions, and frame evolution. This is not a hard-real-time or
provider feedback contract. The namespace is not exported from the stable
`flagquantum` root API.

`RepetitionTemporalDecoder` adds a bounded measurement-error-aware policy. It
requires the same non-zero syndrome in two consecutive rounds before acting.
An isolated readout excursion therefore clears with a paired detection event
and produces no correction. The extra evidence round means errors first seen
in the final round remain visible but unconfirmed. The corresponding modes are
`runtime_temporal_decoder` and `runtime_temporal_pauli_frame`.

`RepetitionNoiseProfile` maps code-specific circuit locations onto the existing
backend-neutral `NoiseModel`: a data bit-flip channel is sampled after matching
parity-check CNOTs, syndrome readout confusion applies to ancilla measurements,
and optional final readout confusion applies to data qubits. Because the middle
data qubit participates in two check CNOTs per round while each edge qubit
participates in one, its configured channel has two opportunities per round.
Finite-shot sweeps report observations and event counts; they are not threshold
or logical-suppression evidence.

## Ten-minute change path

1. Add or modify a QEC-owned code, decoder, or result rule.
2. Keep generic dynamic-program changes in Compiler or Runtime.
3. Add a scenario test under `tests/qec`.
4. Run:

   ```bash
   python -m pytest tests/qec -q
   python tools/check_architecture.py
   ```

Runtime decoder feedback is trajectory-only; explicit batched feedback fails
closed. The stochastic profile is limited to independent bit flips and
independent readout confusion. No general Kraus/correlated/timing-noise,
maximum-likelihood or general measurement-error-tolerant decoder,
logical-error-suppression, threshold, fault-tolerance, realtime-hardware, or
performance claim follows from the workflow.

## Code-independent records

`pauli.py`, `codes.py`, and `circuit.py` add a code-independent layer beside the
frozen repetition profile. A `Pauli` is a phase-free operator over arbitrary qubit
indices; a `StabilizerCode` is a value that declares its distance, qubit layout,
checks, stabilizers, and logical observables; `build_memory_circuit` turns a code
and a round count into circuit source plus a detector layout and an observable
layout. `RepetitionCode` and `RotatedSurfaceCode` are the two records that
implement it.

Detector semantics are fixed. A detector is a measurement parity that is
deterministic in the noiseless circuit. Which parity that is depends on the
check's type, because both the initial state and the terminal data readout are in
the Z basis. A Z-type check is deterministic in round zero and again at the
terminal readout, so it declares one detector per round comparing that round
against its predecessor -- where the first round is compared against the known
all-zero prior state -- plus one detector comparing the final syndrome round
against the terminal data readout. An X-type check is deterministic in neither
place and declares only a detector for every round after the first, comparing two
consecutive syndrome rounds. The count is therefore
`nz * (rounds + 1) + nx * (rounds - 1)`, which reduces to the earlier
`len(checks) * (rounds + 1)` exactly when every check is Z-type; for a
distance-`d` repetition code, whose `len(checks)` is `d - 1` and which has no
X-type check, it equals `(d - 1) * (rounds + 1)`. Logical failure is the parity of
a declared logical observable.

The detector count is validated against the code's declared checks and the
configured round count, and every reference is validated against the code's
declared qubits: a detector's terminal readouts must name declared data qubits and
its syndrome measurements must name declared ancilla qubits inside the configured
rounds, and an observable's readout must name declared data qubits and match the
code's declared logical operators. The validation is membership-based. It does not
check that a detector names the *same* check in each round, that a terminal
detector's data qubits are the support of the check it belongs to, or that the
`source` text is the program those layouts describe, so a hand-built layout can
still be semantically wrong while satisfying every check.

This layer is additive. The frozen repetition types, the two decoder protocols,
three reference decoders, and both existing workflows keep their current
signatures and behavior, and the seven names pinned by
`contracts/hybrid-compilation-private-v0-candidate.json` are unchanged. The
`RepetitionCode` at `distance=3` reproduces the frozen circuit instruction for
instruction, which is asserted by
`tests/qec/test_memory_circuit_execution.py`.

The frozen profile defines logical failure as the majority of its three
frame-corrected data bits. That is a profile-specific decision rule rather than a
linear logical observable, so general-layer failure counts are not claimed to
equal frozen-profile failure counts at `distance=3`. This layer does not change
the frozen profile's arithmetic.

The layer declares codes, detectors, and detector error models. It does not
decode, and it makes no threshold, logical-suppression, real-time, or
fault-tolerance claim.

`dem.py` adds a detector error model over that layer, and `PhenomenologicalNoise`
in `noise.py` is the noise record it consumes. A `DemError` is one independent
physical error mechanism, named by the detectors and the logical observables it
flips; `DetectorErrorModel` is a set of them over a fixed detector and observable
count, with its parity matrices, its exact marginal rates, seeded sampling, and
stim text interchange. `from_memory_circuit` builds the model of a memory circuit
under a `PhenomenologicalNoise`: one mechanism per data qubit per round, and one
per check measurement per round, minus the locations whose probability is zero,
with the mechanisms that flip the same detectors and observables merged. The
model is exact for Pauli noise in the reference gate set.

`dem_construction.py` owns where those mechanisms come from, and it knows two
descriptions of an experiment. `_memory_circuit_entries` is the **forced** route
`from_memory_circuit` calls: it enumerates the locations a noise record
configures, injects one at a time into the source program the circuit carries,
and reads the flip set off the circuit's own detector and observable layouts.
`_code_matrix_entries` is the **read** route: it takes the four CSS blocks
(`hz[k, q]` is one when a Z-type check `k` sees data qubit `q`, `hx`, and the
logical-operator matrices `lz` and `lx` in the same convention), and derives every
signature combinatorially, with nothing lowered or executed.
`DetectorErrorModel.from_code_matrices` is its public entry point and
`css_code_matrices` is the bridge that lifts a code record into the
`CssCodeMatrices` record it consumes, so a code described by its stabilizers
reaches a model without a circuit being written for it. The bridge reads every
block the record declares; a logical observable that is neither pure X nor pure Z
is refused with its index named rather than read as one of the two, so the fault
family a half-read code would have lost cannot be lost silently.

The two routes describe different experiments and the difference is stated rather
than glossed. The matrix route is the code-capacity one: a fault in round `r`
reaches the detector band of round `r` and the band of round `r + 1`, and the
final round has no band after it, so its detector count is `num_rounds *
num_checks` with no terminal readout. A memory circuit measures its data qubits,
so it gains a terminal detector per Z-type check and its count is one band per
round plus that terminal. Neither geometry stands for the other, and each is
pinned against stim through a transcription that states its own assumption. A
fault that is still in the data at the end of the run is still seen by the
logical readout, so a physical fault spans one band, and the model's second band
is the one place the two descriptions genuinely diverge; that divergence is
pinned as the exact relation between the two mechanism sets rather than as a
tolerance.

Both routes carry the same single fault family: a data qubit's bit flip and a
check's syndrome bit flipped at readout. The phase-flip family an `hx`/`lx` pair
describes, and the independent `px`/`py`/`pz`/`pm` rates upstream states, are not
expressible through one data-flip scalar and one measurement-flip scalar; that
limit is recorded against the entry point in the alignment contract rather than
approximated here. On the matrix route an X-type check and an X-type logical
operator therefore contribute no row at all, rather than an all-zero row that
would declare an observable no mechanism ever reports.

Merging is also an operation a caller asks for, not only a step construction
performs. `DetectorErrorModel.merge_duplicate_mechanisms(rule=...)` gives every
shared signature one prior, `DemMergeRule` states the two rules by what they
compute — `INDEPENDENT_PARITY` for mechanisms that are independent and coincide,
`CLAMPED_LINEAR_SUM` for the sum of the group clamped at one — and
`mechanisms_are_unique()` and `require_unique_mechanisms()` are the predicate and
the refusal. The default parity rule is exact rather than tidy: a detector's rate
is a product of `1 - 2p` factors over the mechanisms touching it, and a group's
combined prior is the single `p` whose factor is that product, so regrouping the
factors leaves every detector rate and every observable rate unchanged. The sum
rule does not preserve them, and is offered for the caller who means it.

A model can also state that mechanisms are alternatives rather than independent,
and `dem_alternatives.py` owns that reading. `DemError.error_id` is an optional
non-negative label; the mechanisms sharing one are mutually exclusive, so at most
one of them fires in a shot, which is how a correlated or decomposed fault is
stated. It is the one statement the parity matrices cannot carry, because two
columns of a parity matrix are independent by construction, which is why it lives
on the mechanism's own record and why every entry point either carries it or
refuses the model: `from_memory_circuit` and `from_code_matrices` produce no ids,
`from_stim_text` cannot meet one, and `error_ids` projects the vector back out in
the sense upstream gives that name — one entry per mechanism, mechanisms sharing
an entry being alternatives, and `None`, upstream's `nullopt`, exactly when no
mechanism states an id, which is what every construction route here produces and
what stim's text always describes.

The identifiers are given a distribution rather than left as a bare flag, because
"these are correlated" is not yet a model. The members of a group are read as
disjoint pieces of one shot: each keeps the probability it states, the left-over
mass is the group firing none of them, and a target's marginal rate over the group
is the sum of the members that touch it rather than their parity. Two independent
mechanisms of 0.1 and 0.2 give a detector they both flip a rate of 0.26, and the
same two stated as alternatives give 0.30. Sampling follows the same reading,
drawing one uniform per group and landing the shot in one member's interval or in
the left-over mass, so a member fires in exactly the shots the model says it does
while the shots in which the group fires at all follow the group's mass rather
than the larger share two independent draws would give. A group that sums to
exactly one is admitted — the group then fires every shot — and a group of one
member excludes nothing and behaves as if it were unstated. A group whose
probabilities sum *above* one is refused rather than renormalized: renormalizing
would change every member's stated rate and leave nothing to read back, so the
refusal names the id and the sum. `stated_ids()` and `exclusive_groups()` are the
model's statement of what it excluded — empty and empty respectively exactly when
it is independent — and `error_ids` is the label vector, whose values are opaque
and whose numbering is therefore a normalization: two models that differ only in
how their ids are numbered are the same model, and what is a fact about a model is
the partition the vector induces.

An id is a statement about the distribution, so an operation that assumes
independence refuses an id-carrying model by naming the ids rather than dropping
the structure, and each refusal is at the call site where that operation's own
premise is the thing to explain. `to_stim_text()` refuses, because the format
reads every error instruction as an independent mechanism and printing would
therefore state a different model at the same probabilities.
`merge_duplicate_mechanisms()` refuses, because both of its rules combine
mechanisms by assuming independence and either would invent a shot in which two
alternatives both fired.
`MinimumWeightMatchingDecoder.from_detector_error_model` and the decoding graph
behind it refuse, because one weight per mechanism is the weight of a fault that
fires alone, while a group of alternatives is one fault whose weight is the
negative log-likelihood of the group. What no route here does yet is the
operation that would resolve the refusal rather than avoid it: folding a group
under its exclusivity into the single mechanism a matcher can weigh. That is the
gap the alignment contract records against `dem_canonicalize`, upstream's
`canonicalize_for_rounds` family, and it is more load-bearing now than it was
before the ids landed, because three call sites refuse a model for want of it.

`tests/qec/test_dem_error_ids.py` pins the statement, its arithmetic and its
refusals: the exact marginals against the parity formula they replace, a sampled
rate against the exact one inside a stated standard-error bound, the absence of
any shot in which two members of one group fire, the three refusal messages, and
that a model with no ids draws and weighs exactly as it did before the field
existed.

Construction is exact and does not sample. On the forced route each mechanism's
signature comes from one forced execution: the single error is injected into the
circuit source, the source is lowered and executed twice, and the two shots must
agree before the signature is read off the detector and observable layouts. That
two-shot determinism assertion is both the fail-closed check for a mechanism that
is not a Pauli mechanism in the reference gate set and the evidence that the
signature is not sampled. The model is defined for Pauli noise only: a non-Pauli
channel is refused with a stated reason rather than approximated. The read route
asserts determinism differently, by construction: its input is a support, so a
matrix that is not binary, not two-dimensional, or not indexed by the same data
qubits as its logical matrix is refused rather than rounded into shape.


The model is built on the memory circuit without in-circuit feedback, because it
describes the noise-to-detection mapping that a decoder inverts. The frozen
profile's `compiled_lookup` mode bakes immediate reference feedback into the
frozen profile's own private source rather than a `MemoryCircuit`, and the
forced-error engine drives `MemoryCircuit` sources, so mechanical limits leave
that compiled-feedback path unmodelled.

The detector-rate evidence is a cross-check, not an independent derivation.
Phenomenological noise is the primary evidence: the model's exact marginal rates
are compared at distance three and distance five against rates sampled from an
injected-shot circuit simulator. The comparison shares the injection helper with
construction, so it does not independently re-derive the signatures; what it
tests is that mechanisms compose by XOR in the simulator and that merging
identical signatures yields the right marginals.

`tests/qec/test_dem_stim_reference_rates.py` is the independent check. It
transcribes the memory circuit gate for gate into a `stim.Circuit` with the same
phenomenological noise model and lets stim build a detector error model from it
with stim's own error-analysis pass. Nothing is shared with the FlagQuantum side
except the circuit's gate sequence and the noise model, which is the input the
comparison is about. On the eight configurations distance two and three at one
through four rounds, the two models declare the same shape and the same mechanism
count. Detector marginals agree within the sampling error of the stim side, at
four hundred thousand shots and a tolerance of four standard errors of the
largest measured rate: the worst correct deviation over the sweep is 0.001031 at
distance three with four rounds, while a transcription that applies the data flip
after the round's gates instead of before it deviates by 0.038 and a terminal
detector that omits the data readout deviates by 0.107. The marginals are also
compared across four noise strengths and two seeds.

`tests/qec/test_dem_code_matrices_stim.py` is the read route's separate
independent check, and it is separate on purpose. Its stim circuit states the
code-capacity assumption and nothing else: the data qubits persist across rounds
and no terminal data readout exists, so a fault keeps flipping the syndrome of
every later extraction and the detector count is the matrix route's own. The
experiment is run once per readout basis, because `lz` and `lx` anti-commute and
no single state has both as a deterministic value, and the model is restricted to
the matching basis for each run. An extraction before round zero is noiseless and
declares no detector: it supplies the prior every round-zero band is compared
against, which is the only way an X-type check can have one at all, since an
X-type ancilla on a register never measured in that basis has a coin-toss
outcome. Handing stim that circuit and requiring its own error analysis to
reproduce the shape and every full signature and rate is a statement about the
matrix route rather than about the memory circuit — the memory circuit's
transcription ends in a terminal data readout and has a different detector
count. Running the
matrix route against the memory transcription would conflate the two geometries
and hide which of them a failure belonged to, so the same suite asserts that they
differ: at distance three and three rounds the repetition code has six detectors
through matrices and eight through a memory circuit.

Marginals alone would not separate a model that keeps every marginal and drops
every correlation, so the same suite compares every detector pair rate. Because
the mechanisms fire independently, a pair's exact rate follows from the product
of `1 - 2p` over the mechanisms flipping exactly one of the two, and the
arithmetic asserts on its own diagonal against `detector_rates()`. The worst pair
deviation over the sweep is 0.000374 against a four-standard-error tolerance of
0.000672, and the tolerance is bounded above by the standard error of a
half-rate so that it cannot silently degrade into an assertion that accepts any
divergence. A model holding one mechanism per detector, carrying that detector's
exact marginal, lands between 0.29 and 0.65 of the marginal tolerance — it
reproduces the marginals by construction — while its pair rates miss by 10.7 to
23.6 times the pair tolerance, against a largest true pair covariance of 0.0284
to 0.1794.

The forced route is a forced execution, so it is bounded by the
statevector amplitude ceiling rather than by the detector error model's own cost.
A rotated surface code is `distance**2` data qubits plus one ancilla per check:
`distance=2` is 7 qubits and `distance=3` is 17, and both build in seconds, while
`distance=4` is 31 qubits (2**31 amplitudes) and did not complete in forty-five
minutes, and `distance=5` is 49 qubits and fails on the allocator. The modelled
rotated-surface distance is therefore three. Reaching five and seven needs a
signature route that does not materialise the state — either a first-party
Clifford propagation in this layer, which is a second implementation of an
algorithm this layer otherwise does not own, or an explicit decision that the
larger-distance curve is a reference-only comparison.

The matrix route is bounded by none of that. It reads a support rather than
executing a program, so its cost is the number of nonzero matrix entries times
the round count and a distance-5 or distance-7 patch reaches a model as easily as
a distance-2 one. The ceiling above is a property of the circuit route, not of
the construction contract, and the two are allowed to differ because they answer
different questions.

The stim interchange is a text format, not a package dependency: nothing in
`dem.py` imports `stim`, and the reader is exercised against real stim output by
`tests/qec/test_dem_stim_interop.py`, which skips when the optional distribution
is absent. `to_stim_text()` emits valid stim text, verified against stim 1.16.0.
`from_stim_text()` reads the format `to_stim_text()` emits and hand-written text
in that style.

Three constructs that stim emits and this reader used to refuse are now read, and
each is read the way stim means it rather than the way it is convenient to mean
it. A `shift_detectors` instruction offsets every detector index on the lines
that follow it, in the declarations and in the error mechanisms alike, and
successive instructions accumulate. A `^` separator partitions the groups a
composite mechanism decomposes into; those groups are a decoder's business and
the signature is the symmetric difference of the line's targets, so a repeated
target cancels and the groups are not retained by default. An observable count
that only the error targets state is inferred from them, because stim declares the
observable exactly when no mechanism references it; a declared count still governs.

The separator's groups are also available to a caller that wants them, because a
matching decoder needs one graphlike component per edge and upstream exposes the
same choice as `dem_from_stim_text(dem_text, use_decomp_suggestions=True)`.
`DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)` returns one
mechanism per group, each at the probability the line states, which is the
decomposition the separators carry: on a rotated surface code at distance five and
two rounds it turns 536 mechanisms into 1042. That reading is a different model
from the line and is not an equivalent statement of it, because two components that
each fire independently at probability `p` do not reproduce one mechanism at
probability `p`. Held to stim's own sampler on the same text, the default reading's
worst marginal misses by 0.0016 on the observables and this one misses by 0.048,
which is what makes the tolerance in
`test_the_suggestion_reading_departs_from_stim_where_the_default_reading_does_not`
evidence rather than decoration. The expanded model is a model in its own right --
it prints and re-reads unchanged -- and it is the default reading that states the
line stim wrote.

What remains refused is a `repeat` block, a `#` comment, a declaration that skips
an index, and a malformed line. The `repeat` refusal is deliberate rather than
pending: expanding a block means interpreting a nested instruction stream, and
`str(model.flattened())` already states the same instructions without the block.
`flatten_loops=True` is not a substitute for that call, because stim still emits a
block for a long enough circuit. One more line is refused by the suggested reading
alone: a text such as `error(0.1) D0 D0 ^ D1` has a component that cancels to
nothing, and a component with no targets has no mechanism to become, so the reader
refuses it and names the default reading -- which states that line as `D1` -- as
the route that states it.

The evidence is a developer-time sweep of stim 1.16.0 over 240 detector error
models -- repetition-code and rotated-surface-code memory circuits, distances
three, five and seven, rounds one, two, three, five and nine, noisy and
noise-free, with and without decomposed errors. This reader parsed all 180 that
carried no block, agreeing with a re-read of the same text on the detector and
observable counts, on the error count, and on every `(probability, detectors,
observables)` mechanism; the 60 refusals were all `repeat` blocks, all from
`repetition_code:memory` at five rounds or more. Every one of the same 240 models
parsed when `flattened()` supplied the text, so flattening is a complete route
around the last refusal.

Agreement on the shape and on the mechanism list is agreement with stim's own
reading of a text stim wrote, so the reader is also checked against the physics.
On a rotated surface code at distance five, two rounds and two percent noise,
this reader's marginal detector and observable rates agree with the flat text to
within 1e-9, agree with stim's compiled sampler over 200000 shots to within
0.0018 where a reading that kept only the first group of each decomposed
mechanism would miss by 0.094, and agree with sampling the circuit the text was
derived from over 100000 shots to within 0.0041 inside a budget of 0.008. The
repetition code cannot supply that discrimination: at distance three, three rounds
and one percent noise the two readings of `^` differ by at most 0.0024, so the
test uses a circuit whose decompositions actually merge groups.

The text is exact on this side, and on stim's side the loss depends on the width
stim's printer was built with rather than on the value. `to_stim_text()` writes a
probability with `repr`, the shortest decimal that reads back as the identical
double, so over twenty thousand swept probabilities no written value returned as a
different one. stim's `str()` writes at
`std::setprecision(std::numeric_limits<long double>::digits10 + 1)`, so the number
of digits follows the platform's `long double`: nineteen significant digits where
that type is the x86 80-bit extended one, which is the Linux CI platform, and
sixteen where it is a double, which is arm64 macOS. Seventeen significant decimal
digits name every double uniquely, so at nineteen digits the reprint returned
every one of the swept probabilities unchanged and at sixteen about a quarter of
them came back as a different value. That is a measured format rather than an
estimate. The reader therefore states the printed value, because the printed
value is what the interchange carried.

Where the width is the narrower one, the drift stim introduces is bounded by one
part in `10**15`. Sixteen significant decimal digits round to within half a unit
in the last of them, and reading that decimal back as a double adds at most one
binary unit in the last place, which is under `2**-51` of the value; the two
together stay under the bound. The sweeps confirm it rather than assume it: the
worst relative difference between an in-memory probability and the one stim
printed was `5.4e-16` over the pinned sweep and `4.9e-16` over the earlier
developer sweep, both inside the bound. Widths at or above seventeen digits need
no bound at all, because they round nothing away.
`tests/qec/test_dem_stim_text_precision.py` reads the width off stim's own output
and then holds the digit count, the direction of the loss and the bound against
that width, and it is a separate file because it measures stim's writer rather
than this package's model.

A round-trip assertion must compare against a re-read of the printed text rather
than against the in-memory model, or it reports differences that the printing
caused. That applies to text stim wrote on a platform whose `long double` is a
double; this package's own writer needs no such allowance on any platform, which
the same file demonstrates.

Not covered by the sweeps: `#` comments, gauge detectors, colour codes, distances
above seven, `approximate_disjoint_errors`, and hand-written text outside the
style `to_stim_text()` emits.

`DemSample` carries tensors and defines content equality, and it is deliberately
unhashable: it must not be used as a set member or a dict key.

The scope refusals hold here as they do for the rest of the layer. No threshold
claim, no logical-suppression claim, and no real-time or hardware-feedback claim
follows from a detector error model or from any rate it reports. A rate the model
samples is a DEM-sampled rate, and it is labelled as one wherever it is reported.

`README.md` carries only the decoder's `Decode a detection-event syndrome` entry,
because that entry needs a model to decode and therefore names
`DetectorErrorModel`. The detector error model's own README section is still the
Stage 5 documentation sweep's.

One representational boundary is explicit and enforced. A `CodeCheck` states one
ancilla and one CNOT direction, and the direction is fixed by the check's type
rather than left to the caller: a Z-type check controls from each data qubit in
the stabilizer's support into an ancilla prepared in `|0>`, and an X-type check
controls from an ancilla prepared in `|+>` into each data qubit. Both gadgets
leave the ancilla's Z-basis readout equal to the check's eigenvalue, which is
what makes the two symmetric. A mixed X-and-Z stabilizer is still refused, since
it needs a second ancilla and a second gadget that this record does not describe.

`RotatedSurfaceCode` is the first code here that needs both check types. Data
qubits occupy qubits `0..distance**2 - 1`, indexed so lattice site `(i, j)` is qubit
`j * distance + i`, and ancillas follow them in lattice order. An X-type ancilla
sits on an interior column at odd lattice parity, a Z-type ancilla on an interior
row at even lattice parity, and each check's support is the up to four data
qubits diagonally adjacent to it, so the checks that would fall outside the patch
are truncated to weight two. The declared logical observable is `Z` on the data
row `j == 0`.

The X-type checks change what a memory experiment has to declare. Both the
initial state and the terminal data readout are in the Z basis, so a Z-type check
is deterministic in round zero and again at the terminal readout and gets
`rounds + 1` detectors, while an X-type check is deterministic only against the
round before it and gets `rounds - 1`. The X-type checks are still measured every
round, because a Z error is precisely what the Z-basis readout cannot see: it
commutes with every Z stabilizer and anticommutes with the X stabilizers over
that data qubit, so it is reported as a change in an X-type check between two
consecutive rounds. An X error is reported in the complementary way, by the
round-zero and terminal detectors of the Z-type checks. `build_memory_circuit`
refuses a code whose declared logical observable is not Z-type, because under a
Z-basis readout such a code would be measured under premises that do not hold
for it.

That layout is checked against `stim`'s generated `surface_code:rotated_memory_z`
at 28 (`distance`, `rounds`) points: distances two through eight and rounds one
through four, agreeing on the detector count and the observable count at every
one. `test_surface_code.py` checks the same record against its own algebra
instead: the checks pairwise commute, they generate a group of rank
`distance**2 - 1`, and the declared logical observable is independent of that
group. Executing the distance-three patch through the dynamic hybrid simulator
shows the complementary syndromes directly -- a Z error fires exactly the X-type
detectors adjacent to the data qubit and no Z-type detector, and an X error on
the logical row fires exactly the Z-type round-zero detectors and flips the
observable. Only the distance-three patch is executed through that simulator: it
draws each shot from the full output distribution, so a 17-qubit patch costs a
`2**17`-way draw and a distance-five patch would need `2**49`.

## A decoder over the detector error model

`decoding_graph.py` and `matching.py` add the first decoder in this layer that
reads a detector error model rather than a repetition-code syndrome history. The
split is by responsibility: `decoding_graph.py` states what a model graphs to,
and `matching.py` searches that graph.

A `DecodingGraphEdge` connects exactly two nodes and carries a probability and a
tuple of observable labels. A `DecodingGraph` states a detector count, an
observable count, and a sorted tuple of edges, and its boundary node is one past
the last detector. Edge weight is `log((1 - p) / p)`, the negative log-likelihood
ratio, so the cheapest set of mechanisms is the most likely one; a probability of
one half has weight zero and a probability above one half is refused, because its
weight would be negative and Dijkstra's search assumes non-negative weights.
`DecodingGraphEdge` refuses a probability of zero for the same reason the graph
never receives one: its weight would be infinite.

`from_detector_error_model` reads the mechanisms and nothing else. A mechanism of
probability zero contributes no edge. A mechanism that flips no detector is the
one case the model already refuses at construction, because `DemError` requires a
non-empty signature. A mechanism that flips one detector becomes an edge to the
boundary node, because a decoder that read only the syndrome would have no way to
place a single defect otherwise. A mechanism that flips three or more detectors is
a hyperedge: it is refused with a `CapabilityError` naming the detectors it flips
rather than projected onto a pair, because the projection would drop a detector
and the matcher would then return a correction that does not explain the syndrome.
Two mechanisms that share a detector pair but disagree on their observable labels
stay two edges. Merging them would either lose a logical flip or average two
probabilities into a weight neither mechanism has, and the graph is an exact
statement of the model rather than a summary of it.

Two mechanisms that agree on *both* their detectors and their observables are the
different case, and it is where the merge operation becomes load-bearing. Such a
pair is one fault stated twice, so a matcher that must explain that detector's
defect would charge the cheaper of the two parallel edges — for mechanisms of
`0.1` and `0.2` that is `log 4`, or `1.386` — for a fault whose combined parity
`0.26` has weight `log(0.74 / 0.26)`, or `1.046`. The charge is larger than the
fault's own weight, so a matcher would prefer a longer chain of other mechanisms
over the mechanism that actually fired. The graph is right to keep the two edges
apart, because it has no rule for combining them, so
`from_detector_error_model` calls `require_unique_mechanisms()` before it weights
anything and refuses a model that states one signature twice, naming both
mechanisms and the merge that resolves it. `DecodingGraph.from_detector_error_model`
keeps its parallel edges, which is what makes the refusal a decoder decision
rather than a representation one; a caller who wants the model decoded merges it
first, and `merge_duplicate_mechanisms()` is the operation that does so.
`tests/qec/test_dem_merge.py` pins both halves: the two rules against their
closed forms, the exact preservation of every detector and observable rate under
the parity rule and its denial under the sum rule, idempotence and order
independence, the weight a duplicate would have been matched at against the
fault's own weight, and the decoder's refusal of a split fault followed by the
same syndrome predictions once that fault is merged back.

Nothing about a code, a distance, a round count, or a check layout enters the
graph. Whatever the model states is what the graph holds, so the decoder inherits
every scope limit of the model it consumes, including the modelled rotated-surface
distance of three that the statevector amplitude ceiling imposes on the circuit
route. It also inherits the fault family: a model read from matrices states only
the bit-flip family, so the matching decoder is matched to that model and not to
a channel the matrix route cannot describe. Measured on the two codes this layer
models at three rounds and two percent noise:
the repetition code at distance three gives 15 edges of which 6 reach the boundary
and every weight is 3.8918, and the rotated surface code at distance three gives 45
edges of which 20 reach the boundary, with six of the weights at 3.1991 and the
other 39 at 3.8918. Every mechanism in both models flips one detector or two, so
neither model reaches the hyperedge refusal. A matrix-route model is graphlike
only while it stays inside one round: at two rounds the middle data qubit of the
distance-3 repetition code flips the same check in both detector bands, which is
a four-detector mechanism, and `from_detector_error_model` refuses it as a
hyperedge rather than projecting it onto a pair.

`MinimumWeightMatchingDecoder` decodes a syndrome exactly in two steps. The first
searches the graph from each defective detector and yields the cheapest chain of
mechanisms to every other detector and to the boundary; weights are non-negative,
so that step is Dijkstra's search, and among chains of equal weight the search
settles the smallest node and then the smallest edge, so a tree is a function of
the graph and the source alone. The second chooses, among all the ways to pair the
defective detectors and to route some of them to the boundary, the one of least
total weight. That is a minimum-weight perfect matching on the metric closure of
the defects, and the module enumerates it with a dynamic program over the sets of
defects still to place.

The boundary is a sink rather than a waypoint, and that is what makes a syndrome's
parity irrelevant. A T-join exists for an even set of odd-degree vertices only,
while a mechanism at the edge of the patch flips one detector, so a model can
produce a syndrome with an odd defect count. A defective detector is therefore
allowed to pair with the boundary instead of with another detector, and any number
of them may do so, because each boundary mechanism is an independent explanation.
Chains between two detectors never pass through the boundary, which costs nothing:
going out to the boundary and back from two detectors costs the sum of their
boundary distances either way, and the two edge sets differ only in edges they
share, which cancel. Chains may pass through detectors that are themselves
defective, because the correction is the symmetric difference of the paired
chains, so an edge two chains share cancels rather than being counted twice.

The prediction is the exclusive-or of the selected mechanisms' observable labels,
which is the parity the model's own `observables_flips_matrix` states for those
mechanisms. It is a decision about which of the cheapest explanations the decoder
adopts, not an estimate of the probability that the observable flipped, and it is
not calibrated.

Correctness is checked against two references that share no code with the decoder.
The first is brute force: on syndromes of at most four defects in the repetition
code and in the rotated surface code, the selected mechanisms' total weight equals
the least weight over every set of up to four mechanisms whose odd-degree detectors
are the syndrome, and the maximum optimality gap over every such syndrome the two
models sample is 0.0. The second is an exhaustive enumeration of the model itself: every set of
mechanisms has a probability, a syndrome, and a logical flip, so the syndrome
distribution and the logical-flip distribution follow without sampling, and the
optimal decoder's failure rate is the mass of the less likely logical value in
each syndrome. On the repetition code at distance three, three rounds and two
percent noise the matcher fails at 0.009125123 while that optimum is 0.007714937
and a decoder that always predicts no flip fails at 0.153733002, so the matcher
beats the trivial decoder by about a factor of seventeen and the gap to optimal is
the cost of the mechanisms a matcher cannot tell apart. Those three rates are
pinned in the test to an absolute tolerance of 5e-6 rather than to their last
digit, because the last digits of a probability built by repeated convolution
depend on summation order and are not a property of the decoder.

That gap is not a defect and does not close with a better search. Two mechanisms
can share a detector pair and disagree on their logical label, and a decoder that
reads only the syndrome cannot distinguish them, so the matcher's failure rate is
at or above the optimum on every model and strictly above it once rounds exceed
one. It is exactly optimal where no such pair exists: a single-round repetition
model has one mechanism per detector pair, and at distances three, five and seven
and two percent noise the matcher's failure rate equals the exhaustive optimum with
a measured gap of 0.0 while falling by more than a factor of ten per two units of
distance. That equality at one round is the only optimality claim made here. An
exact scan over noise strengths 0.005, 0.01, 0.02 and 0.05 and one through four
rounds shows the same ordering throughout: the matcher is never below the optimum,
never above the trivial decoder, and at rounds two through four the gap grows with
the round count, from 4.89e-05 at 0.005 and two rounds to 2.07e-03 at 0.02 and four
rounds.

Because the pairing is enumerated, the cost is in the enumeration, so the decoder
accepts at most twenty defective detectors per syndrome by default and refuses a
larger syndrome with a `CapabilityError`. The budget is a constructor argument
rather than a silent truncation, and twenty is chosen because `2**20` states is
far above what a distance-three patch reaches at noise below its threshold.
Detectors that no chain of mechanisms connects are refused with a `CapabilityError`
naming them, rather than answered partially.

`MatchingDecodeResult` is deliberately not the repetition-code `DecodeResult`. A
`Correction` carries a qubit in `{0, 1, 2}` and an X basis only, and
`PauliFrame.from_corrections` is enforced to agree with it, so neither record can
express a surface-code correction over `distance**2` data qubits in the Z basis.
The result states the predicted observables, the mechanisms selected in graph
order, and their total weight, and the decoder therefore does not implement the
repetition-only `Decoder` protocol either. Nothing is built on top of this result
yet: there is no logical-error-rate estimator, no threshold scan, and no
connection to the memory-experiment result records. The implementation imports
`heapq`, `math`, `dataclasses`, and `numbers` and nothing else, so no new
dependency is introduced; `stim` and `pymatching` are not imported by it, and the
cross-check against `pymatching` is a separate module behind the `pymatching`
extra rather than a build dependency. That cross-check is
`flagquantum/qec/adapters.py`, and what it established is stated there rather
than here: the two decoders agree on the cheapest weight of every syndrome and on
the observables wherever the cheapest explanation is unique, and a tie is
uncomparable because PyMatching's own arithmetic is narrower than this module's.

## Reaching a decoder by name

`registry.py` is the factory half of the CUDA-Q QEC decoder surface: upstream
reaches a decoder through `get_decoder(name, H_or_dem_text_or_sparse_matrix,
**options)` and registers one with a decorator, and this module offers
`get_decoder(name, source, **options)`, `register_decoder(name, *,
replace=False)`, `decoder_names()`, and the `DetectorErrorModelDecoder` protocol
those three are written against. `AUTHORITY_NAME` and `CROSS_CHECK_NAME` name the
two registrations this package ships.

Three things about it are narrower than upstream on purpose, and each is a
decision rather than an omission.

The source argument is a *carrier*, not a decoder setting, which is why the three
accepted forms are the three a caller can hold a model in: the detector error
model, stim's text for one, and the decoding graph the model defines. A model is
lifted through the class's own `from_detector_error_model`; a graph is passed to
the constructor, because the graph is already the thing a matcher searches and
rebuilding a model from it would lose the observable labels the caller has in
hand; text is read through `DetectorErrorModel.from_stim_text` first. A
parity-check matrix is not a carrier, although upstream's
`H_or_dem_text_or_sparse_matrix` is, because `from_code_matrices` reads a noise
model and a round count rather than defaulting them, so a factory that lifted a
matrix would also be choosing the noise the caller decodes against. The caller
who holds the matrix and the noise together does the lifting.

The registry holds the detector-error-model family alone. The repetition-code
decoders in this layer take an ordered syndrome history rather than detection
events, and `DetectorErrorModelDecoder` requires
`from_detector_error_model`, so one name space over two input protocols would
make a name mean one of two things. They stay directly constructed, which is also
why the registry is a module of its own rather than methods on `Decoder`.

Registration is checked at registration time, while the registering module is
being imported. `register_decoder` refuses a name that is not a non-empty string,
a second registration of a name unless the caller passes `replace=True`, and a
class missing `decode` or `from_detector_error_model` — the two members every
decoder in this family shares. A `TypeError` at import time, naming the member
that is missing, is a better failure than an `AttributeError` at the first call,
where the name is all the caller has to go on.

`get_decoder` fails closed in two more places. An unregistered name raises
`ValueError` and lists the names that are registered, rather than reaching any
implementation, and a source that is not one of the three carriers raises
`TypeError` naming `DecodingGraph`, since that is the carrier a caller is most
likely to have held. `**options` goes to whichever route the source selects and
is not filtered here, so passing the text reader's `use_decomp_suggestions` to
the graph route raises rather than being dropped.

The optional implementation is registered whether or not it is installed, so
`pymatching` is part of this package's surface rather than the extra's: asking
for it without the extra raises the error that names the extra, instead of a name
that silently is not there. The adapter module is imported, but it reaches
PyMatching through a function rather than at import time, so `import
flagquantum.qec` does not import `pymatching` — a test starts a fresh interpreter
and measures that rather than asserting it. No name is preferred over another, so
`get_decoder(AUTHORITY_NAME, ...)` returns the authority wherever the extra
happens to be installed; the cross-check is never reached by accident.
