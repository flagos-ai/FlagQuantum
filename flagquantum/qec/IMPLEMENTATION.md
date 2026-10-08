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
and optional final readout confusion applies to data wires. Because the middle
data wire participates in two check CNOTs per round while each edge wire
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
frozen repetition profile. A `Pauli` is a phase-free operator over arbitrary wire
indices; a `StabilizerCode` is a value that declares its distance, wire layout,
checks, stabilizers, and logical observables; `build_memory_circuit` turns a code
and a round count into circuit source plus a detector layout and an observable
layout. Eight records implement it: `RepetitionCode`, `RotatedSurfaceCode`,
`SteaneCode`, `triangular_colour_code`, `toric_code`, `ZxxzSurfaceCode`,
`bivariate_bicycle_code` and `reed_muller_code`.

`CssCode` is the second way into that protocol, and it is the way in for a code
this package does not declare. It takes the four CSS blocks as plain sequences of
`0`/`1` rows -- `hz`, `hx`, `lz`, `lx` -- validates them, and exposes the same
protocol attributes the declared families expose, so everything downstream reads
it without knowing which route produced it. The route is the mirror of
`css_code_matrices`, which reads a declared record *out* into the same four
blocks; the two directions are inverse on the declared records and
`tests/qec/test_css_code.py` asserts that round trip rather than asserting it in
prose.

What `__post_init__` establishes, in the order it establishes it: the blocks are
rectangular and binary, no row is all zero because such a row acts on no data
qubit, the non-empty blocks agree on one width, the Z-type and X-type check
families commute (through `Pauli.commutes_with`, which is the same overlap test
the rest of the layer uses), each declared logical operator commutes with the
opposite check family and does not lie in the span of its own (through
`gf2.in_span`), the declared operators of a family pair up non-degenerately
rather than degenerately, the width minus the two family ranks leaves at least one
logical qubit, and each family declares exactly as many operators as that count.
`gf2.py` is the new shared piece: `reduce_rows` puts a row set into row-echelon
form once and returns its pivots, `reduce_vector` reduces one vector against those
pivots, and `rank` and `in_span` are the two readings of that. The pivots come back
in the order they were found rather than sorted, and that order is load-bearing
rather than incidental: a pivot is reduced against the pivots kept before it, so it
carries a zero at each of their leading positions, which is what makes one pass over
the list enough. `tests/qec/test_gf2.py` pins that clearance, pins that a reordered
basis is *not* reduced correctly by the same one pass, and pins that the pivots are
reported in the order they were built. It holds bit-per-column integers, so GF(2)
addition is `^` over Python integers and no array dependency enters the module;
`logical.py` previously carried private copies of the same two operations and now
reads them from here.

`qldpc.py` is the second shared piece, and it is the derivation the declared qLDPC
family used to hold privately. `_logical_basis` takes one check family's span
inside the other family's null space: it walks the null-space basis and keeps a
vector only when it is independent of the stabilizers *and* of the vectors kept
before it, which is what makes the result a basis of the quotient rather than a
set of free columns that happen to be independent. `_paired_basis` then puts the
two families into the symplectic basis whose pairing is the identity, so a
record's own `lz[i]` and `lx[i]` are partners. `qldpc_code` is the public entry:
it reads the two blocks, takes the width from whichever block states one, derives
both families through those two helpers, and hands the result to `CssCode`, which
still does all the checking. `bivariate_bicycle_code` now calls the same two
helpers for its own polynomial pair instead of carrying a second copy, so a
representation the quotient can be taken in has one implementation rather than
two. The derivation is not a declaration, and the distinction is load-bearing
rather than cosmetic: a complement of the stabilizer span has many
representatives, so a derived operator is the same class as a declared one rather
than the same bits, which is why `tests/qec/test_qldpc_code.py` holds every
derived operator to differing from its declared partner by a check of its own
type -- tested against the Steane record, the torus at two sizes and the bicycle
family -- and why the parameters alone would not have held the route: a wrong
representative of the right class reports the same `(n, anc, k, d)`.

The distance is computed rather than declared, which is what makes it a property
of the code instead of a property of the operator the caller happened to write
down: the same Steane code reports three whether its logical operator is stated at
weight three or at weight seven. `_minimum_logical_weight` reduces the stabilizer
span to its pivots once, then tries wire subsets in increasing weight and returns
the first weight at which some subset commutes with every check, is outside the
span, and is not a stabilizer. The search is bounded by
`distance_search_weight`, which defaults to three, and a code whose lightest
logical operator is heavier than the bound is refused with the bound named rather
than answered with the best weight the search reached. The bound is a stated cost
bound and not a distance algorithm: at the default it costs 0.004 s at thirty data
qubits and 1.3 s at a hundred and forty, and raising it to four costs 0.021 s and
37.5 s at the same two sizes, which is why three is the default and why the route
is demonstrated at eight and eighteen data qubits.

One limit survives the new route because it belongs to the protocol rather than to
it, and saying so is the honest boundary: this route states a code as parity-check
matrices, so a stabilizer carrying both an X factor and a Z factor has no row here --
not because the protocol refuses it but because one row of `hz` and one row of `hx`
would describe two checks that do not commute. `build_memory_circuit` still refuses a
declared product that is neither pure X nor pure Z. A Floquet family remains
absent as a *record* and has no matrix route either: a dynamic code states its
stabilizer group per round rather than once, and nothing here records a code whose
group changes. The other family this paragraph used to name beside it is not
absent, and the correction is measured rather than editorial: the framework this
package replaces binds its Reichardt module to the very tesseract record below --
`cudaq/logical/qec/reichardt.py` links `Tesseract`, and
`cudaq/logical/codes/catalog.py` writes that family with the same five check rows,
the same four Z-type and four X-type logical rows, the same two gauge pairs,
`k=4`, `r=2`, `d=4` and the same `path4: L1, L0, L2, L5` protected order that
`tesseract_code()` declares -- so the family has a record here under a different
name and the absence was a naming difference, not a capability gap. What the
subsystem record adds beside that is a second *reading* of matrices this route
already accepts rather than a second set of matrices: `SubsystemCode` takes the
same four blocks and two more, and the two more change what the four mean. The
tesseract's sixteen data qubits carry ten independent check rows and two
anticommuting gauge pairs, so the checks alone leave six classes -- which is
exactly what `CssCode` counts in them, and it refuses the family's four declared
logical operators for that reason -- while the record that knows about the gauge
reports four protected qubits and names the other two as gauge. Neither reading is
an error, and the two are not interchangeable, which is why this is a second
record rather than two optional fields on the first: a protection claim states
which classes survive, and a record that cannot say which centers it took makes
that claim unstated rather than false.
The new record is deliberately **not** a `StabilizerCode`, so the route that turns
a record into CSS matrices refuses it with the protocol named rather than building
matrices the gauge generators would be dropped from, and `tests/qec/test_subsystem_code.py`,
fourteen cases, holds both sides of that: the CSS record refusing the four logicals
of a code it counts six of, and this record absent from the members that route
reads. The
distance it reports is dressed -- every declared logical operator is required to
commute with the opposite checks *and* to lie outside its own family's checks and
gauge -- and the same file measures both readings, so a reader can see that four
is the family's declared distance under each rather than an artifact of which
center the search took. It measures something stronger than that as well, because
the tesseract cannot tell the two readings apart: its dressed and its check-only
distance are both four. The 3x3 Bacon-Shor is the smallest subsystem family where
they disagree -- a search up to the checks alone finds a weight-two operator and
the record reports three -- so "the distance is dressed" is a measured claim rather
than a number the published family happens to satisfy either way. The family's use
is the free logical gate it carries, and the gate is now published beside the
record rather than held by a test: `tesseract_column_swap()` reads the
sixteen-wire permutation off the hypercube layout -- a wire whose lowest coordinate
is clear exchanges the column above it -- and `tesseract_free_cnot_pairs()` states
the two protected pairs it couples, so the claim that the permutation implements
`CNOT(0 -> 1) * CNOT(2 -> 3)` on the declared `path4: L1, L0, L2, L5` order joins
two independently written facts instead of restating one. The permutation maps the
whole gauge group onto itself as a set, all 16384 elements of it, and the action is
measured against the Calderbank-Shor-Steane span a class is read modulo rather than
against the operators the matrix happens to carry. The derived tuple was itself
measured against the one the framework this package replaces publishes for the same
family, wire for wire: a derivation is only worth calling one if it lands on the
published permutation rather than on some other involution of the same layout. The qLDPC
case went the other way: `qldpc_code` takes a caller's own parity-check matrix pair
and derives its logical operators, so the general form of what
`bivariate_bicycle_code` was already doing for its own pair is now reachable
directly, and six routes arrive at a record without the caller writing the
matrices' consequences down -- the colour patch, the torus, the bicycle family, the
punctured Reed-Muller family on the Boolean cube, the tesseract's subsystem
reading and a caller's own pair. What the
colour record changes about the layer above it is the shape of a check: a face on
the triangular patch's edge spans four qubits and a face in its bulk spans six, so
`triangular_colour_code` is the record whose checks reach weight six, and its two
check blocks are literally one matrix because every face carries both stabilizers.
The rotated surface patch also has checks of two weights, but its reduced checks
are weight two, which is a boundary wire pair rather than a face that could have
carried more. A check still names one ancilla and one CNOT direction,
so a weight-six face is six CNOTs onto one ancilla rather than a second ancilla,
and the patch stays inside the protocol. It is also the first record here whose
matrices are derived from a rule about its lattice rather than transcribed, which
is what keeps a declaration from being copied out of another framework's source;
`tests/qec/test_colour_code.py` holds the derivation to the closed forms its
distance implies, and it is the record's own numbers rather than a comparison
against another framework's patch that stand behind them.

The periodic route reaches the same protocol from the other side, and it changes
the shape of the *layout* rather than the shape of a check. `toric_code` identifies
the opposite sides of a square grid, so no face is cut down, every check has weight
four at every size, and the record's distance search bound is the linear size
itself. Two consequences matter downstream. First, the code leaves **two** logical
qubits, so `css_code_matrices` reports four logical operators and
`build_memory_circuit` declares two observables -- the first layout here with more
than one, which is what `tests/qec/test_toric_code.py` holds against the rank of
the record's own rows rather than against a stated ``k``. Second, graphlikeness
becomes a property of the **noise declaration**: a single-round model is graphlike
under any noise, because a Z-type check has a detector at each round boundary and
an X-type check only in the interior rounds, and one round has no interior. Add a
Y fault and run at least two rounds and that fault's X and Z halves land on
different round boundaries, so a mechanism reaches weight four and the matcher
refuses a model it accepted one round earlier. `tests/qec/test_toric_memory_execution.py`
measures both sides at the same code, round count and fault rate rather than
recording the refusal alone, and it feeds the refused side to the decoder that
does answer it: on the two-round declared model every one of three hundred
draws is decoded, the empty syndrome included, every correction is required to
reproduce the syndrome it was handed, and the observables it predicts give a
logical rate of 0.0033 and 0.0100 against the model's own exact 0.1000 and
0.0933. The same file records the rate at which that stops being true -- 0.2667
against 0.3588 at five percent -- so the ordered-statistics pass is not read
there as a suppression claim.

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
declared wires: a detector's terminal readouts must name declared data wires and
its syndrome measurements must name declared ancilla wires inside the configured
rounds, and an observable's readout must name declared data wires and match the
code's declared logical operators. The validation is membership-based. It does not
check that a detector names the *same* check in each round, that a terminal
detector's data wires are the support of the check it belongs to, or that the
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
under a `PhenomenologicalNoise`: one mechanism per data wire per round, and one
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

Both routes read the same four families the noise record states, because both read
supports rather than executing anything: an X fault reaches the Z-type checks, a Z
fault reaches the X-type checks, a Y fault is the two of them at once and reaches
both, and a check's syndrome bit flipped at readout reaches that check's detector.
An earlier revision of this document said both routes carried one fault family and
that an X-type check therefore contributed no row; that was true when the matrix
route landed and stopped being true when `PhenomenologicalNoise` gained its
`phase_flip` and `both_flip` families, and it is corrected here rather than left
standing.

The **sampler** was narrower than the model in the same way, and stopped being
narrow for the same reason. `sampling.py` placed a data flip and a measurement
flip, so the phase and Y data rates reached a detector error model without reaching
a sampled record, and every detector the model attributes to those two families
was sampled at exactly zero. What was missing was never the record and never the
channel: the model read all four families, and the stabilizer engine classifies a
channel by the operators it carries rather than by its name, so a Z channel and a
Y channel were both executable. What was missing was the placement -- the data
location placed the X channel and only the X channel. Each of the three data faults
is now placed as its own channel at the one data location, and the Y fault is one
channel whose single non-identity branch carries the Y operator rather than an X
channel beside a Z channel, because the two would fire independently and that is a
different mechanism. The suite states the difference where it lives rather than
only in the marginals: a one-branch Y model and a two-fault reading of it agree on
every single-detector rate and disagree on the pair rates between the two detector
bands, and the sampled pair rates follow the one-branch model. What the sampler
still does not reach is a channel placed after a named gate and a depolarizing or
damping channel, and that narrower state is recorded in the sampling entry's own
limitations rather than approximated here.

Merging is also an operation a caller asks for, not only a step construction
performs. `DetectorErrorModel.merge_duplicate_mechanisms(rule=...)` gives every
shared signature one prior, `DemMergeRule` states the two rules by what they
compute -- `INDEPENDENT_PARITY` for mechanisms that are independent and coincide,
`CLAMPED_LINEAR_SUM` for the sum of the group clamped at one -- and
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
the sense upstream gives that name -- one entry per mechanism, mechanisms sharing
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
exactly one is admitted -- the group then fires every shot -- and a group of one
member excludes nothing and behaves as if it were unstated. A group whose
probabilities sum *above* one is refused rather than renormalized: renormalizing
would change every member's stated rate and leave nothing to read back, so the
refusal names the id and the sum. `stated_ids()` and `exclusive_groups()` are the
model's statement of what it excluded -- empty and empty respectively exactly when
it is independent -- and `error_ids` is the label vector, whose values are opaque
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
matrix route rather than about the memory circuit -- the memory circuit's
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
exact marginal, lands between 0.29 and 0.65 of the marginal tolerance -- it
reproduces the marginals by construction -- while its pair rates miss by 10.7 to
23.6 times the pair tolerance, against a largest true pair covariance of 0.0284
to 0.1794.

The circuit route is a derivation, not an execution, so it is bounded by the
detector error model's own cost rather than by the statevector amplitude ceiling.
A mechanism's flip set is read off the circuit's detector and observable layouts,
from the code's check types and the frame the experiment is prepared and read out
in, and nothing is lowered or executed. A rotated surface code is `distance**2`
data wires plus one ancilla per check, and reading the layout costs one pass over
its entries per mechanism: `distance=3` is 17 wires and 45 mechanisms, `distance=5`
is 49 wires and 225 mechanisms, and `distance=7` is 97 wires, 336 detectors and
637 mechanisms, and all three build in under a quarter of a second.

Three shapes are still refused with the reason the injection engine stated,
because a model built from a record that misdescribes the program it stores would
be wrong in a way no reader could see: a source missing the round loop anchor, a
source missing a check's measurement anchor, and a detector whose syndrome
reference names an ancilla no check owns. What the derivation gives up is the
executed route's guard against a record that is merely wrong about a mechanism the
program does contain. That guard cannot be recovered without executing, so the
oracle it belonged to survives as `_forced_signature` and the suite holds the
derivation against it location by location: every mechanism of a repetition code,
a Steane code and a rotated surface patch, over every fault family, in both
readout frames. The distance-7 patch is asserted to build for exactly that reason --
it is what fails if the derivation is replaced by an execution again. The executed
route is bounded where the derivation is not: `distance=4` is 31 wires (2**31
amplitudes), and `distance=5` is 49 wires and fails on the allocator.

The matrix route is bounded by none of that either. It reads a support rather than
executing a program, so its cost is the number of nonzero matrix entries times
the round count and a distance-5 or distance-7 patch reaches a model as easily as
a distance-2 one. The two routes answer different questions and their geometries
are allowed to differ, which is why the difference is pinned rather than glossed.

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
style `to_stim_text()` emits. The gauge gap is stated precisely because the
subsystem record does not close it: the record declares which generators are gauge,
the automorphism that preserves them and the logical action that automorphism
implements, but this package has no round structure and no detector model in which
a gauge generator is measured, so a subsystem code has no memory experiment here
and no decoder registered for one. The sweeps' own reading of the gap is unchanged
by the automorphism being published: a permutation of data wires is not a measured
gauge generator, and no detector in this package reads one.

## Reading a model back

`dem_construction.py` answers what model a circuit and a noise record define.
`dem_circuit.py` answers the other direction, and it can, because a model is
already a circuit's worth of arithmetic stated in another vocabulary: a mechanism
is a set of detectors and observables it flips together with a probability, and
mechanisms are independent unless an error id says otherwise, so a model is a
register of detector wires and observable wires carrying one Pauli frame per
fault. `circuit_from_detector_error_model(model)` returns that register as a
`CircuitIR` -- one wire per detector followed by one per observable, one
Pauli-frame channel per fault group, one measurement per wire -- and
`detector_error_model_from_circuit(circuit, num_detectors=...)`, published as
`DetectorErrorModel.from_circuit`, reads it back.

What it returns is not the circuit the model came from, and no reading of a model
can produce that circuit, because the model does not contain it. A detector error
model is a quotient: it states which detector parities a fault flips and never
which measurements compose a detector, so the syndrome-extraction circuit, its
gate sequence, its depth and its ancilla layout are all outside what the model
says. Two different memory experiments of the same distance and the same noise
record produce the same model, and a reader that returned a circuit for one of
them would be asserting a fact the model never carried. What the reader returns
instead is the canonical detector-level circuit every model *does* determine. The
detector and observable split is outside it for the same reason -- a wire index
says nothing about which side of the split it falls on -- so the detector count is
the caller's to state, and a caller who states the wrong count gets a model whose
mechanisms are right and whose split is not rather than an error. The two
functions are meant to be read as a pair.

The round trip is exact, including the numbering of the error ids, and it is
exact rather than approximate because the two halves of a frame are read from
different parts of the instruction. A frame's wire list and its branch operators
say which targets a fault flips; the mass of each branch is declared in the
instruction's `probabilities` parameter. That division is not a convenience.
A branch is written `sqrt(p) * U`, so recovering the mass from the operators
means squaring a square root, and squaring a square root is not the identity on a
double: over 200000 uniform draws it returned a different double for 47.1% of
them, by about one binary unit in the last place (a measured worst relative
difference of `2.2e-16`, inside the one-part-in-`10**15` bound the stim
interchange already carries), and `0.01`, `0.5` and `0.9` are among the values
this domain actually uses. The declared mass is therefore the statement of record
and the operator is checked against it, which is what makes the round trip return
the model it started from rather than one a rounding away from it; a program
whose two halves disagree beyond their stated precision is refused instead of
averaged. This is the division `sampling.py` already writes its noise
instructions with, where a channel states its rate and carries the operators that
rate was turned into.

The realization executes. Every instruction is a Pauli-frame channel, which is
what `flagquantum.simulation.stabilizer` samples, so a model reaches a second
route to the same rates that shares no arithmetic with `dem_sampling`: the
model's own route draws one uniform per mechanism and exclusive-ors the
signatures that fired, and this one goes through a stabilizer engine. Held
against the rates the model *states* -- not against a second sample of the same
route -- the realization's sampled detector and observable rates departed by at
most 2.01 standard errors over the hand-written models and the memory-circuit
models of the three codes that suite builds together, at 100000 shots with one
seed, and a group
whose masses sum to exactly one flips the detector its members share on every one
of those shots rather than on 99.99% of them. At probability one the check is
exact instead of statistical, so a frame placed at the wrong wire fails as a
wrong bit rather than averaging into a rate.

Two things are refused rather than dropped. A mechanism at probability zero has
no branch: every branch mass of its channel is zero, so the operator it would
apply is zero and no shot fires it, and a reader could not tell it from the
identity. Dropping it would silently make the realization stand for a model with
fewer mechanisms than the one it was read from, so `circuit_from_detector_error_model`
names the mechanism and refuses. And exclusivity is executable rather than
merely stated: the members of an id-carrying group are the branches of one
channel, so a realization draws at most one of them per shot, which is the same
statement `dem_alternatives.py` makes in the model's own arithmetic. The
realization does not make a correlated model transcribable -- `to_stim_text()`
refuses an id-carrying model here exactly as it does anywhere else, because the
format reads every error instruction as an independent mechanism.

The evidence is `tests/qec/test_dem_circuit_round_trip.py`, which asserts the
round trip on every hand-written model and on the memory-circuit models of a
repetition code, a Steane code and a rotated surface patch at one round and at
three; holds the realization's sampled rates against the rates the model states;
pins the `sqrt`-then-square loss as the number that forced the declared mass
rather than describing it; and exercises each refusal by constructing the program
that would provoke it. The chain to the external reference is transitive rather
than repeated there: `test_dem_stim_interop.py` pins the text these models are
written to against stim's own reader, and `test_dem_stim_reference_rates.py` pins
the model's rates to a stim reference circuit, so the realization-to-model edge
is the one this file adds.

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

One representational boundary is explicit and enforced, and it is now one rule
rather than two gadgets. A `CodeCheck` states one ancilla and one **coupling order**:
the Z-factor pairs first, each controlling from a data wire in the stabilizer's
Z-factor support into the ancilla, then the X-factor pairs, each controlling from
the ancilla into a data wire in the stabilizer's X-factor support. The order is not
a convention -- the boundary between the two runs is exactly what the emitted `H`
pair wraps -- and it is what makes one ancilla measure the product of the two
factors. A pure check is the case whose second run is empty, so the Z-type gadget
(control from the data, ancilla prepared in `|0>`, read in the Z basis) and the
X-type gadget (the same with an `H` pair around the second run) are the two ends of
one rule and the source they emit is byte-identical to what it was. An earlier
revision of this file recorded that a mixed stabilizer needs a second ancilla;
that reading was wrong and is retracted, because the second ancilla buys nothing
the second CNOT direction does not already give.

`RotatedSurfaceCode` is the first code here that needs both check types. Data
qubits occupy wires `0..distance**2 - 1`, indexed so lattice site `(i, j)` is wire
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
draws each shot from the full output distribution, so a 17-wire patch costs a
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
defect would charge the cheaper of the two parallel edges -- for mechanisms of
`0.1` and `0.2` that is `log 4`, or `1.386` -- for a fault whose combined parity
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
every scope limit of the model it consumes. It also inherits the fault family: a model read from matrices states only
the bit-flip family, so the matching decoder is matched to that model and not to
a channel the matrix route cannot describe. Measured on the two codes this layer
models at three rounds and two percent noise:
the repetition code at distance three gives 15 edges of which 6 reach the boundary
and every weight is 3.8918, and the rotated surface code at distance three gives 45
edges of which 20 reach the boundary, with six of the weights at 3.1991 and the
other 39 at 3.8918. Every mechanism in both models flips one detector or two, so
neither model reaches the hyperedge refusal. A matrix-route model is graphlike
only while it stays inside one round: at two rounds the middle data wire of the
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
rather than a silent truncation. Twenty is a real ceiling on this route and not a
formality once the patches are wide: measured at a physical rate of 0.02 over one
thousand shots of `RotatedSurfaceCode(distance=5)`, the widest syndrome held 21
detection events and the default budget declined one shot in a thousand, while at
0.01 over twenty thousand shots the same patch never passed 16 events and the
default never fired. The derivation and the model have no ceiling of that kind --
a distance-7 patch is 336 detectors -- so the budget, and not the model, is what
bounds the patch a decode of this route can cover, and it is recorded as an owned
gap rather than presented as a property of the model. Raising it trades shots for
enumeration, not memory: the widening of the model does not change this decoder's
cost per syndrome.

Detectors that no chain of mechanisms connects are refused with a `CapabilityError`
naming them, rather than answered partially.

`MatchingDecodeResult` is deliberately not the repetition-code `DecodeResult`. A
`Correction` carries a wire in `{0, 1, 2}` and an X basis only, and
`PauliFrame.from_corrections` is enforced to agree with it, so neither record can
express a surface-code correction over `distance**2` data wires in the Z basis.
The result states the predicted observables, the mechanisms selected in graph
order, and their total weight, and the decoder therefore does not implement the
repetition-only `Decoder` protocol either. The one thing built on this result is
the windowed decoder in `sliding_window.py`, whose own section follows: it returns
this record unchanged, because a band's answer is the same kind of answer as the
whole graph's. Nothing else is: there is no logical-error-rate estimator, no
threshold scan, and no connection to the memory-experiment result records. The
implementation imports `heapq`, `math`, `dataclasses`, and `numbers` and nothing
else, so no new dependency is introduced; `stim` and `pymatching` are not imported
by it, and the cross-check against `pymatching` is a separate module behind the
`pymatching` extra rather than a build dependency. That cross-check is
`flagquantum/qec/adapters.py`, and what it established is stated there rather
than here: the two decoders agree on the cheapest weight of every syndrome and on
the observables wherever the cheapest explanation is unique, and a tie is
uncomparable because PyMatching's own arithmetic is narrower than this module's.

## Deciding a history a band at a time

`sliding_window.py` exists for the one thing the exact matcher above cannot do,
and the reason is the budget rather than the model. The matcher enumerates the
ways to pair the defective detectors, so its cost is exponential in the defect
count and its `max_defects` is what bounds the patch a decode can cover. A
syndrome's defect count grows with the number of rounds, and the package's own
distance-three patch carries well over a hundred defective detectors over 200
rounds at three percent noise -- 110 to 191 across forty seeded shots against a
budget of 20 -- which is far past any enumeration the matcher will pay for and is
also what a decode of a real experiment's history looks like. Upstream states this
family as a decoder named `sliding_window`; this module is the same technique over
a decoding graph this package already builds.

The detectors are cut into windows of `window` consecutive detectors advancing by
`commit`; each window is decoded by `MinimumWeightMatchingDecoder` over the graph
of the mechanisms that begin inside it, and the mechanisms of that answer whose
first endpoint lies in the window's leading `commit` detectors are committed while
the rest are left for a later window, whether or not the window reaches the end of
the graph. A committed mechanism toggles its endpoints
out of the residual syndrome, and each window decides its band against what the
windows before it left behind. A mechanism therefore belongs to exactly one
window: the one whose band holds its first endpoint.

Three facts make a band's decision the same decision the whole graph would have
made about it. The window holds every mechanism whose first endpoint is inside it,
and a mechanism incident to a detector of the band begins at or before that
detector, so every mechanism incident to the band is in the window. Every such
mechanism is committed, because committing is exactly the test `first < start +
commit`. And the window's answer is a T-join of the window's syndrome, so a band
detector has odd degree in that answer precisely when the residual called it
defective. Together they say the committed mechanisms toggle each band detector
exactly as often as it needs, so each band is explained as it is decided and the
residual is empty once the last window has been decided. That is checked rather
than assumed: a non-empty residual raises `CapabilityError` naming the detectors.
The check is the reason a defect in the committing arithmetic fails loudly instead
of returning a correction that merely looks like one, and it is a check rather
than an argument because the induction above rests on each window's answer being
what the window's syndrome needs, which is the matcher's claim and not this
module's.

A mechanism that begins inside the window and ends past its last detector is still
a mechanism of the model, and the window offers it to the matcher as a step to the
model's boundary rather than dropping it. Dropping it is what would leave a
detector of the band with nothing to pair against, so keeping it is what makes the
band answerable at all. It is never committed either, and the floor is what proves
that rather than a separate test: a mechanism the window cannot follow ends at or
past the window's last detector, so its first endpoint would be at or past
`start + commit` if it were inside the band, which the floor puts at or before the
window's last detector. What reaches the caller is therefore always a mechanism
the model has. The
window's answer does treat the region past its last detector as boundary, which is
the approximation a window makes: a mechanism of that region is seen as reaching
the model's boundary rather than as continuing, which can make a band's decision
cheaper than the history supports. What survives the approximation is the record.
Every committed mechanism is a mechanism of the model, and the committed
mechanisms have the whole syndrome as their odd-degree set, so they are a genuine
T-join and their total weight is at least the exact matcher's for the same
syndrome. Both are measured rather than asserted, over four codes and every legal
window, along with the identity that fixes the whole design: with one window
covering the whole graph the decoder is `MinimumWeightMatchingDecoder` mechanism
for mechanism, which is what makes it a replacement for the matcher rather than a
second implementation beside it.

The window has a floor, and the constructor refuses a narrower one by name. A
mechanism begins at or before the band's last detector and spans at most the
model's widest mechanism, so with `window >= commit + span` every mechanism
starting inside the band ends before the window's last detector and the window
never treats it as a boundary step. `span` is measured from the graph rather than
declared, and a window that already reaches the end of the graph is admitted
whatever its width, since nothing is past its last detector then -- which is the
admission that keeps the one-window identity reachable. The band is committed
whether or not the window reaches the graph's end: a window is not widened into
authority over the detectors past its band, and the one-window identity survives
because with a single window the band is the whole graph. The default band is one
`span`, which for a memory circuit is one syndrome round because the widest
mechanism of a memory model connects a detector to its counterpart one round
later; the default window is three bands.

What the narrow window buys is finite work per syndrome: the defect budget bounds
a window's syndrome rather than the whole history's, which is how a syndrome the
exact matcher refuses becomes a syndrome this decoder answers. On the four long
syndromes measured -- 122, 164, 480 and 1600 detectors, where the exact matcher
declined 11, 25, 40 and 40 of 40 shots -- the windowed decoder answered all 40 at
a band of one span and a window of two. What it costs is that a narrow window is a
decision about a window: the mechanisms it selects can differ from the exact
matcher's even where the observables agree, and the observable agreement is bought
with the window's width rather than guaranteed. Agreement improves with the width
and not monotonically at every width: over five short codes and 400 shots each, at
a window of two bands the disagreements were 0, 1, 2, 7 and 12, at three bands 0,
2, 1, 2 and 7, at four bands 0, 0, 0, 0 and 4, and at five bands the widest of
them refused one shot of the 400 while the others agreed on every one -- so width
buys agreement and not monotonically. Widening is not free in the other direction
either, because a wider window also collects more defects and reaches the same
budget the exact matcher has: on the 480-detector patch a window of 48 detectors
answered all 40 shots, a window of 96 answered 24, and a window of 160 answered
one, at the default budget of 20 defects. No threshold and no logical error rate
is estimated here, and no accuracy claim is made for a window narrower than the
whole graph.

There is no streaming entry point and no chunk seam to feed one. The decoder takes
a whole syndrome and answers it band by band, and the seams a streaming decoder
needs are the `DemChunkSpec` family upstream states and this package does not
have. The module imports `math`, `dataclasses`, `numbers`, `collections.abc` and
this package, so no dependency is added and the window arithmetic costs nothing
beyond the matcher calls it makes.

## A decoder for the hyperedge

`bposd.py` is the second decoder in this layer and the first one that answers a
model the matcher refuses. The refusal it exists for is structural rather than
numerical: a mechanism that flips three or more detectors has no pair-graph, so
`from_detector_error_model` raises a `CapabilityError` naming the detectors, and
before this module a caller holding such a model had nothing to hand the syndrome
to. The two halves are belief propagation and ordered statistics, and they are
separate on purpose. Belief propagation estimates one log-likelihood ratio per
mechanism by passing min-sum messages between detectors and mechanisms until the
hard decision reproduces the syndrome or the pass budget runs out; ordered
statistics then solves the syndrome exactly over the mechanisms that estimate
favours most, which is what turns an estimate into an answer with a stated
guarantee.

The iteration is min-sum with a damping factor rather than sum-product. A
detector sends each mechanism the magnitude of the smallest opposing message on
that check, signed by the parity of the check's other messages, and the caller's
`scaling` weights what it sends. A check with a single mechanism has no opposing
message, so its magnitude would be the message limit and the mechanism is forced,
which is the correct reading of a detector only one mechanism can flip; the
magnitude is capped rather than left infinite so that the forced message stays a
finite number a caller can inspect. A probability of one or zero would make the
prior a log-odds ratio of infinity in one direction, so `__post_init__` floors it
by `_MINIMUM_PROBABILITY` instead of letting the arithmetic diverge. The result
states whether the passes converged and how many were spent, because a
non-converged estimate is still a usable ranking for the second half while a
caller who is told nothing would read it as a converged one.

Ordered statistics ranks the mechanisms by the estimate and elects an information
set by Gaussian elimination over GF(2) in that rank order: the earliest
mechanisms that are linearly independent. Every mechanism outside the set is held
at zero, which reproduces the syndrome exactly by construction and need not be
its most likely explanation. That is the whole of the claim, and it is why this
decoder is not described as optimal. Where the syndrome is not in the span of the
model's mechanisms at all, the elimination leaves a row with no pivot and the
decoder raises `CapabilityError` rather than returning a correction that only
looks like one.

What the decoder reaches is measured rather than asserted. On the one-round
Steane model it returns the least weight and the most likely observable for all
64 syndromes. On the one-round rotated surface code it carries more weight than
the least-weight explanation on five syndromes of 256 -- by 0.6904 on two and
2.9444 on three -- and differs from the most likely observable on three of those
five, so its entire gap to an optimal decoder on that model is those five
syndromes and not a property of the code. Against the matcher on the model both
accept, a two-round repetition code, the two reach the least weight on every one
of the 32 syndromes and disagree on six, and every one of those six is a tie the
model itself does not resolve, so the disagreement is two readings of one optimum
rather than a mistake on either side. `tests/qec/test_bposd_decoder.py` pins each
of those numbers, checks the syndrome-consistency invariant on every result, and
uses brute force over the model's own mechanism sets as the independent reference
for both the least weight and the most likely observable.

It is deliberately not registered in `registry.py`, and the reason is a protocol
promise rather than an unimplemented constructor. `DetectorErrorModelDecoder`
says that a decoder can be built from a model and exposes the `DecodingGraph` it
built, and a model carrying a three-detector mechanism has no such graph:
`DecodingGraph.from_detector_error_model` refuses it as the same hyperedge this
module exists to answer. Adding a `from_detector_error_model` classmethod and a
`graph` property that raised on the models this decoder is for would put a class
under a name whose stated contract it cannot satisfy, which is a worse failure
than the one the registry was built to prevent, so the class is constructed
directly while the matcher, the windowed reading of it and the cross-check are
reached by name. Naming it
would mean widening or splitting the protocol first, and that is a decision about
the registry rather than a missing method on this decoder.
`tests/qec/test_bposd_decoder.py` measures the exclusion rather than repeating
it: the matcher's instance carries the graph the protocol promises, this
decoder's carries none, the class is not an instance of the protocol, and
`register_decoder` refuses it for the missing constructor while the name it was
offered stays free.

The module imports `torch` and the standard library and nothing else, so no
dependency is added and no second source of truth for linear algebra appears; a
test starts a fresh interpreter and measures that. It is a decoder for one model
at a time: there is no batching, no streaming and no logical-error-rate estimator
and no threshold sweep, and it is not wired to the stim sampling path, so a
syndrome it decodes comes from the caller rather than from this package's
sampler. The sliding window is the windowed route in `sliding_window.py` and not
this decoder: this one answers a hyperedge model a matcher cannot graph, and that
one bands a graph the matcher can search.

## Reaching a decoder by name

`registry.py` is the factory half of the CUDA-Q QEC decoder surface: upstream
reaches a decoder through `get_decoder(name, H_or_dem_text_or_sparse_matrix,
**options)` and registers one with a decorator, and this module offers
`get_decoder(name, source, **options)`, `register_decoder(name, *,
replace=False)`, `decoder_names()`, and the two protocols those three are
written against. `AUTHORITY_NAME`, `CROSS_CHECK_NAME`, `SLIDING_WINDOW_NAME`
and `BELIEF_PROPAGATION_OSD_NAME` name the four registrations this package
ships: the authority, the optional cross-check, the same authority read through
a window, and the decoder that answers the model the matcher refuses.

Two protocols rather than one is the shape the family took when the fourth name
was added, and it is what makes that name honest. `DetectorErrorModelDecoder`
states what every registered name promises: `decode`, returning
`DetectorErrorModelDecodeResult`, the observables the correction flips.
`GraphlikeDetectorErrorModelDecoder` states the narrower promise the three
matching names keep and the hyperedge decoder cannot: `graph`, the pair graph
that was built. The graph route of `get_decoder` is that sub-protocol's only
consumer, and it measures `graph` in the class's constructor signature before it
hands a graph in, so a name whose implementation searches no pair graph is
refused with its reason instead of failing inside a constructor. Splitting the
promise rather than widening the class was the alternative to leaving the decoder
out, and it is the one this module took.

Three things about it are narrower than upstream on purpose, and each is a
decision rather than an omission.

The source argument is a *carrier*, not a decoder setting, which is why the three
accepted forms are the three a caller can hold a model in: the detector error
model, stim's text for one, and the decoding graph the model defines. A model is
lifted through the class's own `from_detector_error_model`; a graph is passed to
the constructor, because the graph is already the thing a matcher searches and
rebuilding a model from it would lose the observable labels the caller has in
hand, and only a class that takes the graph accepts this carrier; text is read
through `DetectorErrorModel.from_stim_text` first. A
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

The windowed decoder is registered rather than kept out, and the distinction is
what the graphlike protocol promises. It builds the graph the model defines and
holds it, so
it carries the `DecodingGraph` the protocol requires, and its `decode` reads
detection events and returns the same `MatchingDecodeResult` the authority
returns; what it adds is a band width and a window width, and those are
`**options` of one constructor just as `max_defects` already is. Registering it
also does not make the authority ambiguous, because a name reaches one class and
nothing compares two: the windowed name is the one a caller asks for when the
history is longer than the matcher's budget, and the authority's name is the one
it asks for otherwise.

Registration is checked at registration time, while the registering module is
being imported. `register_decoder` refuses a name that is not a non-empty string,
a second registration of a name unless the caller passes `replace=True`, and a
class missing `decode` or `from_detector_error_model` -- the two members every
decoder in this family shares. A `TypeError` at import time, naming the member
that is missing, is a better failure than an `AttributeError` at the first call,
where the name is all the caller has to go on.

`get_decoder` fails closed in three more places. An unregistered name raises
`ValueError` and lists the names that are registered, rather than reaching any
implementation; a source that is not one of the three carriers raises
`TypeError` naming `DecodingGraph`, since that is the carrier a caller is most
likely to have held; and a graph handed to a name whose class takes no graph
raises `CapabilityError` naming the name, the class and the reason, rather than
the `TypeError` the constructor itself would raise, because the registry chose
the route and the registry is what should say so. `**options` goes to whichever
route the source selects and is not filtered here, so passing the text reader's
`use_decomp_suggestions` to the graph route raises rather than being dropped.

The optional implementation is registered whether or not it is installed, so
`pymatching` is part of this package's surface rather than the extra's: asking
for it without the extra raises the error that names the extra, instead of a name
that silently is not there. The adapter module is imported, but it reaches
PyMatching through a function rather than at import time, so `import
flagquantum.qec` does not import `pymatching` -- a test starts a fresh interpreter
and measures that rather than asserting it. No name is preferred over another, so
`get_decoder(AUTHORITY_NAME, ...)` returns the authority wherever the extra
happens to be installed; the cross-check is never reached by accident.
