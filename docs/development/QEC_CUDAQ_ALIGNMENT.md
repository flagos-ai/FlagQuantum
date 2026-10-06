# FlagQuantum QEC ↔ CUDA-Q alignment checklist — the reading

Read this with [`qec-cudaq-alignment-checklist.toml`](../../contracts/qec-cudaq-alignment-checklist.toml)
and [`check_qec_cudaq_alignment.py`](../../tools/check_qec_cudaq_alignment.py)
open. The TOML is the checklist; the Python tool is the check that it still
describes a real repository; this document is the part a person reads once.

```
python tools/check_qec_cudaq_alignment.py        # 313 checks, exit 0 or 1
python -m pytest tests/unit/test_qec_cudaq_alignment_check.py -q
```

The first command is run by CI, so a checklist row that stops describing this
repository fails the build rather than aging quietly. The second damages one
section of the checklist at a time and requires the checker to fail for the
stated reason; it is the evidence that the first command is a check and not a
formality.

## 1. What is being aligned to what

The question this answers is "which framework's functionality does
`flagquantum/qec` have to line up with". The answer has one target and two
supporting standards, and they are not interchangeable:

| Layer | Aligned to | Why not the others |
| --- | --- | --- |
| Product surface — objects, entry points, code families, decoders | **CUDA-Q QEC** (`cudaq-qec`, now `NVIDIA/cudaqx` `libs/qec`) | It is the only live framework with a QEC library of this shape, and `contracts/cudaq-parity-matrix.toml` already names it as the baseline. |
| Detector error model bytes | **Stim** DEM text, stim 1.16.0 | The format is stim's. Aligning it to CUDA-Q would mean inventing a second format nobody else reads. |
| Decoder correctness | **PyMatching** as cross-check, never as authority | Correctness evidence has to be independent of the implementation under test; the self-implemented matcher is the authority. |
| Logical operations (`logical.*`) | **Qualtran**, as an adaptation | CUDA-Q's `logical.*` is a preview namespace of patch/gadget/protocol records. The semantic core is the target; the preview records are not. |

Three things this checklist explicitly does **not** align to:

1. **`qiskit-qec`.** Archived, read-only, early-stage, breaking API, and its PyPI
   project returns 404. `API_CHANGE_PROPOSAL_048` L47–51 already recorded it as
   not a viable parity target.
2. **`nv_qldpc_decoder`.** It ships as `libcudaq-qec-nv-qldpc-decoder.so` under
   the NVIDIA Software License — closed source. Align to its *interface
   position* (a named decoder behind a registry) and implement with Apache-2.0
   PyMatching/LDPC-BP. Note that CUDA-Q QEC's own chromobius and pymatching
   plugins are in-tree open source, which is direct precedent for integrating
   rather than reimplementing.
3. **CUDA-Q's GPU decoder performance.** The matrix's own reason for
   `qec_decoder_family` says integrate first, accelerate later through the
   FlagOS kernel path.

## 2. The 15 alignment rows

`floor` is the priority this repository should hold the row at; where it differs
from the matrix's `priority`, the row states why.

| Row | Status | Floor | Matrix row | The gap in one line |
| --- | --- | --- | --- | --- |
| `qec_code_record` | partial | now | `qec_code_library` | Five records declared — repetition, rotated surface, Steane, triangular colour and square-lattice toric — and each feeds both the circuit and the matrix route in either readout basis; the colour and toric records are derived from a rule about their lattices rather than transcribed, the toric record is the first that leaves two logical qubits, and the record set is still one code per family, so a qLDPC or bivariate-bicycle code has no record and a mixed-type observable is refused. |
| `qec_detector_annotations` | partial | now | — | Layouts beside the source, not annotations in the kernel; no measurement handles. |
| `qec_syndrome_extraction_owner` | partial | now | — | `extract_syndrome` is in the CUDA-Q Logical preview, not CUDA-Q QEC. Both routes are cudaq-qec's own names; the inventory line it corrects is the only thing left. |
| `qec_dem_construction` | partial | now | — | Construction is exact on both routes and the context object landed; no kernel-annotation route, so no X/Y fault family from a kernel body. |
| `qec_dem_matrices_and_rates` | partial | now | — | Orientation matches, the error-id column and the context object landed; no per-mechanism rates vector, no canonicalization. |
| `qec_dem_merge` | aligned | now | — | Closed: both stated rules, the uniqueness predicate and the refusal are present and enforced at the decoder. |
| `qec_dem_chunking` | absent | later | — | No chunks and no seams, so the windowed decoder decides a whole history band by band rather than consuming a stream. |
| `qec_dem_text_interchange` | partial | now | `qec_stim_integration` | Both directions present and independently checked; both separator readings offered under upstream's flag; input end is narrow. |
| `qec_stim_sampling_join` | partial | now | `qec_stim_integration` | The join landed; the noise grammar is one channel at two placement classes, so arbitrary annotated circuits are still declined. |
| `qec_decoder_family` | partial | now | `qec_decoder_family` | A DEM-consuming matching decoder, a windowed reading of it registered beside the exact one, its PyMatching cross-check, a name-keyed registry, and a belief-propagation decoder with ordered statistics all landed; no streaming entry point over a chunk seam, no name for the hyperedge decoder, no plugin boundary. |
| `qec_decoder_configuration` | absent | later | — | A band and a window are constructor options of one decoder rather than a configuration schema over several; a schema waits until a selection has to carry settings a caller cannot state at the constructor. |
| `qec_dialect` | absent | later | `qec_dialect` | Needs an internal IR level to carry the structure. |
| `qec_logical_operations` | partial | later | `qec_logical_operations` | Product rotation landed as a code-declaration operation: a candidate logical product is certified against the code's own checks and one observable's partner is derived over GF(2); lattice surgery and distillation are still absent. |
| `qec_transport_and_objectives` | absent | later | `qec_transport_and_objectives` | Hardware-shaped; out of scope until a neutral-atom target exists. |
| `qec_stim_user_migration` | partial | next | `qec_stim_user_migration` | The guide landed and is executed by its own test; its upstream counterpart is CUDA-Q QEC's own Stim surface rather than a page to translate, and the two halves it documents — the `^` reading and the registry's missing belief-propagation entry — are what keep it short of aligned. |

Everything a `supported` row would need is deliberately *not* claimed here. One
row is `aligned` and the rest are not, and the single `aligned` row is the one
whose upstream surface has no item left unaccounted for, symbol by symbol. The
checker enforces the floor of that bar rather than the bar itself: an `aligned`
row must name a symbol that resolves and may not sit on a matrix row that says
`unsupported`. That a row clears the mechanical bar is therefore not the
argument for calling it aligned; the argument is in the sections below, where
each row states what it closed and what it did not, and a row that still has a
half to name stays `partial`.

**What `qec_decoder_family` closed, and what it did not.** The row's order was
"decoding graph with `log((1-p)/p)` edge weights and observable labels, a
self-implemented minimum-weight matcher, then the optional PyMatching adapter as
the independent cross-check." All three are now in the repository:
`flagquantum/qec/decoding_graph.py` turns a detector error model into a graph
whose boundary node is one past the last detector, and
`flagquantum/qec/matching.py` decodes a syndrome exactly by enumerating the ways
to pair its defective detectors and to route any number of them to that
boundary. Neither imports `stim`, `pymatching`, `networkx` or `scipy`; the graph
carries the observable labels a prediction needs, which is the part the
repetition-profile decoders never had to represent.

Two refusals keep the row honest rather than merely larger. A mechanism flipping
three or more detectors is a hyperedge and raises `CapabilityError` naming the
detectors, because projecting it onto a pair would drop a detector and the
matcher would then return a correction that does not explain the syndrome. The
pairing is enumerated, so the decoder takes a defect budget (twenty by default)
and refuses a larger syndrome rather than returning a pairing that only looks
cheapest. Both are the fail-closed convention, not stubs.

The third item in that order — PyMatching as an independent cross-check behind
the `pymatching` extra — is now in the repository as
`flagquantum/qec/adapters.py`, and what it established is narrower than "the two
decoders agree", because the two instruments are not equal. PyMatching reports
`3.9020747171643912` for a mechanism this package states as
`3.9020746947749574`, so the graph PyMatching minimizes over is not exactly this
one and two explanations closer than that difference can be ordered differently
on the two sides. The comparison is therefore of two *selections*: the cheapest
weight on every syndrome, the observables wherever the cheapest explanation is
unique, and the two refusals as the same set. Ties are uncomparable rather than
evidence against either implementation, and the file pins one tie of the
distance-three repetition code by hand so that the clause is a demonstrated fact
and not a place a disagreement could hide. The translation refuses two graphs
instead of approximating them — a detector pair carrying two mechanisms, which
PyMatching's `independent` strategy would collapse and thereby lose the
logical-label difference, and a detector that no mechanism flips, which
PyMatching cannot represent because it infers its detector count from its edges.
The row named a fourth item after those three — a registry, so that
`get_decoder(name, ...)` reaches a decoder and a call site holding a model does
not have to know which class implements it — and that has since landed as
`flagquantum/qec/registry.py`. It is the factory half of the baseline's decoder
surface and deliberately not more.

The registry's source argument is a *carrier* and not a decoder setting, which is
why it takes the three forms a caller can hold a model in: the detector error
model, stim's text for one, and the decoding graph the model defines. A
parity-check matrix is not among them, although upstream's
`H_or_dem_text_or_sparse_matrix` is, because lifting a matrix would also mean
choosing the noise model and the round count that
`DetectorErrorModel.from_code_matrices` reads rather than defaults — so a factory
that lifted a matrix would be choosing the noise the caller decodes against, and
the caller who holds the matrix and the noise together does the lifting. The
registered family is detector error model decoders alone: the repetition-code
decoders take an ordered syndrome history rather than detection events, and one
name space over two input protocols would make a name mean one of two things, so
they stay directly constructed.

Three refusals, all at the point where the mistake is, keep the registry from
being a look-up table with a fallback. An unregistered name raises and lists the
names that are registered rather than reaching any implementation. A class
missing `decode` or `from_detector_error_model` is refused while the registering
module is being imported, which is the same place upstream refuses it and a
better place than the first caller, where the name is all the caller has to go
on. And a second registration of one name is refused unless the caller says
`replace`, because two classes answering to one name is a choice the registry
cannot make on the caller's behalf.

The optional implementation is registered whether or not it is installed, so the
name `pymatching` is part of this package's surface rather than the extra's, and
asking for it without the extra raises the error that names the extra instead of
a name that silently is not there. No name is preferred over another: the
registry returns the class a name is registered against, so the authority is
returned for its own name wherever the extra happens to be present, and the
cross-check is never reached by accident.

The row's scope has since grown by one more piece, and it is a capability of
its own rather than a fifth item in this row's order.
`flagquantum/qec/bposd.py` decodes a detector error model by iterating min-sum
messages over its mechanisms and then solving the syndrome over the most
error-prone independent ones, so a mechanism flipping three detectors — the
hyperedge the matcher refuses by design — is answered instead of declined. The
two decoders are compared on the model they both accept, a two-round repetition
code with 32 syndromes: they reach the least weight on every one of them and
disagree on exactly six, and each of those six is a tie the model itself does not
resolve, so the disagreement is two readings of one optimum and not a mistake on
either side. On the one-round Steane model the hyperedge decoder reaches the
least weight and the most likely observable for all 64 syndromes. On the
one-round rotated surface code it carries more weight than the least-weight
explanation on five syndromes of 256, by 0.6904 on two and 2.9444 on three, and
differs from the most likely observable on three of those five; that is the whole
of its gap to an optimal decoder on that model, and it is why
`qec_belief_propagation_osd_decoder` is recorded at `development_evidence`
rather than higher. It is deliberately not in `decoder_names()`, and the reason
is the registry's protocol rather than a missing constructor: the protocol
promises a `DecodingGraph` view of what was built, and a model with a
three-detector mechanism has no pair-graph at all, because
`DecodingGraph.from_detector_error_model` refuses one as a hyperedge. Adding the
missing classmethod alone would therefore register a decoder that does not
satisfy the protocol it is registered under, so naming it is a protocol decision
that this row records rather than takes, and the capability itself is owned
where it belongs.

The colour patch is the first shipped family whose decoding evidence is this
route rather than the matcher's, and it says what the row's own scope boundary
means in practice. A face on the patch's edge spans four qubits and a face in its
bulk spans six, so one data fault lights three detectors and the model is not
graphlike; `MinimumWeightMatchingDecoder.from_detector_error_model` refuses it
with the same hyperedge refusal it gives the Steane model, while the
belief-propagation decoder answers it. `tests/qec/test_colour_memory_execution.py`
records what that answer is worth: it decodes every one of the distance-three
patch's seven single faults as they come out of an executed circuit and predicts
the observable flip each fault caused, and at distance five it decodes every
syndrome the model itself samples -- four thousand of them -- with each correction
reproducing the syndrome it was handed. The same file states the limit: the
distance-three patch is the Steane code up to a qubit relabelling, so its model's
mechanism-weight histogram is the Steane model's row for row, and nothing here
separates the two families or claims a threshold.

**What `qec_stim_sampling_join` closed, and what it did not.** The row said the
join was the gap: the stabilizer engine executed noiseless Clifford programs and
refused a noise channel, so detection events could only be sampled from the
model, and every rate it produced had to stay labelled DEM-sampled. The join is
now in the repository as `flagquantum/qec/sample_memory_circuit`, which lowers a
memory circuit once, places each noise location the phenomenological record names
at the instruction it belongs to, executes the result on the stabilizer engine,
and reads the detection events and observable flips off the recorded bits.

The placement is the whole content of the join, and it is derived rather than
asserted. A data error is a round-boundary error and goes *before* the round's
first gate, so it opens the frame the round's detectors compare against; a
measurement error goes immediately *before* the readout of the check it corrupts.
Neither position exists in the source, because the bounded hybrid capture refuses
a channel call outright, so the module derives both from the lowered program and
checks the property the derivation rests on: the syndrome loop has to lower to
`rounds` identical blocks, each measuring every check once in the code's declared
order. A program that lowers to anything else is refused, because a placement
derived from it would attribute a mechanism to the wrong round.

Two things keep the join from being a rewrite of the model in a second place. The
model derives each mechanism's signature from the source; the sampler derives
each mechanism's instruction position from the lowered program; and the tests pin
the two to each other rather than letting one call the other. At probability one
the agreement is exact — every mechanism fires on every shot, so the shot is
fully determined and the comparison is arithmetic rather than statistical — and
at the rate level the two agree within four sigma over 400,000 shots on both the
repetition and the rotated surface code. The join then decodes: a
`MinimumWeightMatchingDecoder` built from the same circuit's model decodes the
sampled events, and the residual logical-failure rate falls below the raw
observable-flip rate and below its own value at the next distance up.

What the row did not close is the noise grammar. The record names two
mechanisms, so there are two placement classes; a depolarizing or damping channel
placed after a named gate has no location here and is declined. The baseline's
`x_` and `z_` variants have no counterpart either, because the code record is a
Z-memory record. The row stays `partial` on that scope, not on the connection.
One placement is worth recording because it is invisible to any parity
comparison: the code gadgets prepare their ancilla with the CNOTs immediately
preceding the readout, so moving a measurement channel one instruction earlier
still reads the same parity. That position is pinned by the instruction index
rather than by a rate, which is where a caller relies on the position.

**What `qec_dem_merge` closed.** This is the one row with no half left to name,
and it is worth being precise about why, because the row was `absent` while the
merge itself was happening all along. Construction has always folded mechanisms
that flip the same detectors and observables, with the product-of-`1 - 2p` rule.
What was absent was the *surface*: a caller could not ask for a merge, could not
choose the rule, and could not ask whether a model needed one. Upstream states
all three — `dem_merge_duplicate_columns(dem, mode)` with `or_combine` and
`sum_combine`, plus `are_dem_columns_unique` and `assert_dem_columns_unique` —
and all three now exist here under names that say what they compute:
`DemMergeRule.INDEPENDENT_PARITY` and `DemMergeRule.CLAMPED_LINEAR_SUM`,
`DetectorErrorModel.merge_duplicate_mechanisms(rule=...)`,
`mechanisms_are_unique()` and `require_unique_mechanisms()`.

The two rules are the upstream formulas and not approximations of them.
`or_combine` is `1/2 (1 - prod(1 - 2p_i))`, which for two mechanisms is
`p1 + p2 - 2 p1 p2`; `sum_combine` is `min(1, sum(p_i))`; a group of one is
passed through bit for bit rather than run through either. The default is the
parity rule, as it is upstream. Naming the modes for what they compute rather
than for the `GF(2)` column operation they are implemented with is the one
deliberate divergence, and it is what makes the row's `renamed` verdict worth
stating rather than glossing: `or_combine` names an implementation, and the
quantity a caller is choosing between is the probability that an odd number of
the mechanisms fire.

The default rule is exact, and that is the part which is checked rather than
asserted. A detector's rate is built from a product of `1 - 2p` factors over the
mechanisms that touch it, and a merged group's combined prior is the single `p`
whose factor *is* that group's product, so regrouping the factors cannot change
the product and every detector rate and every observable rate is preserved
exactly. The test asserts that as an equality on a model with a three-way
duplicate, and denies it of the sum rule in the same file, so the two rules
cannot drift into one another without a failure.

The row also closed the reason merging matters at all, which is a correctness
argument rather than a tidiness one. The decoding graph keeps one edge per
mechanism, so a signature stated twice becomes two parallel edges with two
weights and a matcher charges the cheaper of them — for mechanisms of `0.1` and
`0.2` that is `log 4`, or `1.386`, where the fault's own combined parity `0.26`
has weight `log(0.74 / 0.26)`, or `1.046`. The charge is larger than the fault's
weight, so a matcher would prefer a longer chain of other mechanisms over the
mechanism that actually fired.
`MinimumWeightMatchingDecoder.from_detector_error_model` therefore calls
`require_unique_mechanisms()` before it builds the graph, and a model that states
one signature twice is refused with a message naming both mechanisms and the
merge that resolves them. Upstream's `assert_dem_columns_unique` is never called
in the upstream production code; here the assert is the load-bearing one.

What the row did not close, and what it therefore does not claim: when this row
was written the model had no error identifiers, so there was nothing to carry
across a merge and nothing for a merged group to be an alternative *to*.
Upstream's merge operates on columns of a carrier that records them. That absence
belongs to `dem_error_ids`, and the row it belongs to has since landed; see
section 3.6, which also records that the identifiers made the merge's premise
explicit rather than leaving it unstated.

The construction row then closed its second input end. A detector error model can
now be read off matrices as well as forced through a circuit:
`DetectorErrorModel.from_code_matrices(matrices, noise=..., num_rounds=...)`
takes the parity-check matrices that say which checks a fault on a data qubit
toggles, the logical matrices that say which logical operators it toggles and the
noise record, and derives every mechanism combinatorially. The four blocks arrive
as one record, `flagquantum.qec.css_code_matrices`, which lifts a code record into
`CssCodeMatrices(hz, hx, lz, lx)`. That is upstream's `dem_from_css_matrices`
geometry rather than the memory circuit's, and the difference is a difference
between two experiments rather than between two implementations: the matrix route
has no terminal data readout, so its detector count is `num_rounds * num_checks`
where a memory circuit's is one band per round plus a terminal detector per Z-type
check. Both geometries are pinned separately and neither is allowed to stand for
the other — `tests/qec/test_dem_code_matrices_stim.py` hands stim a circuit that
states the model's assumption and requires stim's own error analysis to reproduce
the shape and every mechanism's signature and rate;
`tests/qec/test_dem_stim_reference_rates.py` does the same for the memory
circuit's gate-for-gate transcription.

Reading the four blocks rather than a Z-type pair is what closed the fault
family. An X fault flips Z-type checks, a Z fault flips X-type checks, a Y fault
flips both, and `PhenomenologicalNoise` now states the three data rates and the
measurement rate independently, so the X detector band and the `lx` observable
rows exist and the three families are separable in the model — a rate read from
the wrong family shows up as a rate difference rather than being checked against
itself. Two things a single run of a memory experiment cannot do are handled
explicitly rather than assumed away. `lz` and `lx` anti-commute, so no one state
has both as a deterministic value: the reference side runs the experiment twice,
once per readout basis, against the model restricted to that basis, and the test
asserts that each run's observable rows are the ones the Pauli character of that
fault family predicts. And an X-type ancilla measured on a register that has never
been measured in that basis has a coin-toss outcome, so the reference circuit
extracts every check once before round zero, noiselessly and without declaring a
detector, to supply the prior the model's round-zero band is compared against.

The one place the model and a physical run genuinely part company is now a tested
fact rather than an unexamined gap. The model gives a data fault in round `r` the
detector band of round `r` and the band of round `r + 1`, which is what upstream's
construction does. A fault that is still in the data at the end of the run is
still seen by the logical readout, so it flips the syndrome of every extraction
from `r` onward, and consecutive differences cancel everywhere except in band `r`:
a physical fault spans one band, and the model's extra band is not observable.
The two agree exactly where the extra band has nothing to claim — in the final
round, which has no band after it, and in every round of a one-round run — and
`tests/qec/test_dem_code_matrices_stim.py` pins the divergence as the exact
relation between the two mechanism sets instead of as a tolerance, so it cannot
widen unnoticed.

What stays open is the rate record, not the fault family. `hx` and `lx` are read,
but upstream's `CssNoise` also carries `px`/`py`/`pz`/`pm` per qubit or per check
and here the record states uniform scalars, so a per-location profile is still
not expressible. That limit is carried by `dem_code_capacity_noise`, which stays
`reshaped`. The record set is the other thing the code-row target implies, and
it has grown by two: the triangular colour code is declared as
`flagquantum.qec.triangular_colour_code` and the square-lattice toric code as
`flagquantum.qec.toric_code`, and both derive their lattices from an argument
instead of transcribing a table, so each family is stated as a rule about its
lattice rather than copied out of another framework's source. What the record
set still does not have is a family beyond the five, which is the absence
`symbol:flagquantum.qec.qldpc_code` and
`symbol:flagquantum.qec.bivariate_bicycle_code` state. The colour patch is worth
naming for one further reason: its two check families are literally one matrix,
because every face carries one X-type and one Z-type stabilizer, so it is the
second self-dual record here and the first that exercises the both-basis readout
on a lattice nobody wrote down.

The torus is worth naming for two reasons of its own, and both are recorded as
measurements rather than as properties of the family. It is the first record here
whose matrices leave **two** logical qubits, so the change is in the shape of the
layout rather than the contents of a row: `css_code_matrices` reports four
logical operators and a default Z-basis memory experiment declares two
observables instead of one. The count is not read off a stated `k` — it is the
rank of the record's own check rows, taken by an independent GF(2) reduction, and
the two logical qubits are pinned further by requiring the anticommutation matrix
between the declared families to be a permutation matrix, so a code with one
logical qubit whose operator was written down twice would fail rather than pass.
It is also the first record whose **graphlikeness depends on the noise
declaration** rather than on the code. A Z-type check carries a detector at every
round boundary and an X-type check only in the interior rounds, so a
single-round model is graphlike under any noise; add a Y data fault and run at
least two rounds and that fault's X and Z halves land on different round
boundaries, one mechanism reaches weight four, and the matcher refuses a model it
accepted at one round. `tests/qec/test_toric_memory_execution.py` measures both
sides at the same code, round count and fault rate, so the row records a
comparison rather than a refusal on its own.

**What `qec_logical_operations` closed, and what it did not.** The row was
`absent` on all three of its named operations and on an absent module path. Of
the three, exactly one can be carried by a code declaration rather than by a
gadget that compiles a patch: product rotation is a statement about a declared
code and the Pauli operators that code declares, and it needs no patch identity
that outlives one experiment. Lattice surgery merges and splits patches across
rounds and distillation consumes several noisy magic states to produce one, so
both need that identity, and neither is here. The row is therefore `partial`
rather than `aligned`, and the third of the semantic core that landed is
`flagquantum/qec/logical.py`, which builds no circuit, executes nothing and
holds no state.

Certification is the half that makes a product a logical operator rather than a
plausible one. `certify_logical_product(code, product)` refuses a non-Pauli, the
identity, a product that mixes X-type and Z-type factors on the same record, a
product naming a wire the code does not declare as data, a product that
anticommutes with any check — naming the check's index, because that check is
the one whose outcome the product would randomize — and a product that lies in
the stabilizer span of the observables of its own type, because such a product's
outcome is fixed and the terminal readout would report a constant. The last
refusal is not a technicality: the declared all-Z operator of an even-distance
repetition patch is the product of its own checks, `Z0*Z1` times `Z2*Z3` for
distance four, so a distance-two or distance-four repetition patch genuinely has
no X-basis readout and the derivation says so instead of returning the declared
operator back.

The partner of a declared observable is derived rather than chosen, and the
derivation is a Gaussian elimination over GF(2) rather than a table of known
codes. `derive_anticommuting_logical_product(code, index)` collects the code's
Z-type checks and its declared Z-type observables as rows, solves `A t = b` with
`b` the target observable's own row, and certifies the resulting product. It
refuses an index that is not a non-boolean integer, an index the code has no
Z-type observable at, and a code for which the system has no solution. The
measured weight-`d` partners are `X0*X1*X2` for the distance-three repetition
patch, `X0*X1*X2*X3*X4` for distance five, `X1*X3` and `X2*X5*X8` for the
distance-two and distance-three rotated patches, and `X2*X4*X5` for the Steane
code, whose own record already declares an X-type observable that the derived
partner recovers up to an X-type check.

Readout rotation is where the row becomes measurable rather than declarative.
`build_memory_circuit(code, rounds, product=P)` inserts `qp.H(wires=w)` for every
wire in the product's X factor directly after the round-loop header and again
before the terminal readout, so one code record covers both bases and no second
circuit shape is added. The retention rule follows from what each check's
boundary outcome is worth in the rotated frame: an X-type check keeps its
round-zero and terminal detectors only when its support lies inside the rotated
wires, a Z-type check keeps them only when its support is disjoint from them, and
every check keeps every round-to-round detector either way. Measured at two
rounds, the Steane code keeps six detectors of twelve, the distance-three rotated
patch keeps twelve of sixteen, and the distance-three repetition patch keeps two
of six; the rotated surface patch's four boundary survivors are the two ancillas
each basis keeps. An experiment that is configured with too few rounds to leave
any deterministic detector at all — a one-round rotated readout is the case —
is refused with the round count named as the remedy rather than emitted empty,
and the default frame's source is byte-identical to what it was before the
rotation path existed for all five patches that carry a certified product.

One defect on the construction side was exposed by making the rotation
measurable, and it is worth recording because it was a disagreement between two
routes about which faults exist rather than a wrong answer on either. The matrix
route had always enumerated three data fault families — `data_flip` as an X
fault, `phase_flip` as a Z fault and `both_flip` as a Y fault — while the program
route enumerated only the X family, on the reasoning that `data_flip` is the only
one a Z-basis readout makes visible. That reasoning holds in the default frame
and fails in a rotated one, where the observable is an X-type product and the
family that opens it is the Z family; a rotated-frame model built from a circuit
carried zero observable-live mechanisms, which is a property of the enumeration
rather than of the frame. The program route now enumerates the same three
families the matrix route always has, under the rule that a data location is the
same location whether the model is built from the program or from the matrices.
The injected fault is the one the record names and not the one the readout basis
makes visible, so a Z fault is written as the three gates whose product is `Z`,
because the capture layer admits no `Z` call, and the difference between the
three-gate product and `Z` is the global phase `i`, which a Pauli frame does not
see. Measured after the correction, a two-round rotated model carries an
observable-live mechanism for each patch: one of four mechanisms for the
distance-three repetition patch, three of twenty-four for the distance-three
rotated patch, and four of fifteen for the Steane code, with no mechanism that
flips nothing in any of the three.

What the row still does not claim is the rest of its target and the reason the
target is worth having. Lattice surgery, magic state distillation, and the
patch, gadget, protocol and program records the baseline composes them from are
all absent, so there is no logical resource estimate, no distance-preserving
operation and no distillation yield here; the two module paths the checklist
searches for, `flagquantum/qec/lattice_surgery.py` and
`flagquantum/qec/distillation.py`, do not exist. Certification is a statement
about the operator and not about a probability: a certified product is not
thereby a corrected logical qubit, and no protocol that measures or protects one
is built here. The remaining work is a patch identity that survives a round
boundary, which is the same prerequisite the chunk and seam row names for a
streamed decode, and it is not delivered by either row today. The windowed
matcher in the decoder row reads a whole history one band at a time and needs no
such identity, because it never holds two chunks of a history at once.

## 3. The 23 field rows — `dem.py` against `DEMResult` / `dem_from_kernel` (both CUDA-Q core)

The short version, because the full table is in the TOML. Across 23 field rows:
2 `equivalent`, 2 `renamed`, 1 `extra`, 13 `reshaped`, 5 `absent`.

**Equivalent (2).** `detector_error_matrix` and `observables_flips_matrix` are
the same matrices in the same orientation — rows are detectors or observables,
columns are error mechanisms — with a container difference only (`numpy uint8`
upstream, `torch int8` here).

**Renamed (2).** `dem_from_stim_text(text, use_decomp_suggestions=False)` is
`DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=False)`, with the
flag keyword-only here because the route in is a classmethod. Both delegate the
format definition to stim's own parser, and both offer the same two readings of a
`^` separator under the same flag name. The default reading states the line stim
wrote: the signature is the symmetric difference of the line's targets, so a
detector named twice cancels and the groups are not retained. The flag returns one
mechanism per component at the line's probability instead, which is the graphlike
decomposition a matching decoder consumes.

Which reading is stim's is settled by stim's own sampler on the same text rather
than by preference. On a distance-five, two-round rotated surface code at 200000
shots, the default reading's worst missed observable marginal is 0.0016 against a
tolerance of 0.004, while expanding the components misses by 0.048 — twelve times
the tolerance, and sixty times the standard error at that shot count. The flag is
therefore offered as a documented departure from the line rather than as an
alternative statement of it, and
`test_the_suggestion_reading_departs_from_stim_where_the_default_reading_does_not`
asserts the separation, so a tolerance that both readings passed would not be
evidence about either. The same separation is measured a second time, at a smaller
code and expressed in the reference's own standard error, by the migration guide a
Stim user meets first: on a distance-three, three-round rotated surface code at
200000 shots the default reading sits **2.1 standard errors** from Stim's sampler
and the expanded reading **56**. Both numbers are compared against the transcript
quoted in
[`docs/guides/STIM_USER_MIGRATION.md`](../guides/STIM_USER_MIGRATION.md) on every
run of `tests/qec/test_stim_user_migration.py`, and the runnable form of the same
comparison is [`examples/qec/stim_user_migration.py`](../../examples/qec/stim_user_migration.py).

What remains of upstream's two stated losses is the error-id one, and that is a
property of the record on both sides rather than of this reader. The rest of the
refusals differ in kind, not in spirit: upstream hands the text to stim and states
what it loses, this repository refuses constructs its own record cannot hold
(`repeat`, comments, skipped indices, malformed lines) rather than dropping them —
and one construct more under the expanded reading alone, a component that cancels
to nothing, which no mechanism can state.

The second is `dem_merge_duplicate_columns(dem, mode)`, a free function over a
dem with a mode enum, which is `DetectorErrorModel.merge_duplicate_mechanisms`
here — a method, because the thing a shared signature is a property of is the
model's own mechanism record rather than a pair of columns. The two rules are
the same two rules and the formulas are the same formulas; what changed is the
noun (`or_combine` names the `GF(2)` column OR the rule is implemented with,
while the value it produces is the probability that an odd number of the
mechanisms fire) and the placement of the uniqueness predicate and the assert,
which upstream offers and never calls and which the local matcher calls before
it weights a model.

**Extra (1).** `to_stim_text`. No writer was found in the cudaq-qec bindings or
libraries; the produced stim text in the CUDA-Q stack comes from
`cudaq.DEMResult.dem` in core and from the Logical preview CLI. This is a
negative result on the upstream side, marked `provenance_unverified`, so it is
"nothing to align to at this layer" rather than "ahead".

Because there is no upstream writer to align to, the writer is aligned to stim's
own, and the two differ in precision: `to_stim_text` writes a probability with
`repr`, the shortest decimal that reads back as the identical double, while
stim's `str` writes at
`std::setprecision(std::numeric_limits<long double>::digits10 + 1)`, so the width
is a property of the platform's `long double` rather than a constant of stim —
nineteen significant digits where that type is the x86 80-bit extended one, so
the Linux CI runners, and sixteen where it is a double, so arm64 macOS.
Seventeen digits name a double uniquely, so at nineteen digits stim's printer
returns the identical double and at sixteen it cannot always: over a pinned sweep
of twenty thousand probabilities the first width lost none and the second changed
about a quarter, and the drift at that width is bounded by one part in `10**15`
and measured at `5.4e-16` worst case. The reader therefore states the printed
value on both paths, because the printed value is what the interchange carried,
and `tests/qec/test_dem_stim_text_precision.py` reads the width off stim's own
output and pins the format, the direction of the loss and the bound against it, so
that a stim release changing its precision fails there rather than invalidating
this record.

**Reshaped (13).** The model carrier, the per-error rates, the error ids, the
counts, the sampling function, the memory-circuit entry point, the matrix-level
entry point, the noise record, the measurement-to-detector map, the kernel
annotation surface, the decoder registry, the decoder result record and the
decoder base protocol. The widest of these:

- **Error ids.** Upstream states the correlation in a vector parallel to the
  rates, and documents nothing about the distribution a group of alternatives
  implies. Here the id is a field of the mechanism's own record, the vector is
  projected back out by `DetectorErrorModel.error_ids`, the group members are
  read as disjoint pieces of one shot so a detector's rate over a group is the
  sum of the members touching it, and a group whose probabilities sum above one
  is refused rather than renormalized. The statement is also the one thing the
  parity matrices cannot carry, so the three operations that would silently drop
  it — stim text, the merge and the decoding graph — name the ids and refuse.

- **Noise record.** Upstream `CssNoise` carries independent X, Y and Z data
  rates plus a measurement rate, each also expressible per qubit or per check.
  `PhenomenologicalNoise` carries the same four scalars and the same four
  vectors — `data_flip_per_qubit`, `phase_flip_per_qubit`, `both_flip_per_qubit`
  and `measurement_flip_per_check` against upstream's `px_per_qubit`,
  `pz_per_qubit`, `py_per_qubit` and `pm_per_check` — with the same override
  rule, and a location whose effective rate is zero enumerates no mechanism on
  either construction route and costs no channel in a sampler program. The
  override is wholesale rather than element by element, so a vector that does not
  name every data qubit or every check is refused naming both counts instead of
  being partially applied. The two per-check orders are related by one named
  translation: the vector is indexed the way the matrices are, Z-type checks
  first, while a code may declare its checks in any order — the rotated surface
  code interleaves the two types — and all three readers share the translation
  rather than each re-stating the convention. What remains a difference is the
  carrier rather than the rates: the record is a standalone frozen dataclass read
  beside a code, where upstream folds it into the argument the matrix entry point
  takes, and the per-check vector is validated against a code's check count
  rather than against a noise object that knows it.
- **Matrix-level entry point.** Upstream
  `dem_from_css_matrices(code: CssCodes | cudaq_qec.Code, noise: CssNoise,
  num_rounds=1)` takes one record carrying all four parity matrices and returns
  a model whose detectors are one band per round, with
  `extended_dem_from_css_matrices` alongside it returning a sparse `H` and `O`.
  Here the same geometry is reached through
  `DetectorErrorModel.from_code_matrices`, whose argument is the four-block
  record `CssCodeMatrices` that `flagquantum.qec.css_code_matrices` builds from a
  code record. Both sides therefore carry the X and Z detector bands and the
  `lz` and `lx` observable rows, and a logical observable that is neither pure X
  nor pure Z is refused here with its index named rather than half-read as one of
  the two. What differs is the accompanying record: upstream pairs the matrices
  with a `CssNoise`, here the matrices and the rates arrive as two arguments
  rather than one. The rates themselves line up: the four families and their
  per-qubit and per-check overrides are read against these matrices, so a
  code-capacity model can carry a per-location profile and not only a uniform
  one. There is no extended-record sibling, which is the absence the field row's
  negative search names.
- **Measurement-to-detector map, and the context that carries it.** Upstream
  reaches this relation two ways: `decoder_init.measurement_to_detectors()`
  returns the dense `(detectors, raw measurements)` matrix `D`, and the
  free-function helpers `d_sparse`, `dem_chunks_to_d_sparse`,
  `dem_chunks_to_o_sparse` and `dem_chunks_to_pcm` produce the `-1`-terminated
  sparse forms a realtime decoder configuration takes. Here one record,
  `MeasurementMap`, stores the row structure itself — one row per detector or
  observable, holding the measurements whose parity it is — and projects to both
  forms: `dense()` gives upstream's orientation and `flattened()` the sparse one,
  with the terminator being what keeps a row that reads nothing visible to a
  sparse reader. Its projections are closed rather than conventional: a row is
  sorted because a parity is a set, it may not name one measurement twice
  because two reads of one measurement cancel rather than add, and an index
  outside the buffer the map states is refused. What still differs is the
  context around it. Upstream's `decoder_context` is a lazy handle over raw
  circuit analysis, with `num_measurements()`, the `x_component()` /
  `z_component()` / `full_component()` methods and a boundary-aware
  canonicalization variant that exists because its `D` is laid out in uniform
  per-round blocks; here `decoder_context_from_memory_circuit` builds the model
  once and each component is a projection of it, with `DecoderInputs` pairing
  the model with both maps. That the layout states its boundary detectors itself
  is what makes the union need no separate bookkeeping, and what makes the
  terminal detectors a member of the Z component rather than a third category.
  Two refusals keep the projection from being a rewrite: a component of a model
  that states error ids is refused, because projecting a group of alternatives
  would either drop the correlation or merge two of its members, and a component
  whose basis has no detector is refused rather than returned empty. The
  chunk-scoped helpers remain absent, which is the seam absence
  `dem_seam_and_chunk_api` records. The one property the whole arrangement rests
  on is pinned by measurement rather than by construction: the numbering this
  module derives from the circuit's *declaration* is asserted equal to the
  numbering the sampler derives from the *lowered program*, so a decoder fed a
  sampled syndrome is matching its detectors against the measurements that
  actually compose them.
- **Sampling function.** Upstream
  `dem_sampling(check_matrix, num_shots, error_probabilities, seed=None, backend="auto")`
  is a free function over a matrix plus a rate vector returning sampled check
  syndromes *and* the sampled error mechanisms. Here it is a method on the model
  returning detectors and observables. The upstream error output is absent here;
  the local observable output is extra there. Upstream also rejects PyTorch CPU
  tensors, which is a full-precision-vs-`uint8` boundary worth knowing before an
  adapter is written.

**Absent (5).** `canonicalize_for_rounds`, the chunk/seam/stitch/close family,
`dem_from_kernel` itself, the sampling backend selector, and the plugin boundary
upstream keeps for open decoders.

Of the absent set, `canonicalize_for_rounds` is the one that sits closest to work
already planned: a matcher wants round structure. The correlated-channel half of
that pair is closed -- `error_ids` landed on the model's own record -- and what
the identifiers left behind is the operation that folds a group under its
exclusivity, which is what `dem_canonicalize` records.

## 4. Findings from the reading, and where the matrix now records them

These came out of reading the upstream source and are *not* criticisms of this
repository — they are places where the recorded baseline and the live upstream
have drifted apart. All five are landed in
[`contracts/cudaq-parity-matrix.toml`](../../contracts/cudaq-parity-matrix.toml);
each entry below names the construct that carries it and what the tool now
refuses.

1. **`dem_from_kernel` and `DEMResult` are CUDA-Q core, not cudaq-qec.** The
   domain's surface item "dem, dem_from_kernel, and DEMResult" mixes two owners.
   cudaq-qec's own construction entry points are `dem_from_stim_text`,
   `dem_from_memory_circuit` with `x_`/`z_` variants, and
   `dem_from_css_matrices`; `dem_from_kernel` and `DEMResult` live in CUDA-Q
   core and are consumed by the library. **Landed as**
   `[[baseline.surface_ownership.items]]` in the contract, owned by `cudaq-core`,
   whose consequence states that the item must be read as two and that the core
   record is a dependency of the QEC row rather than its baseline.

2. **`extract_syndrome`'s owner is the CUDA-Q Logical preview, not cudaq-qec.**
   `cudaq.logical.extract_syndrome` exists and that layer states of itself that
   it does not simulate, sample or decode and emits no detector error models.
   CUDA-Q QEC's own extraction route is `sample_memory_circuit` /
   `x_sample_memory_circuit` / `z_sample_memory_circuit` and
   `decoder_context_from_memory_circuit`. **Landed as** a second
   `surface_ownership` entry owned by `cudaq-logical-preview`, whose consequence
   records that aligning to that layer would not deliver a detector error model
   and that the layer is therefore a second target with a different deliverable,
   not a second baseline for this domain. **Also landed**, later and in the other
   direction: the checklist row's `symbols_absent` list named
   `sample_memory_circuit` as a name this repository did not have, and the join
   implemented it under exactly that name, so the row turned red and was
   rewritten rather than exempted. That is the decision the row was opened to
   force: this repository is on cudaq-qec's route, and the extraction entry point
   is named for what it does because that is what cudaq-qec calls it. The other
   half of that route landed the same way and for the same reason:
   `decoder_context_from_memory_circuit` was named in the same `symbols_absent`
   list, so when the decoder context was implemented under that name the row
   turned red a second time and the name moved to `symbols_present` instead of
   being exempted. Both moves are the row working as intended: it fails closed on
   the names it says are missing.

3. **No `for-stim-users` page exists for cudaq-qec.** The only such page is
   *CUDA-Q Logical for Stim users*, in the preview layer, with a companion *Stim
   emission* page. Both explicitly state that the layer does no detector
   annotations, no detector error models and no sampling. **Landed as** a
   `surface_ownership` entry for the surface item, plus a rewrite of the
   `qec_stim_user_migration` row: its `reason` now names cudaq-qec's own Stim
   surface (`dem_from_stim_text`, `dem_sampling`, the decoder framework's Stim
   DEM entry point) as the counterpart, and the row carries a `search:` evidence
   item recording that no such page and no Stim-text writer exist in cudaq-qec.
   The Stim surface item is attributed too, because both components appear in
   it and only the cudaq-qec half decodes. **Also landed, later and in the other
   direction:** the row's own `negative_search` named
   `docs/guides/STIM_USER_MIGRATION.md`, so the row was red while the guide was
   absent and turned red in the same direction when it was written — the path
   moved from `negative_search` to the row's evidence, and the row's status moved
   with it because a `partial` checkbox row is refused outright when it names no
   `symbols_present` entry. The guide is
   [`docs/guides/STIM_USER_MIGRATION.md`](../guides/STIM_USER_MIGRATION.md), and
   what makes it a row rather than a page is that
   `tests/qec/test_stim_user_migration.py` executes every fence of it in one
   namespace and compares each `print` against the transcript quoted under it, so
   a drifting number fails the suite instead of the reader. The row stays
   `partial` for the reason the guide states in the place a reader meets it: the
   Stim integration the route lands on is itself `partial`, so this row cannot
   be `aligned` while the thing it documents is not. What that costs is measured
   rather than asserted — the default reading of a decomposed mechanism gives an
   observable marginal 2.1 standard errors from stim's own sampler at 200000
   shots and every registered decoder that reads a pair-graph refuses the model
   because it is not graphlike, while the graphlike reading is accepted and
   misses by 56.

4. **`qec_stim_integration` named one maturity entry for two capabilities.** The
   row pointed at `stabilizer_sampling`, which is the sampling half, while the
   interchange half is `detector_error_model`. **Landed as** `maturity_refs`, the
   plural form: the row now names both entries, `tools/parity_matrix.py`
   validates every entry, and the checker reads both forms. The alternatives
   were both worse. Splitting the row would leave the join — which is the thing
   the row is actually about — in neither half, and it would move the capability
   count. A stated override in the checklist would leave the matrix row still
   claiming to be one capability with one entry. The plural form was chosen
   because the row genuinely spans two registry entries, and the contract's
   `maturity_refs_rule` states that a row uses one form or the other and never
   both. The override reason this checklist previously carried on
   `qec_dem_text_interchange` is gone, because it recorded the drift rather than
   the fix. The row now names **three** entries, because the join between the two
   halves became a capability of its own: `qec_memory_circuit_sampling`. That is
   the same argument one step further on, and it is why the plural form was the
   right repair — the row had two halves and no connection, then two halves and
   one connection, and a two-entry plural would have had to be reopened to say
   so.

5. **The version pin does not reach the upstream source.** The matrix pins CUDA-Q
   0.15.1 and 0.16.0.post1; cudaq-qec's release line runs 0.1.0 to 0.8.0 and
   `0.8.0` is what the QEC rows were read at. No cudaq-qec release maps to the
   pinned versions. **Landed as** `[baseline.version_provenance]` in the
   contract: the pinned component, the QEC release line, the version recorded,
   the release index, and a `provenance_limit` stating that the two lines are
   disjoint so a QEC row is not a statement about the pinned core versions. The
   tool fails closed in both directions — a pin that meets the component line
   requires a `version_mapping`, and a mapping across disjoint lines is refused.
   The generated scoreboard and
   [`CUDAQ_PARITY_BASELINE.md`](CUDAQ_PARITY_BASELINE.md)
   both carry it, and the baseline document lists the component release line
   moving as a refresh trigger.

The surface attributions are machine-checked: an attributed item must appear
verbatim in exactly one domain surface list, must name a declared owner, and
must state its consequence, so the table cannot drift from the inventory it
explains and an item owned by another component cannot be read as a cudaq-qec
row.

## 5. How the checker fails closed

The point of `tools/check_qec_cudaq_alignment.py` is that the checklist cannot quietly
become fiction. Resolution is a source-level AST scan, not an import, so it runs
without torch, without an installed FlagQuantum, and without the optional `stim`
distribution. Per row it checks:

- every `evidence` path exists in the repository;
- every `symbols_present` resolves to a real definition;
- every `symbols_absent` resolves to nothing — so a documented gap dies the day
  it is filled, and a claimed alignment dies the day the symbol is deleted;
- every `negative_search` path is still missing;
- `maturity_ref` exists in `capability-maturity.toml` **and is one the parity
  matrix row names** — through `maturity_ref` or through the matrix row's
  `maturity_refs` — unless the row states why it cannot;
- `domain_row` names a real capability of the `quantum_error_correction` domain,
  the checklist status does not contradict the matrix status, the floor does not
  contradict the matrix priority without a stated reason, and every
  `domain_items` string is a CUDA-Q surface item the matrix actually records;
- `partial` and `absent` rows state a `next_action`; `absent` and `out_of_scope`
  rows prove the gap; every row states a `fail_closed`; a row marked
  `provenance_unverified` says `UNVERIFIED` in its own note;
- the header's component version and release index are the ones
  `contracts/cudaq-parity-matrix.toml` records, and a contract that records no
  provenance fails the header rather than passing it, so the `provenance_limit`
  this document states is the contract's limit and not a second copy of it;
- every row id appears in this document, so a row cannot exist only in the TOML.

`tests/unit/test_qec_cudaq_alignment_check.py` also applies its mutations
structurally — by row id and key name — rather than by quoting long literals, so
editing the prose of a row cannot quietly turn a mutation into a no-op.

The AST index was cross-checked against a live import, because a source scan
that disagrees with the package it scans is worse than no scan. Importing
`flagquantum.qec` directly and exercising the rows' own claims —
`from_stim_text` on a two-line model, both parity matrices, `to_stim_text` round
tripping byte-identically, `dem_sampling(shots=5)` returning a `DemSample` of
shape `(5, 1)` on both axes — agrees with what the checklist says each symbol is.

`tests/unit/test_qec_cudaq_alignment_check.py` is what makes that credible: it
damages one row at a time and requires the checker to fail *for the stated
reason*, with three controls that catch a checker which reads the checklist
instead of the tree — a symbol planted in a throwaway copy of the repository
must be seen, a name the tree lacks must be rejected, and the planted claim must
pass again once the definition is removed.

## 6. Where these files live

Four artifacts, all four of them in the repository, each doing one job:

| Artifact | Home | Role |
| --- | --- | --- |
| Checklist | [`contracts/qec-cudaq-alignment-checklist.toml`](../../contracts/qec-cudaq-alignment-checklist.toml) | The rows, as data. A repository contract like its neighbours in `contracts/`. |
| Checker | [`tools/check_qec_cudaq_alignment.py`](../../tools/check_qec_cudaq_alignment.py) | Resolves every claim against this checkout. Run by CI's `quality` job. |
| This reading | `docs/development/QEC_CUDAQ_ALIGNMENT.md` | The prose a person reads once, including the field-row index below. |
| Mutation test | [`tests/unit/test_qec_cudaq_alignment_check.py`](../../tests/unit/test_qec_cudaq_alignment_check.py) | Proves the checker can fail, by damaging one section at a time. |

Defaults inside the checker are absolute, derived from its own location, so
`python tools/check_qec_cudaq_alignment.py` gives the same answer from any
working directory and the CI step does not have to name the repository it is
standing in. `--repo-root`, `--checklist`, and `--reading` remain overridable for
the mutation test, which runs the checker against a throwaway copy of the
repository rather than the real one.

The section 4 findings are contract changes rather than separate artifacts. They
landed in `contracts/cudaq-parity-matrix.toml`, with the regenerated
`docs/reference/CUDAQ_PARITY_MATRIX.md`, `contracts/README.md`,
`docs/development/CUDAQ_PARITY_BASELINE.md`, `tools/parity_matrix.py`, and
`tests/unit/test_cudaq_parity_matrix_contract.py` updated in the same change.

The split between the two documents is deliberate and is the thing to keep
straight when editing either. `cudaq-parity-matrix.toml` answers *which
capability does CUDA-Q have, at what priority, with what evidence*, across the
whole product and all twelve domains. This checklist answers *which upstream
surface does one QEC row line up with, symbol by symbol, and what exactly is
missing*, for one domain. The matrix owns the capability inventory and the
maturity join; this checklist owns the symbol-level diff and the upstream
attribution. Nothing in it should be promoted back into the matrix, and a row
here that disagrees with its matrix row fails the checker.

### Field-row index

Every row in the checklist, keyed by the id the TOML uses. `upstream` is the
CUDA-Q side; the last column is the difference in one line.

| Field row | Verdict | Difference |
| --- | --- | --- |
| `dem_carrier` | reshaped | Upstream is an opaque object with operations and recorded DEM-text provenance; here it is a frozen, validated, immutable record. |
| `dem_detector_matrix` | equivalent | Same orientation, same entries; `numpy uint8` against `torch int8`. |
| `dem_observable_matrix` | equivalent | Same, and upstream also documents the consumer idiom (`O @ errors % 2`). |
| `dem_error_rates` | reshaped | Per-column vector upstream, per-error record here; no vector accessor. |
| `dem_error_ids` | reshaped | The correlation is stated, in the mechanism's own record: `DemError.error_id` groups mechanisms that are alternatives, `DetectorErrorModel.error_ids` projects upstream's parallel vector back out, `None` where upstream has `nullopt`, and the marginal rates and the sampler read a group as one fault. Upstream fixes neither the distribution a group implies nor what happens when a group's probabilities overflow a shot; here the members are disjoint pieces of one shot and an overflow is refused rather than renormalized. Stim text cannot carry the statement, so `to_stim_text` refuses an id-carrying model and names the ids. |
| `dem_counts` | reshaped | Methods upstream against properties here; `num_error_mechanisms` against `num_errors`. |
| `dem_sampling_function` | reshaped | Free function over matrix + rates returning syndromes *and* mechanisms, against a model method returning detectors + observables. |
| `dem_sampling_backend` | absent | No execution-target selector; upstream has `auto`/`cpu`/`gpu`. |
| `dem_from_stim_text` | renamed | Same operation, same parser authority (stim), same `use_decomp_suggestions` flag with the same default, free function against classmethod; the default reading is the one stim's own sampler means, and the expanded one is the approximation upstream documents it as. |
| `dem_to_stim_text` | extra | No writer found upstream at this layer; the produced text comes from CUDA-Q core. |
| `dem_from_css_matrices` | reshaped | Same code-capacity geometry, reached from two keyword matrices instead of one four-matrix record; `hx`/`lx` and the extended-record sibling have no counterpart. The rates arrive as a separate `PhenomenologicalNoise` rather than folded into one `CssNoise`, and the vectors are read against these matrices. |
| `dem_from_memory_circuit` | reshaped | Upstream takes code + operation + rounds + noise model and is split by basis; here the circuit carries rounds and basis, and the noise record states the four families with a per-qubit and per-check override each. The context matches, but upstream canonicalizes lazily against a uniform per-round D layout and here the components are projections of the model as built. |
| `dem_code_capacity_noise` | reshaped | The four families line up one for one against X/Y/Z data rates plus a measurement rate, scalars and per-qubit/per-check vectors alike, with the same wholesale override. The difference is the carrier: a standalone record read beside a code rather than a field of the matrix entry point's argument, so the vector lengths are validated against a count the reader supplies. |
| `dem_canonicalize` | absent | No round-structure operation; the matcher decodes across rounds without one, and the windowed reading of it needs none either, because its bands are cut on detector index rather than on a canonical round layout -- a streamed decode over chunks is where such an operation would be needed. |
| `dem_merge_operation` | renamed | Same two rules and the same formulas, free function with a mode enum against a model method with an enum of its own; the uniqueness assert is called here rather than merely offered. |
| `dem_seam_and_chunk_api` | absent | Monolithic model; no chunk and no seam, so the windowed decoder cuts bands out of a whole history instead of consuming chunks of one. |
| `dem_measurement_to_detector_map` | reshaped | One `MeasurementMap` record against a stored dense D matrix plus free-function sparse helpers: it projects to both forms, but the chunk-scoped helpers have no counterpart. |
| `kernel_annotation_surface` | reshaped | Layouts beside the source against annotations in the kernel body over measurement handles. |
| `kernel_dem_from_kernel` | absent | CUDA-Q core derives the DEM from the kernel's own annotations; here it is assembled by hand. |
| `decoder_registry` | reshaped | Upstream reaches a decoder by name — `get_decoder(name, H_or_dem_text_or_sparse_matrix, **options)` with a decorator putting a class behind a name; here `flagquantum.qec.get_decoder` takes a carrier and `register_decoder` puts one there, checked while the registering module is imported. Narrower on two deliberate points: the source argument is one of the three carriers a caller can hold a model in rather than a parity-check matrix, and only the detector-error-model family is registered, because the repetition-code decoders take an ordered syndrome history rather than detection events. |
| `decoder_result_record` | reshaped | Single-shot record against `converged` + optional results, with separate batch and async records. The DEM-level matcher adds `MatchingDecodeResult`, which carries observables, the selected mechanisms and their weight, and deliberately is not this record. |
| `decoder_base_methods` | reshaped | `decode` + streaming against `decode`/`decode_batch`/`decode_async`/`get_block_size`/`get_syndrome_size`/`get_version` and an errors-vs-observables request. The DEM-level decoder does not implement the repetition-only protocol, because no correction record here can express a surface-code correction. |
| `decoder_plugin_precedent` | absent | Upstream integrates open chromobius and pymatching plugins behind a boundary while its closed decoder ships as a binary — the same split the matrix plans. The integrating half of that split is now realized for one library, as the PyMatching cross-check behind an extra; what is still absent is the boundary itself, since the adapter is a concrete class rather than a registered plugin. |
