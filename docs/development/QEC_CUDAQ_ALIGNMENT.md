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
| Product surface — objects, entry points, code families, decoders | **CUDA-Q QEC** (`cudaq-qec`, the `NVIDIA/cudaq-qec` repository and its `libs/qec`) | It is the only live framework with a QEC library of this shape, and `contracts/cudaq-parity-matrix.toml` already names it as the baseline. |
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
| `qec_code_record` | aligned | now | `qec_code_library` | Closed: three families declared, each reachable by name, each reporting its X-type and Z-type ancilla bands and the two matching stabilizer counts, all three feeding the matrix route, and a record also buildable the other way, out of the parity-check and logical matrices a caller holds. The baseline's per-operation kernel map and an arbitrary non-CSS stabilizer list stay absent and named. |
| `qec_detector_annotations` | aligned | now | — | Closed: identity derived from the code, and every recorded bit addressable by a handle that reads as a boolean vector or as an integer. The kernel-annotation spelling stays absent and named. |
| `qec_syndrome_extraction_owner` | aligned | now | — | Closed: `extract_syndrome` is in the CUDA-Q Logical preview rather than in cudaq-qec, both extraction routes here carry cudaq-qec's own names, and the absence of the preview's name from the cudaq-qec tree is now read at a named revision instead of being marked unverified. |
| `qec_dem_construction` | aligned | now | — | Closed: construction is exact on both routes, the context object landed, and the baseline's `decompose_errors` argument has a counterpart of its own on the circuit route. The kernel-annotation route stays absent and named, so no X/Y fault family follows from a kernel body. |
| `qec_dem_matrices_and_rates` | aligned | now | — | Closed: both matrices in the stim orientation, the error-id column, the per-mechanism rate column, the closed-form marginals and the context object are all present. |
| `qec_dem_merge` | aligned | now | — | Closed: both stated rules, the uniqueness predicate and the refusal are present and enforced at the decoder. |
| `qec_dem_chunking` | partial | now | `qec_decoder_family` | The layer algebra, the window identity, the named seams, the two round trips and the three chunk-scoped projections all landed; what is left is the baseline's sparse spec shorthand, its per-seam tags and its straddle flags, which are reshape decisions rather than gaps. |
| `qec_dem_text_interchange` | aligned | now | `qec_stim_integration` | Closed: both directions present and independently checked, the reader takes the whole grammar stim's writer uses, `repeat` blocks and comments included, both separator readings are offered under upstream's flag, and the detector count is read from the declarations. The narrowness is at the input end and the precision width is a stated tolerance; both are declined rather than left open. |
| `qec_stim_sampling_join` | aligned | now | `qec_stim_integration` | Closed: the join landed, every family the noise record states is placed, and a second grammar places a channel bound to a named gate after the gate it matched. An arbitrary annotated circuit at the input end is declined by the type the sampler states rather than left as a gap. |
| `qec_decoder_family` | partial | now | `qec_decoder_family` | A DEM-consuming matching decoder, a belief-propagation decoder that reads the hyperedges that matcher refuses, a composite-fault decomposition that widens it past the one hyperedge a memory circuit states, its PyMatching cross-check, a name-keyed registry and a sliding-window decoder over the chunk decomposition all landed; no batch result record and no plugin boundary. |
| `qec_decoder_configuration` | absent | later | — | Nothing to configure until more than one decoder can be selected. |
| `qec_dialect` | absent | later | `qec_dialect` | Needs an internal IR level to carry the structure. |
| `qec_logical_operations` | absent | later | `qec_logical_operations` | Depends on the decoder family; align to the semantic core via Qualtran. |
| `qec_transport_and_objectives` | absent | later | `qec_transport_and_objectives` | Hardware-shaped; out of scope until a neutral-atom target exists. |
| `qec_stim_user_migration` | absent | next | `qec_stim_user_migration` | A document, and its upstream counterpart is CUDA-Q QEC's own Stim surface rather than a page to translate. |

Everything a `supported` row would need is deliberately *not* claimed here. Eight
rows are `aligned` and the rest are not, and an `aligned` row is one whose
upstream surface has no item left unaccounted for, symbol by symbol. The checker
enforces the floor of that bar rather than the bar itself: an `aligned` row must
name a symbol that resolves and may not sit on a matrix row that says
`unsupported`. That a row clears the mechanical bar is therefore not the
argument for calling it aligned; the argument is in the sections below, where
each row states what it closed and what it did not, and a row that still has a
half to name stays `partial`.

**What `qec_code_record` closed, and what it did not.** The row's target is a
declared code record that yields its distance, its data wires, its ancilla
wires, its checks, its stabilizers and its logical observables, so that a
memory circuit and its layouts are derived from the code. Three records carry
that much already — `RepetitionCode`, `RotatedSurfaceCode` and `SteaneCode` —
and what this round read is the rest of what the upstream record declares.

The gap was an accessor. `cudaq::qec::code` declares `get_num_data_qubits`,
`get_num_ancilla_qubits`, `get_num_ancilla_x_qubits`, `get_num_ancilla_z_qubits`,
`get_num_x_stabilizers` and `get_num_z_stabilizers` as pure virtuals, and the
per-basis four are not decoration: a syndrome-extraction round visits the
ancillas of one basis, and a detector band is the set of handles one basis
produces, so a record that reports only a total cannot say which ancillas a round
has to visit. `StabilizerCode` declared only the total, so a consumer typed
against the protocol could not ask. It now declares `num_ancilla_x_qubits` and
`num_ancilla_z_qubits` beside `num_ancilla_qubits`, on the protocol rather than
on the three classes, because a member the consumers read has to be on the
protocol or the next record is free to omit it and still be accepted.

The two stabilizer counts came in the same pass, and how they are answered is the
part worth recording. Upstream keeps them as two further virtuals, and every code
class it ships answers the same number to a stabilizer count as to the band count
of that same basis — Steane returns 3 and 3, the surface code `(d²-1)/2` for
each, the repetition code 0 and `distance - 1` — because there one ancilla
measures one stabilizer exactly as it does here. So the two are not a second
quantity to derive but one quantity a caller may read under either name, and
`num_x_stabilizers`/`num_z_stabilizers` return the two band counts rather than
recounting the checks. Deriving the same number twice would be two answers to one
question, which is the failure this row already refuses elsewhere: the alternative
— a record whose stabilizer count and whose ancilla count were computed separately
— is representable and wrong, and nothing would say so. A caller comparing this
record against upstream's accessor list finds the same six counts, and the count
of checks of each basis, which is what upstream counts, agrees with the number
returned.

The two bands are derived rather than declared, and that is the decision worth
recording. A check's ancilla measures exactly the basis the stabilizer's type
fixes, so the split *is* a function of the checks; a record that stated both
could state two answers, and every consumer reads one of them, so the drift would
be silent. `flagquantum.qec.ancilla_bands` reads the split off the checks once,
the three records take their two counts from it, and
`build_memory_circuit` compares the stated counts against the derived bands and
refuses a record where they disagree. That is the fail-closed convention applied
to a quantity that is present rather than missing: the two statements are
compared instead of the derived one being trusted and the stated one ignored. An
ancilla that measures neither basis — a flag, or an idle ancilla — is in neither
band, so the two bands are deliberately not a partition of the declared ancillas
and the comparison is against the bands rather than against the total.

The factory half of the upstream record is the other thing that landed. Upstream
reaches a code two ways: `get_code(name, options)` builds one by name and
`get_available_codes()` lists the names. `flagquantum.qec.get_code(name,
**options)`, `flagquantum.qec.code_names()` and
`flagquantum.qec.register_code(name, *, replace=False)` are the same three
positions, and they are deliberately the same shape as this repository's decoder
registry rather than a second idiom in the same package: registration refuses a
duplicate name unless `replace=True`, an unregistered name raises and lists the
registered ones, and the members a record must carry are read off the
`StabilizerCode` protocol rather than restated, so a class that cannot answer one
of them is refused where it enters the registry with the missing members named.
What is registered is the class and not an instance, because the options are the
caller's, and the options are the record's own fields, checked in the factory so
an unknown field is a refusal that lists the ones the record takes rather than a
traceback from inside the constructor.

What did not land is upstream's second overload, `get_code(name, stabilizers,
options)`, and it was left out rather than overlooked. That overload builds the
named record and then overwrites its stabilizer list, and the shape does not
survive here: the counts this round just read are derived from the checks, so a
record whose stabilizer list were swapped under it would answer with the split of
the class it was built from while its checks stated another, which is exactly the
disagreement `build_memory_circuit` refuses. A record built from an arbitrary
stabilizer list has to decide each check's ancilla and its coupling
direction on the caller's behalf, and a local `CodeCheck` states one ancilla and
one CNOT direction fixed by the check's type — so a mixed X-and-Z stabilizer is
refused rather than given a second ancilla, and the qLDPC, Reichardt and Floquet
families upstream serves with that overload have no route in. A caller holding a
parity-check matrix is served by a route of its own, described below, which reads the
matrix rather than inventing a gadget for it, so the omission is a gap in the declared
record and not a gap in the matrices route.

The colour code is withdrawn from this row rather than kept as its gap, and
reading upstream's headers is what settled it. There is no concrete code type
there at all — only the abstract `cudaq::qec::code`, the `css_code_matrices` and
`css_noise_params` records, an `operation` enum that includes `stabilizer_round`,
and the factory whose own list is Steane, repetition and surface. This repository
carries one code from each of those three families, so the row is not short a
family. A search for a colour code upstream returns nothing, and a construction
was attempted here before that was known: the honeycomb lattice as the dual of
the triangular one gives a CSS code at every patch size tried, and it encodes
many qubits at every one of them rather than the single one a colour code needs,
so no distance was ever obtained. The absence is therefore recorded as an absence
— `symbol:flagquantum.qec.color_code` — and not as work this row is waiting on.

What stays open inside the row is the shape of the record rather than its
declarations. Upstream's code *is* the map from an operation to the kernel that
performs it: `get_operation<T>(op)`, `contains_operation(op)` and the protected
`operation_encodings` map, over an `operation` enum that reaches from `x`, `y`,
`z` and `cx` to `stabilizer_round`, `prep0`, `prep1`, `prepp` and `prepm`; and
`get_stabilizer_schedule_x()/z()` carries a timestep-indexed interaction order,
where entry 0 means no support and entry *k* means an interaction at timestep
*k*, for hook-error-aware measurement orders. Here a record declares its checks
and one source is built from them, so a code carrying its own preparation,
measurement or logical-operation kernels, or its own schedule, would need a
second shape; `symbol:flagquantum.qec.operation_encodings` states that absence.
Note that this one is a decision about this repository's circuit model rather
than a small addition, which is why the row stays `partial` with the gap named
rather than being closed by widening the record.

**The matrices route into a code record.** The other half of this row was the
one its own `next_action` had been recording as missing: a caller holding the four
CSS blocks — `hz`, `hx`, `lz`, `lx` — could reach a detector error model but could
not reach a *code*, so no memory circuit and no detector or observable layout
could be derived from matrices at all. `flagquantum.qec.CssCode` is that route.
It takes a `CssCodeMatrices` and a distance, and it answers the same fourteen
members `StabilizerCode` declares, so every consumer of a code record — the
memory-circuit builder, the ancilla bands, `css_code_matrices` — takes a
matrix-built record on the same terms as a declared one, with no branch for it
anywhere. The Shor code is the demonstration that this is not a second way to
write down the same three families: it is nine data qubits with six Z-type and
two X-type checks and a weight-three transversal logical Z, no class here declares
it, and it reaches a memory circuit, a detector error model and sampled detection
events from four blocks alone.

The distance is stated by the caller, and that is upstream's own division rather
than a shortcut. `cudaq::qec::code` declares no distance accessor at all —
`code.h` takes one out of the options a record is built with, and
`repetition.cpp` and `surface_code.cpp` each throw when it is absent — and
`css_code_matrices` carries no distance for a matrix holder to read either. A
record built from matrices therefore has no distance to copy and nothing to read
it from, and deriving it here would be worse than copying that shape: the
distance of a quantum code is a minimum-weight-codeword problem, which is
exponential in general and would refuse exactly the large matrices this route
exists for. What the
record does instead is prove the one direction a stated logical operator can. An
operator of weight *w* bounds the distance above by *w*, because that operator is
a logical operator, so a record claiming a distance larger than the lightest
operator it states is refused with both numbers named, and a record that
understates its distance is accepted, because the bound is real in that
direction only. That is deliberately a one-sided check: it catches the mistake a
caller actually makes, copying a distance from a paper that belongs to a
different matrix, and it does not pretend to derive what it cannot.

The algebra is where this route goes past the upstream one rather than matching
it, and the difference is recorded as a difference. `dem_construction_utils.cpp`
validates a common column count, the length of each per-element rate vector, and
that each probability lies in `[0, 1]`; it never checks commutation,
orthogonality, logical non-triviality or independence, so a pair of blocks that
do not commute reaches a detector error model there. A record built here is held
to all of it: two check blocks whose product is not the identity are refused, a
logical operator meeting the opposite basis' checks on an odd number of qubits is
refused with the check named, a logical operator that is a product of its own
type's checks is refused as a stabilizer rather than a logical operator, two
dependent rows in one logical block are refused because each row becomes one
observable, and a check or logical row with no support is refused because an
empty row states the identity operator. A reader comparing this against upstream
should read the paragraph above as a strengthening, not as parity: upstream does
not have this check, so passing it here is not evidence of upstream conformance.

Two things the route deliberately does not carry are worth stating so they are
not mistaken for omissions. It is not registered by name: the identity of a
matrix-built record is the matrix, not a string, so `get_code` still reaches the
declared three and a caller does not have to invent a name to use their own
blocks. And it is CSS-shaped by construction, so an arbitrary non-CSS stabilizer
list still has no route in — the same limit the paragraph above describes, for the
same reason, and not a limit this round removed.

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
missing `decode`, `from_detector_error_model` or `from_decoding_graph` is refused
while the registering module is being imported, which is the same place upstream
refuses it and a better place than the first caller, where the name is all the
caller has to go on. The third member is required because the three carriers are
one documented surface: a name that could be built from a model and not from a
graph would be a name whose accepted sources depend on which name was asked for,
so the matcher and the cross-check both carry the constructor the graph carrier
reaches and the registry calls it rather than a bare `__init__`. And a second registration of one name is refused unless the caller says
`replace`, because two classes answering to one name is a choice the registry
cannot make on the caller's behalf.

The optional implementation is registered whether or not it is installed, so the
name `pymatching` is part of this package's surface rather than the extra's, and
asking for it without the extra raises the error that names the extra instead of
a name that silently is not there. No name is preferred over another: the
registry returns the class a name is registered against, so the authority is
returned for its own name wherever the extra happens to be present, and the
cross-check is never reached by accident.

**How the hyperedge front end landed, and why it is a construction option.** The
row named a hyperedge-decomposition front end as the thing that would widen the
matcher's graphlike scope. It landed, and it landed in
`DetectorErrorModel.from_memory_circuit` rather than in the decoder, because a
hyperedge is a statement the *model* makes and a decoder can only refuse it. The
keyword is `decompose_composite_faults`, and the composite fault it reads apart
is the one a memory circuit states: the Y family, which is an X flip and a Z flip
at one location, enumerated as those two parts at the parent's own rate, forced
through the same injector at the parent's own round and wire and read by the same
signature reader the single-Pauli families use.

What it buys is measured rather than asserted. On a distance-three rotated
surface code at two rounds with all four families at `0.01`, the combined model
states four mechanisms of three detectors and one of four, and
`MinimumWeightMatchingDecoder.from_detector_error_model` refuses it naming the
detectors of the first hyperedge. The decomposed model states none above two, and
that same matcher builds on it and returns corrections for the syndromes those
mechanisms explain. On a repetition code — whose checks are Z-type alone, so a Z
fault reaches no detector in any round — and on a one-round surface experiment,
the two readings produce the *same* model mechanism for mechanism, so the option
costs nothing where it has nothing to do.

Three things keep it from being a rewrite of the model in a second place. The
default is the combined reading, so no earlier caller's model changes and the
compatibility claim is asserted as an identity of models rather than as an
agreement of marginals. The reading is documented as a *different joint law*, not
as an equivalent statement: the parts always fire together in the composite fault
and independently when read apart, so each detector's and each observable's
marginal rate is unchanged — measured at `5.6e-17` worst case over both rate
vectors — while the mechanism sets are asserted to differ. And the matrix route
deliberately carries no such keyword, with the absence pinned by a test rather
than left to convention, because the code-capacity route's fault spans two
detector bands: a part there still reaches four detectors at two rounds on
rotated surface codes at distances three, five and seven. A keyword whose name
promised the matcher a graphlike model would be stating something that route
cannot deliver.

The rule is also not upstream's, and the row says so. Upstream states
`decompose_errors=False` on `dem_from_memory_circuit` and documents it as
hyperedge mechanisms being "decomposed into pairs of two-detector edges by Stim
before returning" — the pairing is whatever Stim chose, which is why no pairing
rule exists in upstream's own sources to read. Here the parts are derived from
the program, and the claim is bounded to what that derivation supports: a part is
graphlike exactly where the corresponding single-Pauli fault at that location is
and no further. A hyperedge from any other source is still refused by name.

The second decoder that registry holds is the one the matcher's own scope left
out, and it is `flagquantum/qec/belief_propagation.py`, registered under
`belief_propagation`. The row named "belief propagation with ordered statistics
decoding" as one of the two families the matcher does not cover, and the reason
it is a family rather than a setting is the shape of the input: a matcher needs a
graphlike model, so a mechanism touching three or more detectors has no edge to
become and is refused. A factor graph has no such limit. A mechanism is one
variable, a detector is one check, and a mechanism touching three detectors joins
three checks, so the factor graph is the model as written rather than a
projection of it — which is exactly the projection the matcher's refusal
declined to make.

What the exchange is, and what is checked about it. The prior of a variable is
the log-likelihood ratio `log((1 - p) / p)` of that mechanism's own rate and not
a marginal. A check composes the tangents of half of each *other* variable's
message, inverts the product when its own detector fired, and sends the result
back: excluding the variable's own report is what separates the check's evidence
about the variable from what the variable already said, and the syndrome bit is
in the update because a check whose detector fired states the complement of what
it states when the detector is quiet. A run returns the first iterate whose hard
decision explains the syndrome, and the empty syndrome is answered from the
priors alone before any message is sent. The correctness claim is made against
the model's own distribution and not against the exchange's opinion of itself:
every mechanism set has a probability, a detector signature and a logical label,
so the most likely explanation of a syndrome is a number computed by enumeration,
and on a factor graph that is a tree — where the beliefs are exact — the decoder
must reach it for every syndrome the model can produce. The test does that on a
five-mechanism chain against an exhaustive enumeration, and it pins the syndrome
bit separately by turning the fallback off: a decoder that dropped the bit would
be transmitting the empty syndrome on every other one, would never settle, and
would raise instead of answering.

Off a tree the exchange is approximate, and that is stated rather than left to be
discovered. On the triangle-plus-hyperedge model the beliefs settle on
explanations that flip the syndrome and are heavier than the cheapest, which is
the known behaviour of sum-product on a graph with cycles and the reason a caller
who needs the cheapest explanation on a graphlike model wants the matcher. What
the tests hold the decoder to instead is the invariant that survives: every
syndrome is answered with a set that flips it, the reported weight is the sum of
the selected mechanisms' own ratios, and the weight is never below the cheapest
explanation's — a smaller number would be a weight that is not the quantity it
claims to be. A cycle that settles on a heavier-than-cheapest explanation is
asserted to happen, so the approximation is a demonstrated fact rather than a
clause in a docstring.

When the exchange does not settle, ordered statistics answers, and the flag is
what says which of the two answered. The fallback orders the mechanisms by their
posterior belief, takes a greedily chosen independent set of their columns as the
information set, solves the reduced system over that set, and fixes every other
mechanism at its belief decision; if the residual check row is inconsistent it
refuses rather than returning a set that does not explain the syndrome. A caller
who needs the exchange's own answer rather than a solve sets the fallback off,
and then an unsettled run raises with the reason. `converged` is false exactly
when the fallback answered, so the flag is a fact about the path taken and not a
quality score.

The refusals are the region the exchange cannot carry, and they are raised while
the caller still holds the model. A model that states its mechanisms are
alternatives is refused, because the exchange weighs every variable as an
independent fault and a group of alternatives is one fault whose members cannot
fire together, so reading the group as independent would invent shots in which
two members both fired. A mechanism of rate zero is refused, because its prior
ratio is not a number and a mechanism no shot can select is not evidence. A
syndrome that no set of mechanisms can produce is refused by name, because the
mechanisms span a subspace of the detector space and a syndrome outside it did
not come from this model; the least-bad set would be a correction that does not
explain what it was asked about.

**How the sliding window landed, and the one narrowing it states.** The row's
last unbuilt decoder half was the sliding window, and it is now
`flagquantum/qec/sliding_window.py`, built on the chunk seams that
`flagquantum/qec/chunks.py` states and that `qec_dem_chunking` records. The
reshape is that a window's matrices are already cut. Upstream slices detector rows
and error columns out of one flat matrix to build a window and re-indexes both,
which is why its per-window `H_round` and its `first_columns` list are separate
facts it carries around; here the decomposition did that already, because
`dem_close_all` lays a window out at its own place in the model's numbering and a
window's own model numbers its rows from zero and holds only that window's
mechanisms. So the inner decoder of a window is built from that window's model and
from nothing else, and upstream's column-offset commit has no counterpart: a
window's correction is final when it is committed, nothing downstream can revise
it, and no residue has to be re-derived from an offset.

What an inner decoder has to report is the substantive narrowing, and it is
stated rather than worked around. A window's syndrome is its own detector rows
*less the support of the mechanisms the window before it committed*, because two
adjacent windows share the boundary round's detectors and one event may not be
explained twice. Forming that residue needs the selected mechanisms, so an inner
record must state them as `mechanisms` in that window's own column order, which
`BeliefPropagationDecodeResult` does and `MatchingDecodeResult` does not — that
record states a correction as graph edges rather than as column indices. A window
built with the authority matcher therefore refuses at the first commit by name,
naming the inner decoder, its record type and the window, instead of silently
dropping the correction; that refusal is a property of the matcher's record and
is recorded as such. A window that cannot explain its residue refuses for the
neighbouring reason and resets the stream, which is a limit rather than a defect:
a window's residue is a subsystem the window need not span, so an arbitrary
detector subset is not a syndrome any window promises to read. Measured on the
distance-three repetition code at four rounds, a two-round window abandons the
terminal mechanisms' syndromes where a three-round window answers every mechanism
of the same model, and the tests assert both by driving the model's own mechanisms
rather than random subsets.

Three decisions about the record are worth naming because they are divergences
rather than translations. `SlidingWindowDecodeResult` reports the committed faults
*while the stream is still sliding*, where upstream returns an empty vector until
the final window because its result vector is indexed by a column space in which
the shared columns are still open — no column is shared here, so an empty vector
would be stating less than the decoder knows. Its `complete` flag is what
upstream's empty-vector return encodes, and its `converged` field is the inner
decoders' flags anded, or `None` when none of them reports one, because the
matcher's record has no such flag and crediting a decoder that said nothing with
settling would be worse than saying nothing. And the decoder is deliberately *not*
a registered name: a registered decoder is built from a model or from the graph
one defines, and a sliding window needs the windows of a decomposition, which a
model does not state, so a caller constructs it from a `DemChunksSpec` and names
the registry decoder each window is built from.

The three chunk-scoped projections landed with it. `dem_chunks_to_pcm` and
`dem_chunks_to_o_sparse` index one shared fault column space in the order
`dem_close_all` states, so `dem_chunks_to_pcm(chunks)` is the closed model's
`detector_error_matrix()` read one detector row at a time and
`dem_chunks_to_o_sparse(chunks)` is its `observables_flips_matrix()`, each in
sparse form and neither built dense. `dem_chunks_to_pcm` is deliberately the
trail-row form rather than a dense matrix product: a long model's mechanism count
makes the dense product the thing to avoid, and the row and column index spaces
are what a caller reads either way. `dem_chunks_to_d_sparse` is the one projection
that meets a third numbering — the measurement bit of a flat `rounds * d` buffer,
which is what a memory experiment's raw measurements occupy — so it is round-major
and its precondition is checked rather than assumed: every round of the sequence
must be the same `d` detectors wide. That is what makes it refuse a surface code's
boundary rounds by name, since a distance-three surface code's boundary rounds are
four rows wide where its interior rounds are eight; that geometry's correspondence
comes from the circuit's measurement handles through `MeasurementMap` instead.

What the row still does not have is now two items and neither is a decoder. The
baseline's `DecoderResult` — the batch record whose `opt_results` channel is
compared as a boolean flag and whose empty batch yields a `(0, 0)` result with a
`(0,)` converged vector — has no local carrier; `BeliefPropagationDecodeResult`
and `SlidingWindowDecodeResult` are deliberately each decoder's own record and not
that one, because one answer is a convergence flag over a selected set and the
other is a stream of committed windows. And the plugin boundary is still the
baseline's precedent rather than a protocol here.

**What `qec_dem_chunking` closed, and what it did not.** The row was `absent`
with a floor of `later`, and the reason it was `later` was stated in the row
itself: nothing in this repository decoded, so a chunk seam was substrate without
a consumer. That reason has expired — a matching decoder, a PyMatching
cross-check and a name-keyed registry all landed, and the row is now the one the
decoder family's `next_action` names first. So the floor moved to `now`, the row
moved to `partial`, and the parity matrix row `qec_decoder_family` now names
`detector_error_model_chunking` beside `detector_error_matching_decoder`.

The decomposition is `flagquantum/qec/chunks.py`, and what it is had to be
decided before it could be written, because "a chunk" is not one thing. A *layer*
here is the set of detectors sharing one round index, so a layout is a tuple of
round widths in detector order whose sum is the model's detector count, and
`ChunkLayout.from_memory_circuit` reads it off the circuit rather than accepting
it. Two conventions are settled there. A terminal detector — one whose parity
references the terminal data readout and no round — is placed in the last layer
instead of being given a round of its own, which is what makes a memory circuit's
layers contiguous rather than one-short-of-contiguous. And a numbering that
reaches a layer after skipping one is refused rather than closed up, because
renumbering the rounds after a missing one would silently move every window that
follows it; this repository's `DetectorLayout` already refuses a non-dense
numbering on the circuit side, so the two agree by construction rather than by
coincidence.

A *window* spans a contiguous run of layers and its bands are fixed: the leading
boundary layer it shares with its predecessor, its own interior, and the trailing
boundary layer it shares with its successor. Its stride is the window width minus
one, which is what makes the windows share exactly one layer each and therefore
tile. Three shapes are refused with the layer count named — a window of one
layer, a window wider than the layout, and a layer count that does not tile at
that stride — and the refusal of the third is the honest answer rather than a
gap: a caller who wants to advance by less reads overlapping windows and simply
does not stitch them, whereas a decomposition whose windows did not tile would
not be a decomposition.

What makes the windows a partition rather than a cover is a fact about the model
rather than a rule imposed on it. A mechanism spans exactly two adjacent layers,
because a fault is placed at one location and flips that round's detectors and the
round before's; a mechanism lying inside a single layer does not arise in either
construction route. So owning a mechanism by the *first* window that contains it
whole owns every mechanism exactly once, and the round trip is an equality rather
than a tolerance. Measured on `RepetitionCode(distance=3)` at three rounds under
`PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)`: four layers of
width two, eight detectors, fifteen mechanisms, cut at window two into three
windows labelled `init`/`bulk`/`final` holding eight, five and two mechanisms —
which sums to fifteen — with `dem_close_all` of the three and `dem_close` of their
stitch each returning the model. On `RotatedSurfaceCode(distance=3)` at four
rounds the same noise gives five layers `(4, 8, 8, 8, 4)` — the boundary-aware
round of four and the steady-state round of eight that upstream's `[B | S…S | B]`
layout also states — thirty-two detectors and sixty mechanisms, cut at window two
into four windows of twelve, sixteen, sixteen and twelve local detectors holding
22, 15, 19 and 4 mechanisms, again summing to sixty. The same model at window
three gives two windows of twenty local detectors holding 37 and 23 mechanisms,
and the same two round trips hold. At window two the interior of the two middle
windows is empty and the interiors of the two end windows hold four detectors
each, which is the band model's own prediction: a window of `w` layers spans
`w - 2` interior layers, so the end windows of a decomposition span one layer
fewer than a middle one.

Two mechanisms are refused rather than placed, and both refusals are the reason
the decomposition can claim to be exact. A mechanism that no window contains
whole is refused and never split, because splitting it would state two weaker
faults where the model stated one; the refusal names the layers it flips and
points at the caller's two remedies, a wider window or the fault stated per
round. And a mechanism flipping no detector is refused, because an observable-only
fault has no round to be placed in and choosing one would be inventing placement;
in both construction routes such a mechanism does not arise, so this is a
statement about what a window can hold rather than about what the model does.

The operations over the windows are where the identity does its work, and one
decision was revised during the round rather than carried in. `dem_stitch`
contracts the shared layer of two adjacent windows and lays it out once, between
their interiors, so the result spans one layer fewer than the sum of its parts;
`dem_close` lays one window out at its own place in the model's own numbering
without renumbering from zero, so a window's detectors are the model's detectors.
A seam's rows carry the *global* detector index rather than a position within the
seam, and that is the whole reason a stitch compares identities instead of
widths: two boundary bands of equal width that are not the same boundary are
refused, where a positional reading could not tell them apart. `dem_stitch` and
`dem_close_all` both require the two sides to share *exactly one* detector layer —
`left.last_layer == right.first_layer`, not `left.last_layer + 1 ==
right.first_layer` — because a windowed decomposition of a shared boundary layer
is not an adjacent-block decomposition, and the first version of this check was
written for the adjacent-block reading and refused the correct case. The check
that a stitch leaves no third seam unaccounted for was written and then removed:
with the bands fixed at `prev_round | interior | next_round` and the contracted
band laid out once, a stitch's result can carry at most the outer seams of its
two sides, so the condition was unreachable and asserted nothing. A test that
cannot fail is worse than no test, and the removal is recorded here rather than
left as a hole in the coverage list.

Four differences from upstream's seam surface are decisions, and the field row
`dem_seam_and_chunk_api` records all four. Upstream hashes a seam's name into a
`uint32` at compile time and keeps a name registry only so a diagnostic can turn
the hash back into text; `SeamId` keeps the name as the identity, so nothing
hashes, nothing interns, and two seams are the same seam exactly when their names
are equal. Upstream's per-seam tags are caller-chosen labels numbered positionally
within a seam, which makes its tag check the same statement as its width check;
here the rows are detector identities, which is what makes the check a comparison
of boundaries. Upstream's spec is a phase graph with caller-named phases, edges, a
self-loop for the repeating phase and a round count supplied at expansion time;
here the spec is linear — `init`, a repetition of `bulk`, `final` — so
`DemChunksSpec.chunk_specs` *is* the expansion upstream spells `expand_dem_chunks`,
and a second name for one operation would be a second spelling of it. And
upstream's `extended_dem` is a model type holding the check matrix, the observable
matrix, the rate vector and the seams; here a window is a plain
`DetectorErrorModel` plus its own seams, which is the same decomposition without a
second model type, and `dem_close` is what turns one back into an addressed model.
The chunk-scoped matrix projection landed with the decoder that reads a window
rather than ahead of it, which is where this row said it belonged:
`dem_chunks_to_pcm` and `dem_chunks_to_o_sparse` index one shared fault column
space in the order `dem_close_all` states, so each is the closed model's own
detector or observable matrix in sparse form, and `dem_chunks_to_d_sparse` is the
memory XOR map that meets a third numbering — the measurement bit of a flat
`rounds * d` buffer — and checks its own precondition of uniform round widths
rather than assuming it, which is why it refuses a surface code's boundary rounds
by name and leaves that geometry to `MeasurementMap`. What the row still names is
therefore only the baseline's sparse spec shorthand, its per-seam tags and its
straddle flags, and those are reshape decisions recorded above rather than
pending work.

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

What the row did not close is the noise grammar. The record names four
families, and all four are now placed, but they are placed at two kinds of
location: a data fault at a round boundary and a flip at a check's readout. The
engine executes exactly one channel — a single-wire bit flip — so the Z and Y
data families are *that* channel conjugated by the Clifford that turns the flip
into the Pauli they name: `h` around it for `phase_flip`, since `H X H = Z`, and
`sdg`/`s` around it for `both_flip`, since `S_DAG X S = iY`. Conjugating one draw
is what makes a fault with one rate one fault; sampling two independent bit flips
at that rate would be a different distribution, so the two are not compared
against each other anywhere. The wrapper cancels when the channel does not fire,
which is why the noiseless circuit is untouched. The construction route states
the same identity in the language it injects into, which carries `h` and `x` and
no other parameter-free single-qubit gate, by composing `H X H` and `H X H X`
instead of emitting a conjugation. What has no location here is therefore not a
Pauli family but a *placement*: a depolarizing or damping channel, which is
still absent, and a channel bound to a named gate, which is now placed. The
baseline's `x_` and `z_` variants are no longer one of them either, and the
reason is a field of the record rather than a placement: the record states its
readout basis, the sampler reads it, and the rotation that basis adds to the
lowered program is a stated offset beside the round block rather than a round of
its own, so an X-basis experiment's faults land where its own model puts them,
and the section below states that half. The entry points themselves stay absent
on purpose, because a basis is a field of the record here rather than a second
way to ask for one thing. The row stays `partial` on that scope, not on the
connection or on which families reach a sampled record.

The placement grammar is now two grammars rather than one, and the record the
caller states is what selects one, so neither route gained a second entry point.
A `PhenomenologicalNoise` states round boundaries, as before. A
`flagquantum.noise.NoiseModel` states gates, and a rule's fault follows the gate
its rule matched — upstream CUDA-Q's placement for a channel bound to a named
gate, which acts on the state that gate leaves behind — once per round the gate
appears in, so the round structure comes from the lowered program's block rather
than from the record. A rule whose gate the program does not execute places
nothing, because a record stated over a gate set says nothing about a gate
outside it, and a matched rule contributes one location per round while a
round-boundary field contributes one per experiment. On both grammars the engine
still executes one channel, so a phase fault after a named gate is that channel
conjugated by `h` on each side — `H X H = Z`, the same identity the
round-boundary route states as a pair of `h` gates around the flip, and the same
one the construction route states by composing `H X H`. The measurement channel
of a gate-bound record is not placed at all: a readout fault's position is the
check it corrupts, which is what the phenomenological record's measurement
family states, so a rule naming `measure` or `reset` is refused by name rather
than placed after every readout including the terminal data readouts. The
remaining refusals are the grammar's own: a channel that is not a single-qubit
Pauli fault, a one-qubit channel bound to two wires, and a fault that would
follow the program's last instruction. The model and the sampler share the
mechanism list and the lowering, so a record one route can place and the other
cannot is unreachable, and the refusals are asserted on both routes.



**What this round closed instead: the readout basis.** The remaining gap the
row named first was the basis, and it is closed on both routes at once because
it closed in the record. Upstream derives a memory experiment's basis from the
preparation kernel it is handed — `is_z_prep = statePrep == prep0 || prep1` —
and offers `x_dem_from_memory_circuit` and `z_dem_from_memory_circuit` beside
the full one; there is no flag anywhere in its signature, because the circuit
already says which experiment was built. Here the same information has to be
stated, and the reason is a difference in what a record is: upstream is handed
a code *and* an operation, where this record builds one source from a code
alone, so a code that declares a logical observable of each type describes two
experiments and nothing in the record would say which one was asked for. So
`MemoryCircuit` carries `readout_basis`, defaulting to `"z"`, and it is
checked against the code rather than trusted: the builder refuses a basis the
code has no logical observable for, naming both the basis and the type, so a
`RotatedSurfaceCode`, which declares a Z observable only, cannot be read out in
X by asking nicely. `SteaneCode` is the record that can be read either way.
The two upstream entry points therefore stay absent *by name* — the checklist
now lists `x_dem_from_memory_circuit` in `symbols_absent` beside its Z twin —
because a basis is a component of the record here, not a second way to ask for
one thing.

What the basis changes is one rule read over the other check class, not a
second rule. The preparation and the terminal readout share a basis, so the
check class of *that* basis is the one deterministic before the first round and
after the last, and it is the class that gets the round-zero and terminal
detectors and the observable rows; the other class is deterministic only
against the round before it. The count is the same formula with the two classes
exchanged, and the source states the change by rotating: `H` on every data
qubit before the round loop and the same `H` gates again before the terminal
readout. That is one instruction per data qubit at each end, which is why the
sampler states it as a length rather than as a round — a round is a block that
measures every check once, and a plan that read the rotation as one would
attribute a fault to the wrong round or refuse the program outright. The
rotation is the whole difference between the two lowered programs: both bases
lower to exactly one `for round_index in range(rounds):` anchor and the same
round body, which is what the construction route's fault injection reads off,
and the location counts, the round block and the readout offsets inside it are
identical between the two. Both halves are pinned, and they are pinned
separately rather than by one comparison standing for the other:
`test_the_model_and_the_sampler_agree_in_either_readout_basis` runs the exact
forced comparison and the pair-rate comparison in each basis,
`test_the_readout_basis_decides_which_data_fault_moves_the_observable` requires
the fault family that moves the observable to swap between them and the other
family to move nothing, and
`test_the_readout_basis_swaps_which_round_zero_handles_are_pinned` requires the
pinned and free round-zero handles to swap, so an implementation that ignored
the field and always prepared in Z would fail all three.

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
where a memory circuit's is one band per round plus a terminal detector per check
of the readout basis it is built in. Both geometries are pinned separately and neither is allowed to stand for
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

What stays open is the carrier of the rate record and not the rates. `hx` and
`lx` are read, and the four families `CssNoise` carries — `px`, `pz`, `py` and
`pm` — are stated here as `data_flip`, `phase_flip`, `both_flip` and
`measurement_flip` with the same four per-qubit and per-check override vectors
and the same override rule, so a per-location profile *is* expressible and a code
reaches a code-capacity model at one. What is not the same is where the record
lives: upstream folds it into the argument its matrix entry point takes, while
here it is a standalone frozen dataclass whose vectors are resolved against a
count the reader supplies rather than against a noise object that knows it. That
limit is carried by `dem_code_capacity_noise`, which stays `reshaped`. The record
set is the other thing the code-row target implies, and what upstream's own
headers say it contains is in the `qec_code_record` subsection below.

**What `qec_dem_matrices_and_rates` closed.** This row is a data-shape row
rather than a capability of its own, and it was the one that kept moving because
the shapes it pins are what the two rows around it consume. Upstream's baseline
for it names six things, and the row is `aligned` because all six are now
reachable under one orientation: `detector_error_matrix` and
`observables_flips_matrix` are the same matrices with rows for detectors and
observables and columns for mechanisms, `error_rates` is the same one-rate-per-
column vector, `error_ids` is the same optional column that says which mechanisms
are alternatives, `num_detectors()` / `num_error_mechanisms()` / `num_observables()`
are `num_detectors`, `num_errors` and `num_observables`, and
`measurement_to_detectors()` is `DecoderInputs.measurement_to_detectors` from
section 3.5. The closed-form half — `detector_rates` and `observable_rates` — is
this repository's own and is what makes the marginal rates an exact statement
rather than a simulation.

The rate column was the last of the three additions the row listed, and it is
worth stating what it is *not*, because the obvious implementation would be
wrong. `DetectorErrorModel.error_rates` is not a probability distribution over
the model and it is not `detector_rates()` under a new name: entry `i` is the
rate mechanism `i` itself states, so a model whose two mechanisms both touch
detector zero reports `(0.1, 0.2)` there while `detector_rates()` reports
`0.26`, and a model that states two mechanisms are alternatives reports each
member's own rate rather than the group's summed mass. Folding a group stays in
the marginal accessors, where the group is a statement about which mechanisms
fire together rather than a rewrite of what any one of them is worth. The test
asserts both readings side by side on one model, so an accessor that returned the
marginals, composed two mechanisms on one detector by parity, or sorted the rates
would fail rather than pass.

The one property the vector rests on is that its order is the matrices' column
order and not the caller's argument order, and that is a fact about the model
rather than about the accessor: a `DetectorErrorModel` normalizes its mechanisms
into a single sequence, and every column view — both matrices, both vectors — is
read from that sequence. Two models stated in opposite orders therefore produce
the same matrices *and* the same rate vector, which is what lets a matcher take a
column's support and that column's weight from one index. The test pins each
entry against the column it weights rather than against the vector alone.

What the row does not own, and therefore does not wait on, is canonicalization.
The round structure a matcher would read is the operation `canonicalize_for_rounds`
performs upstream, and here it is recorded where it belongs: as the `absent` field
row `dem_canonicalize`, whose negative search names the symbol, with
`qec_dem_chunking` now carrying the round-window identity that operation would
sit behind. That absence is a fact about a different operation rather than a gap
in this row's surface, so the row is `aligned` and the absence is named in one
place instead of two.

**What `qec_detector_annotations` closed, and what it deliberately did not.**
The row had two halves and only one of them was about spelling. The first half —
detector and observable identity derived from the code rather than asserted
beside it — was already done when the layouts landed: the detector count is the
code's checks and rounds, every reference is tied to a declared wire and round,
and the observable readout is refused unless it is terminal and matches the
declared support. The second half was the one that read `absent`, and it was
read that way for a good reason. In a CUDA-Q kernel a measurement *handle* is a
value: `z = mz(q)` names a measurement, `detector(z)` consumes it, and `to_bools`
or `to_integer` reads the same handle on another line without measuring again.
This layer had no such value. A recorded bit was a column index inside the
lowering, and the layouts named references rather than handles, so a caller who
wanted a bit that no detector names — which is exactly the bit an X-type check
measures in round zero — had nowhere to ask.

`MemoryCircuit.measurement_refs` and `MeasurementSamples` are that half, and they
are deliberately not a kernel annotation. The handle vector is read off the code
and the round count rather than collected from the layouts, so a position in it
names a recorded bit and not a caller's guess, and the two sets need not agree in
either direction — which is the point, because the reason this layer is worth
having is that the layout layer is right to drop some bits. Round zero of a
surface patch measures an X-type check whose outcome is not deterministic under
the all-zero preparation, so no detector may name it; the bit is still measured
and is now still readable. `MeasurementSamples` holds that run's outcomes,
`outcome` reads one handle, and `vector` and `integer` read a chosen sub-vector
as booleans or packed into one integer, with the caller's order preserved in both.
That is `to_bools` and `to_integer`'s job done over the record that holds the
bits, rather than over a per-measurement conversion call, and the two names stay
absent for that reason rather than by omission.

The half that stays absent is the annotation *form*, and it stays absent on
purpose. CUDA-Q sets detector identity in the kernel body, where the compiler has
to carry it into IR, and the price of that is a handle type, an interception rule
and a host-scope refusal. This layer sets the same identity in a record beside a
source string that a bounded hybrid capture would refuse a channel call inside
anyway, and it pays for that with the refusals the layouts already enforce. The
three names `detector`, `detectors` and `logical_observable` therefore stay in
`symbols_absent`, so the row cannot drift into claiming a kernel-level annotation
surface it does not have, and the same list is what makes the row fail the day
one of them appears.

What makes the two halves one story rather than two is a test that recomputes
every detector's parity and every observable's flip from the handles that detector
and observable name, and requires the two readings of one run to agree shot for
shot. The readings cannot disagree, but they can be read wrongly, so the tests
also pin the parts that are easy to get backwards. The packing is checked against
the bit vector rather than against a second call of the same method, so the two
readings are not each other's witness; two orders of the same handles state two
integers, so the order is the caller's and not the record's; a handle the
experiment does not record raises with the handle named rather than reading a
neighbouring column; an empty reading raises; and a record whose tensor is not
two-dimensional or whose column count does not match its handle vector raises
rather than being truncated to the shorter of the two.

The measured part of the layer is the border between determinism and freedom, and
it is asserted as a rate rather than as a convention. A noiseless run reads the readout
basis' own check class as 0 in every round — a check of that class leaves the
prepared state alone — the other class as unbiased, the two swapping places when
the experiment is prepared and read out in the other basis, and an individual
terminal data wire as unbiased, because the gadgets of the other class entangle
the data with their ancillae; and every detector and every observable reads 0, because the layouts
name parities rather than handles. The two failure modes that pins are symmetric:
a channel rate read off a single handle would be a rate read off the state, and a
detector claimed to be a bit would be a parity claimed to be a measurement. Under
`measurement_flip = 0.25` the pinned handles fire at 0.25 and the free ones stay
near 0.5, which is the arithmetic of flipping an unbiased bit rather than a
statement about the channel; the test splits the handles by what they measure for
exactly that reason, since pooling the two kinds would average the distinction
away.

**What converging four rows on their own targets means, and what it does not.**
`qec_code_record`, `qec_dem_construction`, `qec_dem_text_interchange` and
`qec_stim_sampling_join` were `partial`, and the prose each carried in
`next_action` read as work outstanding. Reading that prose against the row's own
`target`, clause by clause, showed it was not: every clause of all four targets is
satisfied, and what `next_action` named was a *divergence this repository declines*
rather than a gap it has not closed. `qec_code_record` declines the baseline's
per-operation kernel map, so a code here is a record that builds a source rather
than a map from an operation to the kernel that performs it.
`qec_dem_construction` declines the same form through a different name: a model
built from a kernel's own detector annotations, so no X-type or Y-type fault family
follows from a kernel body. `qec_dem_text_interchange` and
`qec_stim_sampling_join` decline an arbitrary annotated circuit at the input end —
the baseline's `stimulus` — so a model comes from a circuit record plus a noise
record and the join is over a circuit this layer builds.

The distinction is the one `qec_detector_annotations` already draws, and it is the
only reason those four rows moved. A clause of a `target` is work: it stays
`partial` until it is done. A *form* the baseline has and this repository answers
differently is a divergence, and a divergence belongs in `fail_closed` beside the
name that would make the row stale, not in `next_action` as though someone were
still going to do it. Each row therefore keeps every `symbols_absent` and
`negative_search` entry it had — `CssCodes`, `CssNoise`, `stabilizer_round`,
`operation_encodings` and the colour-code search on the first; `dem_from_kernel`,
`DEMResult`, the baseline's two entry-point spellings and the two basis-directed
variants on the second — and the two text-and-sampling rows carry none, which is a
decision rather than an omission: the name an arbitrary annotated circuit would
arrive under is not a name this repository would define, so there is nothing for
the rule to watch. That is the same state `qec_dem_matrices_and_rates` is in, and
the divergence is held there by the type the sampler states, since anything that is
not a `MemoryCircuit` is refused as a `TypeError`. The text row's precision
difference is likewise stated as a tolerance the caller carries — one part in
`10**15`, measured at a `5.4e-16` worst case — rather than as an open item.

**No matrix row changed status.** `qec_code_library` and `qec_stim_integration`
are still `partial`, and both still state the kernel-expression divergence in
their own `reason`. An `aligned` alignment row is a statement about that row's
own target, and a matrix row covers more than one alignment row: the record half
of `qec_code_library` is closed while its kernel half is not, and both halves of
the stim interchange are closed while the input end is not. Reading the two
documents as one claim is what the checker prevents rather than what it performs —
`aligned` beside a matrix `partial` is permitted exactly because the two are not
the same statement, and `aligned` beside a matrix `unsupported` or `out_of_scope`
is refused because there they would be.

**Why the remaining `now` rows did not move in the same change.** The word
*stimulus* is what let the drift stand for four rounds. In
`tests/qec/test_dem_stim_interop.py` it names the stim detector error model
**text**, which this package reads and writes; in the checklist and the matrix it
named the **unbuilt input end** — an annotated circuit with no route to a model or
a sample. One word, two referents: a format that is read, and a form that is
declined. Prose that reused the first sense to name the second is how a declined
divergence read as an open gap. The two rows that remain `partial` are held back
for a different reason and not for this one: `qec_dem_chunking` and
`qec_decoder_family` are both modified by the open sliding-window pull request, and
converging a row in a change separate from the change that modifies it would stack
two pull requests on one line. Each converges with the change that modifies it,
from the `main` that change lands on.

**What held the checker's own mutation test together.** Two of the 35 mutations in
`tests/unit/test_qec_cudaq_alignment_check.py` were planted in rows that this
series of changes converged, and each time a row moved, the mutation stopped
testing the branch it exists to reach: one could not be applied at all, because the
row no longer had the `next_action` key, and the other fired the `aligned` message
where its stated substring is the `partial` one. Re-pointing them at whichever row
happens to be `partial` today only moves the problem to the next convergence, so
both now state the precondition they test. The `next_action` mutation sits on
`qec_decoder_configuration`, an `absent` row, because `next_action` is required of
every row that is not `aligned` and the checker asks for it in one loop; the
`symbols_present` mutation writes the `partial` status it is about instead of
borrowing a row. The test was then run against a checklist in which **all nine
`now`-floor rows are `aligned`**, with the two remaining rows' `next_action` keys
removed, and all 35 mutations still pass — which is the evidence that the coupling
is gone rather than deferred.

## 3. The 23 field rows — the detector error model against `DEMResult` / `dem_from_kernel` (both CUDA-Q core)

The short version, because the full table is in the TOML. Across 23 field rows:
2 `equivalent`, 2 `renamed`, 1 `extra`, 14 `reshaped`, 4 `absent`.

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
therefore offered as a documented approximation of the line, not as an alternative
statement of it, and
`test_the_suggestion_reading_departs_from_stim_where_the_default_reading_does_not`
asserts the separation, so a tolerance that both readings passed would not be
evidence about either.

What remains of upstream's two stated losses is the error-id one, and that is a
property of the record on both sides rather than of this reader. The rest of the
refusals differ in kind, not in spirit: upstream hands the text to stim and states
what it loses, this repository refuses a construct its own record cannot hold (a
declaration that skips an index, an error mechanism that flips nothing, a
malformed line, a block that never closes, a closing brace with no block open)
rather than dropping it — and one construct more under the expanded reading alone,
a component that cancels to nothing, which no mechanism can state. `repeat` blocks
and `#` comments are read rather than refused, and a block is read by expansion,
which is the one cost this reader pays that stim's does not; the sixty swept models
that carry a block are read as the model stim's own `flattened()` form states.

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

**Reshaped (14).** The model carrier, the per-error rates, the error ids, the
counts, the sampling function, the memory-circuit entry point, the matrix-level
entry point, the noise record, the measurement-to-detector map, the
seam-and-chunk surface, the kernel annotation surface, the decoder registry, the
decoder result record and the decoder base protocol. The widest of these:

- **Seam and chunk surface.** Upstream a model is a sequence of chunks with named
  seams that can be stitched and closed, and the seam names are values derived
  from a name string so that a seam identity is stable across processes. That
  surface now exists here, and the four differences are decisions rather than
  omissions. The seam identity is the name itself rather than an `uint32` hash of
  one, so nothing hashes and nothing interns. A seam's rows carry the global
  detector index rather than a caller-chosen tag numbered positionally within the
  seam, which is what lets `dem_stitch` refuse two boundary bands of equal width
  that are not the same boundary. The spec is linear — `init`, a repetition of
  `bulk`, `final` — where upstream's is a phase graph with caller-named phases,
  edges, a self-loop for the repeating phase and a round count supplied at
  expansion, so what upstream spells `expand_dem_chunks` is
  `DemChunksSpec.chunk_specs` here and a second name would be a second spelling
  of one operation. And upstream's `extended_dem` record — a model type holding
  the check matrix, the observable matrix, the rate vector and the seams — has no
  counterpart because a window here is a plain `DetectorErrorModel` plus its
  seams, which is the same decomposition without a second model type. The
  chunk-scoped matrix projection has since landed beside the sliding-window
  decoder rather than being part of this record, and that is recorded in
  `qec_dem_chunking`: a window's matrices are the input shape of the decoder that
  reads it, so `dem_chunks_to_pcm`, `dem_chunks_to_o_sparse` and
  `dem_chunks_to_d_sparse` read a sequence through the model it decomposes and
  this row's `extended_dem` absence is unaffected by them.

- **Per-error rates.** Upstream keeps one parallel `error_rates` vector beside
  the matrices; here each mechanism's rate is a field of its own record and
  `DetectorErrorModel.error_rates` projects the vector back out, so the two views
  cannot drift the way a stored pair of parallel arrays can. The vector is the
  same length as a matrix's column count and in the same order — the model's
  normalized mechanism order, so entry `i` and column `i` are one mechanism —
  which is what lets a reader take a column's support and its weight from one
  index. It is deliberately not a distribution: a mechanism reports the rate it
  states, so a grouped model reports each member's own rate and the folding stays
  in `detector_rates` and `observable_rates`, which are different quantities
  again (marginals over detectors and over observables, not per mechanism).

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
  sparse forms a realtime decoder configuration takes, the last three of which
  have since landed as the chunk-scoped projections of the sliding-window
  decoder. Here one record,
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
  chunk-scoped helpers remain absent, which is a different absence from the one
  `dem_seam_and_chunk_api` now records: a model does decompose into windows over
  named seams, but a window's matrices are the input shape of the decoder that
  reads a window, so the projection lands with that decoder and the alignment row
  `qec_dem_chunking` names it. The one property the whole arrangement rests
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

**Absent (4).** `canonicalize_for_rounds`, `dem_from_kernel` itself, the sampling
backend selector, and the plugin boundary upstream keeps for open decoders. The
seam/stitch/close family is no longer among them, and `qec_dem_chunking` records
what of it landed and what did not. The chunk-scoped matrix projection
(`dem_chunks_to_d_sparse`, `dem_chunks_to_o_sparse`, `dem_chunks_to_pcm`) has
landed as well, with the sliding-window decoder that reads a window: it belongs to
that decoder rather than to the model record, it is recorded inside
`dem_measurement_to_detector_map`, `dem_seam_and_chunk_api` and
`qec_dem_chunking`, and the field row it is read through is unchanged, because
none of those three projections states a model of its own.

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
   it and only the cudaq-qec half decodes.

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

6. **The upstream tree the field rows were read from was renamed, and its
   negatives were unpinned.** The contract recorded the source read as
   "NVIDIA/cudaqx main libs/qec". That name now redirects: the API answers 301
   and the git remote resolves it to `NVIDIA/cudaq-qec`, which is where the tree
   lives, and `libs/qec` is still its path inside. **Landed as** a corrected
   baseline name plus the revision the read was taken at,
   `8df583e6` (the tree the checkout resolved to), and the same correction in
   the parity contract's `component` string, so the two contracts name the
   repository that serves them rather than one that forwards to it. The
   revision is what made the three `UNVERIFIED` marks in this contract
   reviewable rather than permanent: each was a negative taken over a moving
   source — no `extract_syndrome` in the cudaq-qec tree, no `for-stim-users`
   page in its documentation, no stim-text writer in its headers, library,
   bindings or examples — and each is now stated against a revision a reader can
   take, so the marks come off and `qec_syndrome_extraction_owner` closes.
   The rule the mark encoded is kept: it is held by a mutation that plants it
   rather than by a row that happens to carry it, so settling the last negative
   could not quietly retire the check.

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
  `provenance_unverified` says `UNVERIFIED` in its own note, and no row carries
  that mark because every upstream negative here is pinned to the revision it
  was taken at, so the rule is held by a mutation that plants the mark rather
  than by an example;
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
| `dem_error_rates` | reshaped | Same quantity and now the same vector; per-mechanism record projected out rather than stored per column. |
| `dem_error_ids` | reshaped | The correlation is stated, in the mechanism's own record: `DemError.error_id` groups mechanisms that are alternatives, `DetectorErrorModel.error_ids` projects upstream's parallel vector back out, `None` where upstream has `nullopt`, and the marginal rates and the sampler read a group as one fault. Upstream fixes neither the distribution a group implies nor what happens when a group's probabilities overflow a shot; here the members are disjoint pieces of one shot and an overflow is refused rather than renormalized. Stim text cannot carry the statement, so `to_stim_text` refuses an id-carrying model and names the ids. |
| `dem_counts` | reshaped | Methods upstream against properties here; `num_error_mechanisms` against `num_errors`. |
| `dem_sampling_function` | reshaped | Free function over matrix + rates returning syndromes *and* mechanisms, against a model method returning detectors + observables. |
| `dem_sampling_backend` | absent | No execution-target selector; upstream has `auto`/`cpu`/`gpu`. |
| `dem_from_stim_text` | renamed | Same operation, same parser authority (stim), same `use_decomp_suggestions` flag with the same default, free function against classmethod; the default reading is the one stim's own sampler means, and the expanded one is the approximation upstream documents it as. |
| `dem_to_stim_text` | extra | No writer found upstream at this layer; the produced text comes from CUDA-Q core. |
| `dem_from_css_matrices` | reshaped | Same code-capacity geometry, reached from two keyword matrices instead of one four-matrix record; `hx`/`lx` and the extended-record sibling have no counterpart. The rates arrive as a separate `PhenomenologicalNoise` rather than folded into one `CssNoise`, and the vectors are read against these matrices. |
| `dem_from_memory_circuit` | reshaped | Upstream takes code + operation + rounds + noise model and is split by basis; here the circuit carries rounds and basis, and the noise record states the four families with a per-qubit and per-check override each. The `decompose_errors` argument has a counterpart of its own now — `decompose_composite_faults` splits the Y family into the X and Z faults it is the XOR of, at the parent's rate and read off the program — but it is not upstream's rule: upstream's pairing is whatever Stim chose, while this one is derived from the program and claims only that a part is graphlike where the corresponding single-Pauli fault is. The context matches, but upstream canonicalizes lazily against a uniform per-round D layout and here the components are projections of the model as built. |
| `dem_code_capacity_noise` | reshaped | The four families line up one for one against X/Y/Z data rates plus a measurement rate, scalars and per-qubit/per-check vectors alike, with the same wholesale override. The difference is the carrier: a standalone record read beside a code rather than a field of the matrix entry point's argument, so the vector lengths are validated against a count the reader supplies. |
| `dem_canonicalize` | absent | No round-folding operation, so a model that states a group of alternatives still reaches a matcher unfolded; the round *structure* a window needs is now stated by `dem_seam_and_chunk_api`, and folding the group under its exclusivity is what remains. |
| `dem_merge_operation` | renamed | Same two rules and the same formulas, free function with a mode enum against a model method with an enum of its own; the uniqueness assert is called here rather than merely offered. |
| `dem_seam_and_chunk_api` | reshaped | Windows over named seams on both sides; the seam identity is a name rather than a hash of one, a seam's rows are global detector indices rather than positional tags, the spec is linear rather than a phase graph, and the chunk-scoped matrices landed with the sliding-window decoder as `dem_chunks_to_pcm` / `dem_chunks_to_o_sparse` / `dem_chunks_to_d_sparse`. |
| `dem_measurement_to_detector_map` | reshaped | One `MeasurementMap` record against a stored dense D matrix plus free-function sparse helpers: it projects to both forms, and a model now decomposes into windows over named seams; the chunk-scoped helpers landed beside this map rather than inside it, because they read a sequence through the model it decomposes, and `dem_chunks_to_d_sparse` is the one that meets this map's own measurement-bit numbering while refusing a non-uniform round width by name. |
| `kernel_annotation_surface` | reshaped | Layouts beside the source against annotations in the kernel body over measurement handles. |
| `kernel_dem_from_kernel` | absent | CUDA-Q core derives the DEM from the kernel's own annotations; here it is assembled by hand. |
| `decoder_registry` | reshaped | Upstream reaches a decoder by name — `get_decoder(name, H_or_dem_text_or_sparse_matrix, **options)` with a decorator putting a class behind a name; here `flagquantum.qec.get_decoder` takes a carrier and `register_decoder` puts one there, checked while the registering module is imported. Narrower on two deliberate points: the source argument is one of the three carriers a caller can hold a model in rather than a parity-check matrix, and only the detector-error-model family is registered, because the repetition-code decoders take an ordered syndrome history rather than detection events. |
| `decoder_result_record` | reshaped | Single-shot record against `converged` + optional results, with separate batch and async records. The DEM-level matcher adds `MatchingDecodeResult` and the belief-propagation decoder adds `BeliefPropagationDecodeResult`, which is the one carrying the convergence flag; the sliding window adds `SlidingWindowDecodeResult`, whose `converged` is the inner flags anded or `None` when none reports one, and which is the one local record that states a stream rather than a shot. None of the three is this record and none has a batch or async form. |
| `decoder_base_methods` | reshaped | `decode` + streaming against `decode`/`decode_batch`/`decode_async`/`get_block_size`/`get_syndrome_size`/`get_version` and an errors-vs-observables request. The DEM-level decoder does not implement the repetition-only protocol, because no correction record here can express a surface-code correction; the sliding window instead states two entry points — a whole block in the model's numbering and one round in that round's own — and its record reports the committed faults and the observables that same vector flips, which is upstream's errors-vs-observables request answered rather than asked. |
| `decoder_plugin_precedent` | absent | Upstream integrates open chromobius and pymatching plugins behind a boundary while its closed decoder ships as a binary — the same split the matrix plans. The integrating half of that split is now realized for one library, as the PyMatching cross-check behind an extra; what is still absent is the boundary itself, since the adapter is a concrete class rather than a registered plugin. |
