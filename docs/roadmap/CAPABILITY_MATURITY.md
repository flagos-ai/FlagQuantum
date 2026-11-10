# Capability maturity

`capability-maturity.toml` is the authoritative, machine-validated capability
matrix. Marketing text, benchmark summaries, and release notes must not assign
a stronger status than this matrix.

The levels are deliberately non-interchangeable:

| Level | Meaning | Permitted claim |
| --- | --- | --- |
| `experimental` | Unstable research implementation. | The implementation is available for evaluation. |
| `development_evidence` | Executed development or semantic evidence. | The constrained path was executed; no production or scalability claim. |
| `production_supported` | Supported path with integration, hardware, and operational evidence. | The documented workload and environment are supported in production. |
| `release_certified` | Audited and reproducible release evidence with a release gate. | The named release artifact is certified for its exact declared scope. |

Status is capability-specific. A stable public API does not promote an
experimental backend, and CPU semantic evidence does not promote a distributed
runtime. Promotion requires adding every evidence field required by the target
level and passing `python tools/check_capability_maturity.py`.

Schema v2 also makes public performance claims fail closed. Each claim must
identify a checked-in raw JSON artifact and its SHA-256 digest, match the code
version recorded by that artifact, inherit the capability maturity, derive its
displayed measurements and recorded environment from JSON selectors, and state
its exact scope and metadata boundary. The documentation generator emits the
same validated values into the README, capability catalog, and Known
Limitations; edits inside generated regions are rejected by the CI check.

Agreeing with the artifact is not enough for the recorded code version, because
both values are written by the same check-in: a revision that names no commit
satisfies that comparison exactly like one a reader can check out. Each claim
therefore also declares `code_version_origin`. `repository_history` requires the
revision to resolve against this repository, which is what makes the evidence
reproducible and is the expected value for a run recorded from a merged
integrator commit. `producing_host_history` states that the run happened in a
history this repository does not contain; the pin is then published with that
disclosure rather than presented as a revision to check out. The check fails
closed in a shallow clone, which cannot separate a revision it never fetched from
one the repository never contained, so repository-history pins are resolved by
the full-history `quality` job. `tools/evidence_provenance.py` owns both values
and applies the same rule to the checked-in evidence under `artifacts/`.

Applying that rule to one artifact at a time left most of the evidence unexamined.
Only six checked-in validators resolved a revision, and each had its artifact path
written into it, so an artifact no validator named was never asked: nineteen
artifacts under `artifacts/` recorded revisions this repository does not contain,
and thirteen of them said nothing about it. `tools/check_evidence_revisions.py`
therefore walks every citation file type under `artifacts/`, `benchmarks/results/` and
`docs/development/evidence/` rather than a list, and asks both questions a reader
depends on. A revision is obtainable when this repository holds the commit and a ref
of this repository reaches it, because a clone and a plain fetch obtain exactly the
commits the refs they fetch reach; of the sixty-three distinct revisions recorded
under those roots, one is obtainable. Every other revision must be declared in
`evidence-revision-origins.toml`, which names the origin a reader can obtain it
from. Asking only whether the object was in the database was not enough:
forty revisions are unreachable from every ref, so a clone obtains none of
them, and thirty-one of the forty are absent from this checkout's object database as
well, so an answer read from the checkout would differ between checkouts and the
declaration is what makes the verdict the same everywhere. `unreferenced_object`
covers those, since the remote still serves the object by name while nothing
guarantees that it will. The class is measured rather than frozen: its most recent
entry is a revision recorded by an artifact merged after this table was written, whose
branch was deleted when its pull request merged, so an unobtainable pin reached `main`
within a day of the gate that asks about it. `producing_host_history` covers the
eighteen revisions of history this repository does not contain. Seventeen belong to
`FlagQuantum/FlagQuantum`, a private repository re-created from a product baseline
rather than cloned from this one; this repository's public history carries
look-alike commits with the same subjects but different objects, trees, and
parents, so a reader can obtain none of the seventeen and re-pointing a pin would
present a run as having executed code it never executed. The eighteenth,
`6cfc4c3708227a4bb2c6e35c9f71e502239a2e63`, is published in no repository at all:
two cross-framework comparison artifacts record it as `source_revision_full` after
recording a null commit, because the runtime containers ship no git binary, so the
`runner_sha256` read back inside each container is the only link from the
measurement to the code. `external_dependency` covers the four revisions of code
this repository does not vendor and names the repository a reader obtains each from.
A revision that a clone obtains needs no origin, and declaring one for it fails the
gate, so the table cannot be used to hide a checkable pin behind a claim. A
declaration names an artifact or a directory, so a bulk evidence surface states one
origin once instead of repeating it per file. An artifact may also account for its
own revisions by stating `<field>_origin` beside any field that records one, which
six artifacts under `artifacts/` do and which
`docs/development/evidence/jiuding_workspace_credentials_20260910.json` does for the
one abbreviated value no measurement here expands; both records are read, and a
revision whose two records disagree fails the gate rather than passing twice.

The gate walks every citation file type under exactly those three roots, which is a
boundary rather than a claim of full coverage, and what falls outside it is measured
rather than assumed. Counting a pin as any full-length hexadecimal value in a tracked
file outside those roots, and a pair only when no walked artifact records that
revision, sixty-nine (file, revision) pairs across fifteen files record a revision no
walked artifact records: sixty-three of the pairs are under `docs/`, five under
`examples/`, and one in `.github/`. Those pairs name fifty-four distinct revisions, of
which fifty-one resolve in this checkout and three do not. The count is a measurement
of this tree rather than a constant, because it includes the pins in this document: a
paragraph that names a revision raises the number it reports, so a reader who repeats
the method here reads a later count than the one written down.

Adding `docs/development/evidence/` accounted for the largest group of those pairs
rather than only reporting it. All fourteen unreferenced revisions that root's
artifacts record are declared in `evidence-revision-origins.toml` -- five written out
in full by the four Jiuding records that did so when the root was added, and nine that
the field-name rule reached afterwards -- so the change states the
unreferenced-object class fourteen entries larger than the walk alone would have.
Those records are what `docs/guides/JIUDING.md`, the
release notes and this document cite as their hardware evidence, and until this root
was walked their pins were disclosures a reader relied on that nothing checked.

Whether that origin is the right one turns on a measurement, so the measurement had
to be worth trusting, and the first attempt at it was not. Separating
`unreferenced_object` from a revision no repository publishes is what separates a
remote that still serves the object by name from one that does not, and the first
probe counted a failed `git fetch` as a refusal -- on a host that loses connectivity
to `github.com` for minutes at a time, which turns a dropped connection into a claim
about the server. A failed attempt is uninformative in either direction: one of the
twenty-five failed an attempt while the remote's ref advertisement was succeeding, and
then served twenty of twenty further attempts. What supports the class is the
successes, so the method counts only attempts in which the ref advertisement succeeds
immediately before and after the fetch. All twenty-five reached three counted fetches
that way; under the earlier method four of them had been recorded below three, which
reads as objects the server may already have dropped. `evidence-revision-origins.toml`
states the method and its result. No check re-runs it, because the checks in this
repository do not use the network, so this is the one claim behind the table that
rests on a stated method and a recorded count rather than on a green check.

That fetch count no longer answers, which is why the class now rests on a second
measurement that can be repeated. Fetching a revision by name from the public remote
hangs from this host instead of failing -- a bounded attempt on two of the revisions
ended in a timeout after ninety seconds, and a timeout is uninformative in either
direction. What answers instead is a request for the commit object, scoped to the
repository the revision is declared against: GitHub's
`GET /repos/flagos-ai/FlagQuantum/git/commits/<sha>`, and the same value under
`/commits/<sha>`, each asked twice. That endpoint answers 200 for all forty
`unreferenced_object` revisions, 404 for all eighteen `producing_host_history` ones
and 404 for all four `external_dependency` ones, while also answering 200 for a commit
on `main`, 404 for a commit of `pytorch/pytorch` and 404 for a commit that exists only
in the checkout that produced it -- so the endpoint reads this repository's object
database, not GitHub's, and the forty it holds are exactly the forty no ref
reaches. It is reproducible with `gh api` by anyone who reads the table, which the
fetch count is not.

What remains outside the walk is three values, and none is a pin a reader is asked to
check out. `.github/workflows/ci.yml` compares a base revision against the
git zero-SHA placeholder, so the all-zero value is a shell literal rather than a
revision. `examples/assets/manifest.json` records a revision of
`google-bert/bert-base-uncased` in `assets[2].revision` and `assets[3].revision` --
the value quoted as `BERT_TOKENIZER_REVISION` in `examples/quantum_transformer.py` --
and `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` is the same class in `stanfordnlp/imdb`;
no FlagQuantum repository contains either and neither is this repository's to account
for. The class the walk left outside until now -- a revision named in prose by a
comparison record under `benchmarks/results/` -- is no longer outside:
`benchmarks/results/comparison/FLAGQUANTUM_QISKIT_AER_CPU_ARM64_20260923.md` and
`benchmarks/results/comparison/CUDAQ_BACKEND_COMPARE_A800_20260930.md` are asked about
the three revisions they name, `aaa3a70c698dc629c37baa2719607444aa3b6381` among them,
and the three are declared in `evidence-revision-origins.toml` against the files that
name them. Which file type a revision is written in no longer decides whether a reader
is told what it is.

The walk asks what an artifact records; it does not ask whether a reader can turn the
citation into a commit. A shortened hexadecimal run cannot be expanded by a reader
who does not already have the commit, so `tools/check_evidence_revisions.py` also
requires any shortened run under a declared citation root to be given in full in the
same file, and it reports a run that names exactly one revision this repository gives
in full, because that is what separates a citation from a task identifier or a date.
Eighteen such citations were reported at this branch point and all eighteen were
completed: one in the Quafu artifact, nine in three Jiuding records, one in a CUDA-Q
comparison record, one in the Quafu live-execution document, one in the split
real/imag contract, and five in this document. Completing the Quafu artifact's
`source_revision`
is why the table gained an `unreferenced_object` row: the eight-character
pin named a commit this repository does not hold, and writing the revision out in
full is what let the gate ask about it for the first time. Three Jiuding records were
already inside the walk before their citations were completed, and the gate read
nothing in them until then: a widened root is not enough if the artifact abbreviates
the revision it records. The `docs/development/INTEGRATION_WORKFLOW.md` and
`docs/roadmap/QUANTUM_KERNEL_ARCHITECTURE_PLAN.md` citations that abbreviate a
revision do state it in full in the same file, so the check leaves them alone.

The citation rule can only report a prefix it can expand, and it expands a prefix
against a full-length value this repository records somewhere. Counting every JSON
string scalar of 7 to 39 hexadecimal characters under the three walked roots gives
979, and the value test cannot separate a citation from the rest: `deadbeef` and
`2609091513234674683` are not less hexadecimal than a revision. Sixteen of the 979
sit in a field named for a revision instead -- `base_commit`, `host_commit`,
`implementation_commit`, `source_commit`, `source_revision`,
`validation_driver_commit` -- and a field named for a revision names a commit by
construction, which is a property of the name and not of the value. Before this
change nothing in this repository expanded any of the sixteen, so neither half of the
gate asked about them: the revision walk needs forty characters, and the citation rule
needs a value to expand against. One of the sixteen, `6cfc4c3`, was already answered:
the two CUDA-Q comparison records state the full value beside the recorded
abbreviation, which the check accepts rather than asking that recorded output be
edited. A pin no check can read is the kind of disclosure this table exists to remove,
so the walk now reads a shortened value in a revision-named field. It reads a
suffix rather than a list, because a list is a surface an artifact extends by naming a
field: any name ending in `revision` or `commit` is read at any depth, the reviewed
names are declared in `REVISION_FIELD_NAMES` for a reader rather than used to decide
what to read, and a name this repository has not reviewed is reported rather than
ignored. The check asks a shortened value the two questions it can answer: whether the
artifact states the revision in full beside it, and whether it states the origin that
accounts for the value instead. Stating an origin beside one field accounts for every
field holding the same value, so an aggregate that repeats its runs' `source_revision`
is asked once, because the origin belongs to the revision rather than to a repetition
of it.

Fifteen values were left, and fourteen of them now have the revision written out in
full in the artifact that records them, so the same two questions the walk asks of
every other revision are asked of these for the first time. All fourteen answer
`unreferenced_object`: four MPS dispatch records under
`benchmarks/results/local` -- environment, environment-channels, one-site and
projected-two-site -- the pool-workspace summary
under `benchmarks/results/smoke/release_candidates`, and nine Jiuding records under
`docs/development/evidence/` -- each an object of this repository's remote that no ref
reaches, so a clone obtains none of them, and each was measured with the
repository-scoped commit request the table's method paragraph states rather than
inferred from the field it sat in. Four of those artifacts record the value once per
run and repeat it in an aggregate, so the revision is written out beside each run and
the aggregate's copy is accounted for by the same artifact, because the origin
belongs to the revision rather than to a repetition of it. Writing the revision out in
full is the stronger of the two answers the check accepts, because it lets a reader
obtain the commit rather than only naming who holds it, and it is what made fourteen
unchecked pins into fourteen measured ones. The fifteenth, `host_commit` in
`docs/development/evidence/jiuding_workspace_credentials_20260910.json`, is the one
abbreviated revision-named value under the walked roots that no measurement here
expands: the same request answers `404` for it at both endpoints, so the artifact
states `producing_host_history` beside the value and the value is kept as the run
recorded it. The table gained no new class, and no recorded value was rewritten to fit
one.

`examples/` is why the root boundary is a decision rather than a missing line.
`examples/assets/manifest.json` records `e628166` in `assets[0].revision` and
`assets[1].revision` under the field name `revision`, which is now a name the walk
reads, and the value is a revision of `stanfordnlp/imdb` that Hugging Face resolves
to `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` and this repository does not hold and
should not. Its `assets[2].revision` and `assets[3].revision` hold the full-length
`b644e1d1cc3f080c2e6d4db69cfdfac83719c814`, a revision of
`google-bert/bert-base-uncased` -- the value quoted as `BERT_TOKENIZER_REVISION` in
`examples/quantum_transformer.py` -- and the manifest's schema is a third-party asset
record rather than a FlagQuantum evidence artifact. Reading that field as a
FlagQuantum pin would ask a reader to check out a commit of another project, so the
root stays outside the walk and the reason is stated here rather than left to the
next person who widens it. The field-name rule is what makes that boundary
expressible: a root may now be added once its revision-named fields are accounted
for, and this root's are not this repository's to account for. `ci/` is left out for a
different reason and does not wait on that work:
`ci/flagos_cuda_reference.lock.json` records
`2e00b393cf80088706b460a187aef185d3a283f4`, an external dependency already declared
for the artifacts under `artifacts/`, but two
validators read that file as an input, and a lock file pins an installation rather
than recording what a measurement produced.

The split real/imag P5 line currently exposes an experimental CPU-only
first-order PyTorch autograd bridge over P4 Double-Single execution. Its
returned loss and `.grad` are explicitly FP32 delivery boundaries. A separate
experimental CPU Double-Single SGD lane consumes explicit high/low gradients
and retains a high/low master parameter without using `.grad`. The identical
optimizer trajectory has development portability evidence on native A800 CUDA
and CUDA-backed Torch-FL `flagos:0`; this is not domestic-hardware or
convergence certification. Richer optimizers, higher-order autograd, and
distributed execution remain gated future work. A separate fail-closed domestic
single-card harness now collects provisioner-attested P0-P5 execution candidates;
no domestic capability is promoted until a real-card result is reviewed and
checked in.

## Private training compilation development status

The approved #936 boundary has a reusable CPU Torch training candidate behind a
closed private promotion gate. Numerical/recovery conformance and candidate
measurements do not promote default training performance or a new capability
level. In particular, ordinary short jobs remain eager, Adam optimizer updates
remain eager, and no GPU/distributed performance or whole-step fusion is claimed.
The parent #918 acceptance remains separate from this implementation slice.

## Qmod unsigned record integration slice

The UP-06 integration branch extends flat bit records with fixed-width unsigned
integer scalar/array fields. Five unchanged pinned originals have two nonzero
parameter sets checked against independent full-state Circuits and classical
cost-phase oracles. This is a local exact CPU correctness slice, not corpus-wide
completion or a capability-level promotion. The corpus gate and full integration
CI remain required before release.

Two-dimensional fixed bit fields extend that same checked layout and physical
storage. Row/column access, row views and named row readouts are part of the local
UP-06 candidate; they do not promote general nested records or numeric matrices.
The original electric-grid, induced-color-subgraph and minimum-coloring programs
retain their source bytes, layer counts and independent full-state acceptance
requirements. Remaining corpus blockers and the final integration gate stay open.

The guarded-real-phase slice adds bounded integer predicates and classical
slice-element binding for the three original Knapsack programs. Predicate and
phase reads may share storage, and a guarded constant retains its conditional
relative phase. Acceptance includes full complex128 state vectors against direct
dense-diagonal Circuits, resource budgets, clean workspace and the existing
216-original integration gate. This remains local integration work without a
corpus-wide capability promotion.

The checked qperm slice retains a source proof boundary and reuses existing
register storage, inverse/control composition and arithmetic lowering. Fixed
Output array helpers and singleton multi_suzuki_trotter add three original paths
covering two distinct blobs. Independent full-state references cover positive and
negative molecular evolution times, coherent walk wraparound and zero scratch
leakage. This adds no native qperm API or capability-level promotion and does not
complete the 216-original gate.

Bounded large expressions and shared Boolean phase factors add the unchanged TSP
and facility-location originals to the local UP-06 candidate. Each has two
nonzero parameter vectors checked against independent Circuits derived from
exact classical arithmetic, with absolute phase and all workspace checked.
Shared arithmetic compute/uncompute, six-active-bit Boolean cube recognition
and existing clean MCX kernels provide the implementation; no filename or cost
function special cases are used. Native qfunc arithmetic also uses that Compiler
candidate. Expanded and factor-preserving paths have explicit finite budgets.
This is local CPU correctness evidence, with no corpus-wide maturity promotion.

The large-Pauli UP-06 slice moves the existing physical template into Compiler
behind an independently verified compact register effect. Two unchanged originals
retain 551 terms and 12 target bits; the qDRIFT case retains all 2000 fixed-seed
samples. Source-order products, identity/global phase, coherent controls, finite
budgets and zero workspace are checked against independent references. Data
budgets grow to 512 KiB/65536 tokens while the 4096 owned-node and 100000 final
instruction caps remain. This is a local integration slice, not a complete
Hamiltonian-language, classical-workflow or corpus-wide capability promotion.

The integer-composition UP-06 slice adds exact bounded Qmod classical int/array
bindings and checked static power/if composition. Single-owner Pauli views and
bounded multi-operator first-order products reuse Compiler rotation semantics;
default phase coefficients reuse the real phase path. Three unchanged originals
have independent full-state Circuit checks: Fermi iterations 1/2, two nonzero
credit-card data pairs and original BB84 bits. This is local integration evidence
without a native Python integer-port or corpus-wide maturity promotion.
The checked-Grover UP-06 slice admits one borrowed Const array with diagonal
phase effects, bounded integer specialization and a closed static callback.
Uniform-state reflection reuses existing owned gates and phase lowering,
including its absolute minus sign. Two unchanged originals have six full-state
Circuit and analytic checks (r=0/1/2), bit-order and clean-scratch acceptance,
with alias/escape and budget refusals. This remains local integration evidence,
not a general callback language, native Python Grover export or corpus-wide
maturity promotion.

The predicate-oracle UP-06 slice reuses checked flat record views, bounded
integer predicates, existing conditional phase effects and Grover reflection.
Named/two-argument lambda predicates admit a single distinct XOR marker;
nonzero and coherent marker states, negative differences, global phase and clean
scratch are tested independently. Three unchanged originals are checked, with
graph marking >=4 kept distinct from the earlier <=4 example. The eight-input
predicate and all physical caps remain. This is local integration evidence,
not general callback, nested-record, Python QStruct or corpus-wide promotion.

## Bounded decision predicates

The integration profile retains the eight-bit local phase evaluator. A budget
refusal can use a separate twelve-input decision profile with at most 262144
integer evaluator steps, 4096 reduced decision nodes and 128 disjoint accepting
paths. Eight-bit leaves and shared equal branches produce signed cubes. Groups
of at most six literals reuse existing Boolean compute candidates. Each cube's
flags are uncomputed and released before the next; Compiler clean-tail scheduling
and clean MCX lowering own physical reuse. No new canonical IR or public export
is introduced. Existing degree, indexed-phase width, owned-node and final costs
remain independently bounded. Static eleven/twelve-bit Grover reflections reuse
this path; smaller reflections preserve their previous resource template.

The two unchanged graph-coloring and whitebox originals retain one and ten
rounds. Run tools/validate_qmod_decision_predicates.py with --case
whitebox_fuzzing (or grover_graph_coloring) and --output /tmp/decision.json.
References use an independent gate-level Grover Circuit and analytic amplitudes;
production still executes every original round. The largest validation-only
reference uses its own clean conjunction workspace and no dense operator matrix.
This is local CPU correctness evidence, not a performance or corpus-wide claim.


## Closed quantum callback composition

The private Qmod adapter additionally admits borrowed `qfunc(qbit[])` and
`qfunc(qnum,qbit)` parameters. Closed named functions and lambdas specialize to
existing first-order statements before oracle normalization. Quantum captures,
Output/ownership escape, aliases, wrong signatures, nested callback values and
recursive wrappers refuse. Expansion remains bounded by 4096 statements and
sixteen levels. Existing real callbacks keep their previous path. Only borrowed
Hadamard preparation is admitted in this wrapper within/apply profile; it is
self-inverse and emits preparation, apply, reverse preparation. Callback arrays
and general captured lambdas are not supported.

The unchanged hidden_shift.qmod and deutsch_jozsa.qmod originals have independent
public Circuit and analytic full-state checks, including absolute phase, numeric
bit order and clean workspace. Their numeric predicates reuse the local
eight-input budget, including exact integer min/max; dynamic array widths are
proven by literal allocation. Run tools/validate_qmod_quantum_callbacks.py with
--case hidden_shift (or deutsch_jozsa) and --output /tmp/callback.json. Run
python -m examples.synthesis.algorithms.hidden_shift.quantum_callbacks.main for the original Qmod, existing
native qfunc phase-composition and Circuit comparison. This adds no higher-order
Python qfunc annotations or native phase_oracle export. Evidence is local CPU
correctness, not a performance or full-corpus claim.


## Checked exact amplification and retained work

The private Qmod exact_amplitude_amplification profile accepts one initialized
zero target of one through seven unsigned bits, a closed sign oracle, a closed
product-RY preparation and a positive marked-subspace amplitude. It independently
checks that amplitude against the preparation and predicate mass to absolute
1e-12. It reduces the Grover angle using a retained hidden ancilla and admits at
most 32 rounds. The ancilla and any simultaneous Compiler scratch count toward
max_workspace; its allocation is counted once in total physical width. No early
release or reset is asserted. Existing hidden semantic data keeps its previous
accounting. Owned workspace markers must be exactly True.

inplace_prepare_state accepts normalized product probability vectors with
amplitude L2 factorization defect at most 1e-12. Existing exponential RY angles
are reusable in this checked composition. Nonproduct/complex preparations,
quantum captures, consuming/Output callbacks, non-sign oracles, nonzero or
unproved-zero input, unsupported scopes and over-budget requests refuse.
Unsigned literal += offsets in -256..256 reduce modulo register width and
compose the existing +/-1 kernel; they are not a new optimized adder template.
Zero offsets still check initialization. All physical costs remain budgeted.

The unchanged exact_amplitude_amplification.qmod has independent Circuit and
analytic full-state acceptance. The unchanged partial_exponential_state_preparation.qmod
has a supplied amplitude 0.996625424765961 inconsistent with the approximately
0.9979545784437375 mass of its exp(+x/2), x>=21 preparation. It deliberately
refuses; a clearly labelled corrected diagnostic copy tests that full path,
including x += -18, without increasing original coverage. No original parameter
is silently changed. Run tools/validate_qmod_exact_amplification.py with --case
exact_amplitude_amplification (or partial_exponential_state_preparation) and
--output /tmp/amplification.json. The second choice validates the corrected copy
and the original refusal. Run python -m examples.synthesis.algorithms.amplitude_amplification.exact_amplification.main
for original Qmod, existing native R/phase composition and Circuit comparison.
Native rounds and its auxiliary output are explicit; this adds no native exact
amplification export or native for-loop support. Evidence is local CPU correctness.


## Scalar-controlled Pauli evolution and classical angles

The experimental Qmod boundary supports one scalar unsigned bit (including an
indexed qbit array element) controlling pure Pauli evolution, with literal
zero/one polarity and an optional else branch. Static helpers reuse the same
specialization and existing owned Pauli effects. Identity terms retain their
relative phase. Nested controls, allocation or mixed effects inside these
branches, overlapping control/target storage and signed numeric controls refuse.
Positive Const controls borrow storage; negative/else controls require writable
storage because the existing implementation uses paired X operations. Signed
argument metadata is preserved through bound helper calls.

Finite classical angle expressions support +, -, *, /, pi and bound real inputs.
Matrix, probability and state-preparation literal domains retain their previous
scanner and checks. No quantum-dependent angle is admitted by this profile.
Run python -m examples.synthesis.algorithms.hamiltonian_simulation.controlled_evolution.main for the unchanged BPDE
Qmod, native phase-composition qfunc and independent Circuit full-state comparison
at two nonzero delta values. Native steps are explicit; this adds no native
Hamiltonian-array export or Python for-loop support. The original's dropped
Hamiltonian target is semantic data, not a promised zero auxiliary register.
The separately recorded partial-exponential amplitude mismatch remains refused;
no original corpus source or baseline is rewritten. Evidence is local CPU only.


## Checked swap test and source-data refusal

The experimental Qmod adapter composes swap_test from existing Hadamard and
indexed controlled-X effects. Equal nonempty initialized writable input views
and one fresh scalar qbit Output must have disjoint locations. Existing owned
allocation and Const checks remain authoritative, including scalar Output
forwarding through static helpers. No public native swap_test export is added.

The pinned swap_test.qmod has amplitude squared norms 1.1072 and 1.1076 with
zero preparation error bound, violating the official unit-norm contract. It
remains refused; no automatic normalization or weakened verifier is introduced.
A separately archived, normalized diagnostic copy checks the complete path
against independent Circuit permutation and symmetric/antisymmetric algebra.
It contributes zero original execution coverage. Run tools/validate_qmod_swap_test.py
--output /tmp/swap-test.json. Run python -m examples.synthesis.algorithms.state_preparation.swap_test.main for valid
one-bit Qmod, existing native phase/CX qfunc composition and Circuit comparison.
Dropped input states are retained semantic data, not clean workspace. This is
local CPU correctness evidence only. Existing partial-exponential source-amplitude
refusal and historical baselines remain unchanged.


## Lexical binding-only within/apply

The experimental Qmod adapter supports a within preparation consisting only of
register -> bindings, followed by bounded static gates and specialized gate
helpers in apply. It restores incoming view, width and suspended-name metadata
on scope exit; temporary aliases cannot escape. This is a lexical view scope,
not allocation, numerical inversion or a proof that arbitrary quantum data is
zero. Allocation, initialization, release and other ownership-changing apply
effects refuse after specialization. Existing arithmetic and Hadamard within
paths retain their contracts. Static loops in other quantum scopes are unchanged.

Whole registers and scalar/static indexed locations can concatenate into one
helper array actual, preserving low-to-high source order and existing alias
checks. Bound helpers may declare an unallocated unsized qbit[] alias. All
source initialization, Const mutation, width, loop shadow and resource budgets
remain checked. Run tools/validate_qmod_binding_scopes.py --case 0 --output
/tmp/autoencoder.json (also case 1) for the unchanged quantum_autoencoder.qmod
with two nonzero parameter bindings and independent Circuit full-state comparison.
Run python -m examples.synthesis.applications.machine_learning.autoencoder.main for the original Qmod, existing
scalar native qfunc composition and Circuit comparison. No native array/loop
export or numerical kernel is added. Dropped coded data retains physical state;
it is not zero workspace. Existing source-semantic refusals and baselines remain.


## Checked static amplitude-amplification iterations

The Qmod boundary admits amplitude_amplification(reps,oracle,space_transform,target)
with named single-array callbacks and bounded static repetitions. It composes
existing preparation and Q=-A*S0*A^-1*S_chi without changing the original global
minus sign. Oracle validation covers indexed Z and the checked pi sign-phase
profile, including zero-round and unused definitions. Closed +,-,* index arithmetic
and array length preserve actual bounds; fractional phases, captures, writes and
Output escape refuse. Existing pure phase/pair effects invert in reverse order.
No amplitude parameter or state-normalization repair is inferred.

Untyped array/slice arguments may borrow a flat record layout only at its exact
width; named records remain nominal and signed/fractional casts refuse. A local
flat record retains semantic storage like an existing local array. Binding-only
apply accepts the same checked pure unitary effects; ownership changes refuse.
X apply_to_all and phase-flip views reuse indexed lowering; whole-register routes
remain unchanged. Static loops and physical/effect/helper budgets remain bounded.

Run tools/validate_qmod_iterations.py with --case quantum_counting (also
quantum_counting_iqae or qmci_pi_estimation), --repetitions 0/1/2 and --output
/tmp/iterations.json for unchanged originals against independent Circuit and
analytic full vectors. Run python -m examples.synthesis.algorithms.amplitude_amplification.amplification_iterations.main
for native scalar phase/CX qfunc, Qmod and Circuit comparison. Native rounds are
explicit; no native amplification, array or mutable helper export is added.
Dropped problem data remains semantic, not clean workspace. Actual scratch is
checked separately. Source-semantic refusals and historical baselines are retained;
this is local CPU correctness evidence only.


## Ordinary QPE source integration evidence

Status: development_evidence, exact local CPU complex128. The unchanged
quantum_counting_qpe, qpe_for_matrix and qpe_for_unitary_matrix originals compare
complete complex statevectors against independent matrix powers and an explicit
inverse Fourier transform. Source/license hashes are pinned in
tests/fixtures/classiq_qmod/static_qpe_sources.json; run
tools/validate_qmod_static_qpe.py for individual cases. Dropped problem registers
remain semantic data and workspace is checked separately.

The source slice reuses existing controls, phases, preparation and inverse QFT.
Compiler adds bounded registered-gate matrix powers and controlled Givens
composition. Structural commutation representatives preserve the old partitions
and logical pair counters while avoiding repeated equivalent predicates.
No numerical kernel, public source schema or native array-QPE export is added.
See the [source scope and runnable comparison](../../flagquantum/ecosystem/_qmod/PROFILE_HISTORY.md#bounded-ordinary-qpe-composition)
and [limitations](../reference/KNOWN_LIMITATIONS.md#ordinary-qpe-source-boundary).
Global corpus counts remain separate from these three original acceptance cases.

## Native synthesis Fourier slice

The unreleased Python qfunc entry `qft` reuses checked `qreg.qft` and the shared
Fourier kernel. The four-qubit original and complete two-cell source Notebook
have a native counterpart at `python -m examples.synthesis.functions.qft.main`.
Acceptance compares all basis columns for widths 1..4 and complex coherent
inputs against an independent dense Fourier matrix, including absolute phase,
offset wires and zero workspace. Original/native resources remain 4 qubits,
12 operations and depth 8. The maturity is local CPU semantic verification;
partial views, Const targets and new controlled/inverse helper forms are outside
this admission. See the [contract](../api-changes/FQ-SYNTH-NATIVE-FOURIER-20261109.md).

## Native power-of-two modulo slice

Unreleased Python expression and qfunc capture rewrite unsigned `%` by literal
positive powers of two to checked bitwise AND. All small input pairs, complex
coherent inputs under both schedules, divisor-one encoding and wider masks are
validated independently. The native examples preserve both retained arithmetic
expression paths and the five-bit input domains of the modulo example. General
or quantum divisors remain outside this local CPU profile. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-POWER-TWO-MODULO-20261109.md).

## Native bounded unitary slice

Unreleased Python `unitary(matrix,target)` reuses existing matrix verification,
indexed `qreg.unitary` effects and absolute-phase-preserving gate synthesis.
Original unitary and three-layer quantum-volume programs have native counterparts.
Independent direct matrix Circuits verify retained complex data, complete small
matrix columns, complex coherence, helper composition, offset/sliced wires and
consistent least-significant-first array/list views. Host constructors, mutable
global matrices and unsupported ownership fail closed. The maturity is local CPU
semantic validation; complete application Notebooks require separate acceptance.
See the [contract](../api-changes/FQ-SYNTH-NATIVE-UNITARY-20261109.md).

## Native unsigned conditional values

Typed qfunc fresh assignment reuses scalar `qint.select` for Python conditional
values. The unchanged minimum original has a native counterpart; independent
reversible truth tables verify ties, nested values, wider branches, complex
coherent states under both objectives, helper composition and clean workspace.
Ownership and resource ceilings remain checked. Maturity is local CPU semantic
validation, with no signed/fractional or dynamic host-branch claim. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-CONDITIONAL-VALUES-20261109.md).

## Native bounded modular powers

Native `modular_exp` reuses checked register permutations and scratch scheduling.
The retained example and full local Notebook sampling/assertion workflow have
native counterparts. Independent permutation matrices cover all small columns,
padding states, nontrivial coherent phases, helper composition and width bounds.
Maturity is local CPU semantic validation with resource refusal checks; this does
not establish scalable Shor performance or hardware execution. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-MODULAR-EXP-20261109.md).


## Native bounded real and predicate-gated phases

Unreleased native Python phase arithmetic and phase-only control reuse the
Compiler-owned Boolean-polynomial expansion also used by the Qmod adapter.
Sparse unsigned-word and scalar-array reads may share physical bits; constant
and conditional constant phases remain absolute. Writable helpers preserve
borrowing rules and their existing adjoint/control limits. Bounds are 24 total
input bits, 8 predicate bits, degree 8, 256 monomials, 20,000 products and 128 native
expression depth. These are local CPU semantic checks, not hardware performance
or general native qstruct/signed/fractional support. Independent Circuit states
cover complete ADAPT-QAOA, MaxCut and constrained Knapsack programs; full Notebook
optimizer workflows require separate acceptance. See the
[phase contract](../api-changes/FQ-SYNTH-NATIVE-POLYNOMIAL-PHASE-20261109.md) and
[guarded phase contract](../api-changes/FQ-SYNTH-NATIVE-GUARDED-PHASE-20261109.md).

## Native checked unitary region slice

Maturity: `development_evidence`. Native adjoint and pure-gate within/apply reuse
shared Compiler inverse rules on initialized storage, including static writable
helpers, QFT, explicit matrices, Pauli rotations and constant global phases.
Retained QCE, credit-card and randomized-benchmarking programs have independent
absolute-amplitude Circuit checks in
`tests/integration/test_synthesis_native_unitary_regions.py`. Existing arithmetic
cleanup and Const-only inverse regressions remain required. Closed generated
predicate scratch now composes under inverse and within/apply, with fresh local
identities and verified zero cleanup; direct QFT also accepts bounded composite
comparison predicates. The native Fourier-LCU example has independent complete
state checks, two nonzero weight/data models and actual workspace-budget refusal
in `tests/integration/test_synthesis_native_fourier_lcu.py`; the source and native
resource gap remains explicit. User allocation, Output publication, general
callbacks and ownership changes refuse. This does not
promote runtime gradients, hardware, scalability or complete Notebook coverage.

## Native predicate-controlled walk slice

Maturity: `development_evidence`. Bounded comparison/composite predicate-X reuses
Compiler graph capture, indexed control-X and direct compute/uncompute cleanup.
Borrowed numeric input resolution uses complete raw registers and actual widths;
root and formal names need not match. Native hypercube and qfold programs have
unchanged-source and independent Circuit amplitude checks in
`tests/integration/test_synthesis_native_predicate_walks.py`; branch XOR,
coherence, borrowed roles, bounds and inverse-region safety are covered by
`tests/integration/test_synthesis_native_predicate_x.py`. Actual width, work,
operations, depth and memory constraints remain authoritative. In-place increments,
free/reset, arbitrary higher-order callbacks, full Notebook workflows, hardware
and resource-optimality claims remain outside this slice.

## Borrowed native expression XOR

Maturity: `development_evidence`. Static all-inout helpers reuse `qreg.xor`
for complete raw unsigned scalar destinations and expressions with complete
register inputs. Active formal roles and aliases survive nested helper capture;
nonzero XOR targets, phase-weighted coherent inputs, disjointness and actual
resource limits are covered in
`tests/integration/test_synthesis_native_expression_xor.py`. Partial/permuted
views, array writes, formatted arithmetic and mutable-helper control/inverse
remain explicit gaps. No additional native corpus coverage is credited merely
for this primitive.

## Native modular walk updates

Maturity: `development_evidence`. Native bounded static modular increments and
decrements reuse verified gate/control-X effects on complete raw unsigned words.
Single-update helper controls reuse bounded predicate scratch and cleanup.
`tests/integration/test_synthesis_native_modular_increment.py` verifies complete
basis domains, coherent controls, absolute phases and refusal boundaries.
`tests/integration/test_synthesis_native_grid_walk.py` compares the unchanged
two-dimensional walk, native program and independent Circuit at full original
widths with retained coin environment and zero compiler work. Resource costs are
recorded separately; no optimality, complete Notebook or QLBM acceptance follows
from this slice alone.

## Native exact interval MPS admission

Local CPU semantic coverage includes native qfunc programs wider than 32 bits
whose connected physical gate intervals are each at most twelve sites. Services
reuses Runtime's existing exact bond proof and shared planning authority. Four
admission tests verify complete occupied-support complex amplitudes and unit
support probability on an 83-bit program, no truncation or dense conversion,
insufficient-memory and unproved-long-interval refusal, and unchanged small
statevector routing. Sixteen basis-MPS and four original zero-walk regressions
also pass. This is local exact execution evidence, not application cancellation,
Notebook completion, accelerator or distributed scalability evidence.

Run `python -m pytest tests/integration/test_synthesis_native_interval_mps.py tests/integration/test_synthesis_native_basis_mps.py tests/integration/test_synthesis_qmod_tensor_zero_walk.py -q`.
