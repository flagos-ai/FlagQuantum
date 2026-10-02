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
therefore walks `artifacts/`, `benchmarks/results/` and
`docs/development/evidence/` rather than a list, and asks both questions a reader
depends on. A revision is obtainable when this repository holds the commit and a ref
of this repository reaches it, because a clone and a plain fetch obtain exactly the
commits the refs they fetch reach; of the sixty-two distinct revisions recorded
under those roots, one is obtainable. Every other revision must be declared in
`evidence-revision-origins.toml`, which names the origin a reader can obtain it
from. Asking only whether the object was in the database was not enough:
thirty-nine revisions are unreachable from every ref, so a clone obtains none of
them, and thirty of the thirty-nine are absent from this checkout's object database as
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

The gate walks JSON artifacts under exactly those three roots, which is a boundary
rather than a claim of full coverage, and what falls outside it is measured rather
than assumed. Counting a pin as any full-length hexadecimal value, fifty-nine
(file, revision) pairs across sixteen files record a revision that no artifact under
those roots records: fifty-five of the pairs are under `docs/`, two under
`examples/`, one in `.github/`, and one under `benchmarks/` in prose the JSON-only
walk cannot see. Those pairs name fifty-four distinct revisions, of which fifty-one
resolve in this checkout and three do not.

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
`/commits/<sha>`, each asked twice. That endpoint answers 200 for all thirty-nine
`unreferenced_object` revisions, 404 for all eighteen `producing_host_history` ones
and 404 for all four `external_dependency` ones, while also answering 200 for a commit
on `main`, 404 for a commit of `pytorch/pytorch` and 404 for a commit that exists only
in the checkout that produced it -- so the endpoint reads this repository's object
database, not GitHub's, and the thirty-nine it holds are exactly the thirty-nine no ref
reaches. It is reproducible with `gh api` by anyone who reads the table, which the
fetch count is not.

What remains outside the walk is three revisions, and none is a pin a reader is
asked to check out. `.github/workflows/ci.yml` compares a base revision against the
git zero-SHA placeholder, so the all-zero value is a shell literal rather than a
revision. `examples/assets/manifest.json` records a revision of
`google-bert/bert-base-uncased` in `assets[2].revision` and `assets[3].revision` --
the value quoted as `BERT_TOKENIZER_REVISION` in `examples/quantum_transformer.py` --
and no FlagQuantum repository contains it or should. The third,
`aaa3a70c698dc629c37baa2719607444aa3b6381`, is named in prose by
`benchmarks/results/comparison/FLAGQUANTUM_QISKIT_AER_CPU_ARM64_20260923.md`; the
remote serves it, so it belongs to the unreferenced-object class declared above, and
it stays outside because the walk reads JSON artifacts and not the markdown citing
them, which is a limit of the walk rather than a statement about the revision.

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
