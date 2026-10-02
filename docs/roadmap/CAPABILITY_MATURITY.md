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
commits the refs they fetch reach; of the forty-eight distinct revisions recorded
under those roots, one is obtainable. Every other revision must be declared in
`evidence-revision-origins.toml`, which names the origin a reader can obtain it
from. Asking only whether the object was in the database was not enough:
twenty-five revisions are unreachable from every ref, so a clone obtains none of
them, and twenty of the twenty-five are absent from this checkout's object database as
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
six artifacts under `artifacts/` do; both records are read, and a revision whose two
records disagree fails the gate rather than passing twice.

The gate walks JSON artifacts under exactly those three roots, which is a boundary
rather than a claim of full coverage, and what falls outside it is measured rather
than assumed. Counting a pin as any full-length hexadecimal value, fifty-nine
(file, revision) pairs across sixteen files record a revision that no artifact under
those roots records: fifty-five of the pairs are under `docs/`, two under
`examples/`, one in `.github/`, and one under `benchmarks/` in prose the JSON-only
walk cannot see. Those pairs name fifty-four distinct revisions, of which fifty-one
resolve in this checkout and three do not.

Adding `docs/development/evidence/` accounted for the largest group of those pairs
rather than only reporting it. Five revisions are recorded by the four Jiuding
records under that root that write the revision out in full, and all five are
unreferenced objects of this repository's remote, so they are declared in
`evidence-revision-origins.toml`, whose unreferenced-object class the change
also states one entry larger. Those records are what `docs/guides/JIUDING.md`, the
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
969, which is mostly digests, task identifiers and container identifiers; exactly two
of them prefix a revision this repository records in full. Both are the
`source_revision` / `source_revision_full` pair in
`benchmarks/results/comparison/cudaq_backend_compare_a800_20260930.json` and its
gradient counterpart, where the file states the full value beside the recorded
abbreviation and the check therefore accepts it rather than asking that recorded
output be edited. Fourteen of the 969 sit in a field named for a revision --
`base_commit`, `host_commit`, `implementation_commit`, `source_commit`,
`source_revision`, `validation_driver_commit` -- and nothing in this repository
expands any of them, so neither half of the gate asks about them: the revision walk
needs forty characters, and the citation rule needs a value to expand against. That
residue is the reason the declared-field work is the next change rather than an
optional one, and it is reported here rather than repaired, because completing those
values means reading the full revision out of a clone that holds the commit rather
than guessing it from the abbreviation.

`examples/` is why widening the roots is a decision rather than a one-line change.
The walk is key-agnostic by design: it treats any
full-length hexadecimal value at any depth as a revision, which is how it finds the
field names this repository uses without maintaining a list of them. Walking
`examples/` would therefore demand a FlagQuantum origin for a third-party dataset
revision, reporting a defect where the tree is correct. Narrowing the rule to a
declared set of revision-bearing field names, or declaring a root's revision fields
with it, is the precondition for walking any root that mixes FlagQuantum pins with
revisions of another system. `ci/` is left out for a different reason and does not
wait on that work: `ci/flagos_cuda_reference.lock.json` records
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
