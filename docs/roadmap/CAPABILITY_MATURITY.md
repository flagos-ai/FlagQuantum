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
therefore walks `artifacts/` and `benchmarks/results/` rather than a list, and asks
both questions a reader depends on. A revision is obtainable when this repository
holds the commit and a ref of this repository reaches it, because a clone and a
plain fetch obtain exactly the commits the refs they fetch reach; of the forty-one
distinct revisions recorded under those roots, one is obtainable. Every other
revision must be declared in `evidence-revision-origins.toml`, which names the
origin a reader can obtain it from. Asking only whether the object was in the
database was not enough: eighteen revisions are held by this repository's remote
while no branch and no tag reaches them, so a clone obtains none of them, and
`unreferenced_object` covers those, since the remote still serves the object by
name while nothing guarantees that it will. `producing_host_history` covers the
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
