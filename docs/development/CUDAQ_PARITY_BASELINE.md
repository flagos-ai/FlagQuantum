# CUDA-Q parity baseline

Status: recorded baseline survey. This document does not claim parity; it records
the external capability set that `contracts/cudaq-parity-matrix.toml` compares
against.

Owner: Integration

Recorded: 2026-10-06

Pinned baseline versions: `0.15.1`, `0.16.0.post1`

Pinned component: CUDA-Q core. One matrix domain is read against a component that
ships on its own release line; that separation is recorded under
[component provenance](#component-provenance).

## What this document is

`contracts/cudaq-parity-matrix.toml` holds one row per CUDA-Q capability and the
FlagQuantum verdict for it. This document records where the CUDA-Q side of those
rows came from, so a reviewer can tell a measured fact from a survey entry.

The pinned versions are the ones the repository already targets in
`contracts/cudaq-export-contract.toml`. A CUDA-Q release outside that range
requires a new capture and a new row review.

## Capture method

Capture method: `documented_capability_survey_with_probes`.

The CUDA-Q capability set was read from the published CUDA-Q documentation, the
public Python API surface, the documented backend roster, and the shipped
repository tree of the pinned versions.

Two repository probes back the parts of the baseline that a document cannot
settle, and `contracts/cudaq-parity-matrix.toml` registers both under
`[[baseline.probes]]`:

| Probe | What it establishes |
| --- | --- |
| `benchmarks/cudaq_gradient_capability_probe.py` | Which gradient strategies the installed CUDA-Q exposes, by runtime public-symbol inspection rather than by reading documentation. |
| `benchmarks/cudaq_backend_compare.py` | Whether one FlagQuantum JAX kernel and one CUDA-Q target agree numerically on a matched circuit, observable, dtype, gradient method, warmup, and iteration count, and how long each took. |

They now carry the same status, because both stand on landed evidence.

`benchmarks/cudaq_gradient_capability_probe.py` is `status = "measured"`. Its
payload is
`benchmarks/results/comparison/cudaq_gradient_capability_a800_20260930.json`. The
survey ran three times on each of two hosts on 2026-09-30 and returned a
byte-identical capability record every time, so what it establishes is a property
of the CUDA-Q release rather than of one installed wheel. Those hosts reached
CUDA-Q through different binary distributions — `cuda-quantum-cu12` on
`jp-a800-172` and `cuda-quantum-cu13` on `jp-a800-171` — which is the part that
makes the agreement informative.

`benchmarks/cudaq_backend_compare.py` is `status = "measured"`, which means a
payload from a named machine is landed under `benchmarks/results/` with its
source revision, container image, environment, and methodology recorded next to
it, and `benchmarks/audit_results.py` accepts it. That payload is
`benchmarks/results/comparison/cudaq_backend_compare_a800_20260930.json`, read in
`benchmarks/results/comparison/CUDAQ_BACKEND_COMPARE_A800_20260930.md`. It was
captured by hand on 2026-09-30 on both `jp-a800-172` and `jp-a800-171` against
CUDA-Q 0.16.0.post1, from two different base images. Every FlagQuantum figure in it
agreed across the two hosts to within 10 percent.

Two of its findings are recorded here because they bound how the payload may be
cited. The forward-only gap is warmup-insensitive: the 22-wire case moves 0.1
percent between one and three warmups, so the conclusion that CUDA-Q is faster at
every width is a property of the implementations. The gradient ratio is not: the
18-wire case moves monotonically from 13.6x to 16.3x between one and five warmups
and has not converged. An earlier capture reported 7.9x for that configuration
and is withdrawn rather than reproduced. A gradient ratio from this protocol is
therefore quoted with its warmup count, and the payload carries the sweep under
`protocol_sensitivity`.

No CI lane in this repository installs CUDA-Q, so a payload is only ever
captured by hand: the PyPI `cudaq` release is a source distribution whose build
hook resolves a platform binary wheel at install time, and a lane cannot install
the pinned version from a fixed index without a CUDA runtime present. That is a
property of how CUDA-Q ships, not a choice this repository made.

Consequences of that method, stated so they are not mistaken for strengths:

- A capability absent from the matrix is not evidence that CUDA-Q lacks it. It is
  evidence that this survey did not record it. An unrecorded capability is a
  documentation defect in this baseline, not a FlagQuantum claim.
- A capability listed as present is listed because it is documented or shipped in
  the pinned versions, not because it was reproduced here.
- Backend behaviour under a workload, rather than the existence of the backend, is
  out of scope for this survey. That is the part the probes measure, and one
  workload on two machines is now recorded. A recorded payload is still not a
  survey finding: it describes the machines, revision, and workload it names, and
  no row of the matrix is restated from it.
- Both probes measure a single process on a single device. Nothing in this baseline
  observes how either framework distributes a workload, so no entry here carries a
  capacity or multi-device meaning.

## Scope

In scope: capabilities observable from the public Python API, the documented
backend and provider roster, and the shipped repository tree of the pinned
versions, organized into the twelve domains declared by `domain_order` in
`contracts/cudaq-parity-matrix.toml`.

Out of scope:

- performance and scale comparisons, which require an audited benchmark payload
  in `benchmarks/results/`;
- licensing or commercial terms of the CUDA-Q distribution;
- the internal implementation of any CUDA-Q pass, kernel, or backend beyond what
  determines whether a capability exists and which dependency it rests on;
- CUDA-Q developer tooling that has no FlagQuantum counterpart and no product
  consequence, such as its own test harness layout.

## Component provenance

CUDA-Q is one product with more than one release line, and the `0.15.1` /
`0.16.0.post1` pin reaches only the core. The `quantum_error_correction` domain is
read against CUDA-Q QEC (`cudaq-qec`, also shipped as `libs/qec` inside
`NVIDIA/cudaqx`), which releases independently: its release line runs from
`0.1.0` to `0.8.0`, and `0.8.0` is the version every QEC row was read at.
`contracts/cudaq-parity-matrix.toml` records this under
`[[baseline.version_provenance]]`, including the release index the line was read
from.

The two lines are disjoint. No `cudaq-qec` release maps to either pinned core
version, so nothing here establishes that a QEC row describes the same surface at
the pinned baseline. A QEC row must therefore not be read as a statement about
CUDA-Q core `0.15.1` or `0.16.0.post1`; a row that needs to be is a new capture.
This is a limit of the survey, not a defect in the component.

Two of the surface items the QEC domain lists belong to a component other than
`cudaq-qec`, and one more is worth attributing. A `cudaq_surface` list is an
inventory of the surface a user meets, not an attribution, so the attribution
lives in `baseline.surface_ownership` in the contract, where the tool checks each
entry against the inventory it claims to explain:

| Surface item | Owner | What it means |
| --- | --- | --- |
| `dem, dem_from_kernel, and DEMResult` | CUDA-Q core | One item, two owners. `dem_from_kernel` and `DEMResult` are core and are read back by core; CUDA-Q QEC's own construction entries are `dem_from_stim_text`, `dem_from_memory_circuit` with its `x_` and `z_` variants, and `dem_from_css_matrices`. |
| `extract_syndrome` | CUDA-Q Logical preview | Not a CUDA-Q QEC surface. CUDA-Q QEC's own extraction route is `sample_memory_circuit` with its `x_` and `z_` variants, feeding `decoder_context_from_memory_circuit`. |
| `for-stim-users migration` | CUDA-Q Logical preview | CUDA-Q QEC publishes no such page. The only page with that title belongs to the preview layer, which states that it does not simulate, sample, or decode and emits no detector error models. |

The consequence for this baseline is that a QEC row is read against CUDA-Q QEC's
own surface, and that the two preview-layer items are attributed rather than
adopted: the logical layer is a separate target with a different deliverable, and
it produces no detector error model for a QEC row to align against.

## The dependency question

The reason this baseline is recorded at all is that a CUDA-Q capability is not
one thing to replace. Each row declares a `dependency_class`, and that class
decides what closing the gap actually costs.

| Class | Meaning | Consequence for FlagQuantum |
| --- | --- | --- |
| `A_nvidia_proprietary` | The CUDA-Q implementation rests on an NVIDIA-proprietary component. | This is the replacement battlefield. FlagQuantum must supply its own kernel or a FlagOS-family component. Depending on the original is not an option. |
| `B_open_neutral` | The CUDA-Q implementation uses a component that is permissively licensed and vendor-neutral. | FlagQuantum may use the same component directly. The remaining work is integration, conformance, and ownership rather than research. |
| `C_flagos_replacement` | FlagQuantum satisfies the capability through a FlagOS-family component. | The replacement target is named. The gap is maturity and evidence, not existence. |
| `none` | The capability is not dependency-bearing. | The work is engineering, not replacement. |

The framework layer of CUDA-Q does not itself depend on an NVIDIA component: its
CPU statevector, density-matrix, and stabilizer backends run without one. That is
why this baseline separates the two questions it is often collapsed into. A gap in
the compiler or the language model is not a gap in the numeric cores, and closing
one does not close the other.

## How the baseline is consumed

`tools/parity_matrix.py` validates `contracts/cudaq-parity-matrix.toml` against
this document's existence, against `capability-maturity.toml`, and against the
repository, then renders `docs/reference/CUDAQ_PARITY_MATRIX.md`. The generated
document is the published scoreboard.

The validator enforces:

- every declared domain appears in `domain_order`, and every status, priority, and
  dependency class is one of the declared values;
- every `maturity_ref` names an entry that exists in `capability-maturity.toml`,
  and a `supported` row has one; a row whose implementation spans two registry
  entries names both through `maturity_refs` and uses one form or the other;
- every row carries evidence, taken from the row or from its domain default, and
  every evidence item is either an existing repository path or a `search:` token
  recording the negative search that established an absence;
- every `surface_ownership` entry matches exactly one domain surface item
  verbatim, names a declared owner, and states the consequence for the row it
  feeds;
- the component release line is recorded, the version the QEC rows were read at is
  on it, and a version mapping is present exactly when the two release lines have
  a version in common;
- no row asserts scalability.

A scale statement in the matrix stays out of scope even after a payload lands:
the matrix records capability presence and absence, and no row in it is a
performance or capacity claim. The landed payload is registered in the contract
because it is the evidence behind the probe's own status, not because it promotes
any row.

## Relationship to the existing capability index

`docs/reference/FEATURE_PARITY_MATRIX.md` is an index of FlagQuantum's own
authoritative sources, and it states that planned capabilities do not appear as
supported rows until executable manifests and tests exist. This baseline does not
change that rule. Because every `supported` row in
`contracts/cudaq-parity-matrix.toml` carries a `maturity_ref`, every such row
already has a registered, executable FlagQuantum capability behind it. A row
whose implementation spans two registry entries names both through
`maturity_refs`, and the tool checks every entry it names.

## Maintenance

Refreshing this baseline is required when any of the following happens:

- the pinned CUDA-Q version range changes;
- the CUDA-Q QEC release line moves, because that re-opens the provenance of every
  `quantum_error_correction` row;
- a CUDA-Q capability that the matrix does not record becomes relevant to a
  planned FlagQuantum feature;
- a `maturity_ref` or `maturity_refs` target changes level, is renamed, or is
  removed.
- the surface ownership of a `cudaq_surface` item changes, because that is what
  decides which component a row is read against.

The refresh updates both this document and `contracts/cudaq-parity-matrix.toml`
in one change, and the generated document is regenerated from the contract. A
hand edit to `docs/reference/CUDAQ_PARITY_MATRIX.md` is a defect.
