# JAX representation conversion and shard-gate ownership

Status: prepared against `32f1d3fff711966acd72454ba94ee7ffeb01288c`.
Implemented in the actual Git checkout after exact-parent CI qualification.
Formal source regressions, quality checks and final installed-artifact acceptance
passed. New exact published-head CI remains pending at this publication snapshot.

## Problem and final boundary

The shared JAX conversion module contains a complete in-process statevector
gate algorithm alongside MPS-only evidence assembly and tensor-network-only
conversion. These responsibilities already have concrete owners. This change
uses those existing modules without adding a package, facade, registry or record.

| Previous location | Owner after migration |
| --- | --- |
| `array_conversions._apply_gate_to_jax_shards` numerical body | `simulation.statevector.jax.kernels.jax_apply_gate_to_shard_arrays` |
| Statevector gate readiness, shape admission and record reconstruction | `runtime.executors.jax.statevector.execution._apply_gate_to_jax_shards` |
| `array_conversions._jax_split_record` | `runtime.executors.jax.mps.execution._jax_split_record` |
| `array_conversions._jax_nodes_from_torch_nodes` | `runtime.executors.jax.tensor_network.execution._jax_nodes_from_torch_nodes` |
| Gate and parameter conversions actually shared by representations | Remain in `array_conversions` |

Runtime retains plans, device placement, collective selection, gate counts,
communication accounting, claimability and the original shard record types.
The numerical function receives array and index sequences, the admitted gate,
wire order and qubit count. It performs basis grouping, diagonal factors,
matrix multiplication and functional amplitude updates. It neither imports
Runtime nor performs communication. In-process execution retains its existing
development classification and makes no capacity claim.

The statevector adapter performs real admission and metadata reconstruction;
it is not a historical forwarding entry. All consumers of the three previous
shared names use their representation owner directly. The shared module retains
none of those names or the obsolete basis-helper import. Public exports,
dataclass locations, schemas, signatures and native ABI remain unchanged.

MPS circuit lowering is outside this closure. Its combined IR interpretation,
gate admission, fusion and lowering must be reviewed with Construction/IR.
No isolated matrix operation is extracted from that path here.

## Independence from PR 1084

The full, nontruncated incoming tree at
`3dc30804b66df19599dc5c69467148349939742f` and actual common ancestor
`e48251b01754cecd783ddf0206e24f75583f7434` have identical blobs for all five
existing Runtime modules in this closure. The Simulation destination is absent
in both upstream trees and already exists on current main. Ordinary three-way
integration therefore retains the current-side migration; this statement does
not treat the incoming standalone old source as an integrated result.

The unmodified changed-incoming Python set has exactly the same 17 raw retired
import diagnoses under the current 586-row and proposed 589-row ledgers.
Nothing is filtered, and those existing G0 obligations remain open. A later
incoming head and the final integrated tree still require verification.

## Original evidence and preview acceptance

Original evidence was generated from the unchanged actual Git checkout before
preparation, using the existing functions and real optional JAX installation.
The original producers and output files are retained separately and are not
regenerated from the proposed implementation.

- 672 original gate trajectories cover one through four qubits, batch sizes
  one and two, complex64/complex128, contiguous/interleaved/reversed shards,
  single-wire and ordered two-wire gates, diagonal and dense updates, complete
  and incomplete coverage, gradients, input preservation and metadata identity.
  All output bytes match the preview. Original SHA-256:
  `fb3ac5c0eeac4d940b5896011963189f64f40cdf2a24c838cd7dc56239e695b7`.
- 39 original MPS split-record cases and 24 Torch-to-JAX tensor-node cases
  preserve result values, refusal behavior, dtype, shape, labels, names,
  metadata identity, input gradient ownership and empty conversion. All output
  bytes match. Original SHA-256:
  `ac07c282ffac18479fc090dfceb002e5269916ce0f721c14f822a19b41b8d097`.
- All 32 pre-existing definitions remaining in the six edited modules have
  unchanged ASTs. The MPS evidence and tensor-node conversion bodies move with
  exactly identical ASTs. Existing gate arithmetic order is retained and
  checked against the independently captured original results.
- The focused preview passes 135 tests, with six existing hardware skips.
  All 329 loaded product modules come from the proposed source export. This
  is a bounded focused run, not a full regression from a non-Git export.
- Dense forward and analytic-gradient references, missing-coverage refusal,
  shape/empty ordering, metadata identity and a replacement implementation
  exercise the actual Runtime consumer. The existing JAX boundary assertion
  follows the new canonical numerical module; no compatibility import is kept
  solely to satisfy the previous binding assertion.

Preparation failures remain recorded: the initial sparse proposal stopped at
an assertion about an existing tensor-network import and was archived; an
unconfigured Ruff invocation used rules outside the repository configuration;
the first new fixture used an unavailable experimental JAX context manager and
was corrected to restore the established configuration in `finally`; one
preview test invocation ran from the original checkout and consequently loaded
the unmigrated implementation. None of those runs qualifies the candidate.
The final focused runner verifies every loaded product module's source.

## Required formal acceptance

Before activation, the exact parent PR must finish its own CI qualification.
Then apply the complete listed closure in the actual Git checkout, preserve
original fixtures and frozen contracts, and run the required source regressions,
quality checks, cold imports, source relocation guards and public workflows.
Build a new native wheel, compare its product bytes to committed source, install
it separately and repeat the relevant original-reader and JAX workflows.
Publish only after that acceptance and monitor the new exact-head CI. No
preview result qualifies an untested published commit.


## Actual-source qualification

The combined change also includes the user-requested
[CPU/CI critical-path repair](ci-partition-critical-path.md).
The actual Git checkout passed Unit/API/Team **19,516 passed / 972 skipped**,
QEC **2,801 / 329**, root/local integration **1,592 / 223**, and complete
benchmark contracts **1,096 / 50**. Real optional JAX passed **135 / 6**;
all six skips are existing hardware conditions. All eleven quality checks,
strict Mypy over 1,265 product Python files and four supplementary gates pass.

The original first formal attempt is retained: fourteen failures came from
one existing module-size ceiling, interpolated job names rejected by the
unchanged required-check policy, and stale capacity assertions. The private
kernel explanation moved to its existing README, automatic job names retained
the required-check rules, and capacity tests now cover the authorized fan-out.
All 187 corrective focused scenarios pass. No failed test or diagnostic was
filtered. New per-node timing telemetry also preserves intentionally failing
and skipped subprocess outcomes in both serial and xdist scenarios.

The final native wheel has SHA-256
`114b07f5a99eb97293af843608d76da91017584640029f6b19a47494533210b1`.
All 1,265 product Python files match the actual checkout, wheel and separate
installation. All 735 original outcomes remain byte-identical. Installed
JAX/CPU scenarios pass 197/6, and unchanged public/compatibility scenarios pass
724/7, with one source-only AST case covered in the full actual-source suite.
All loaded product modules resolve to the wheel installation. Both native
extensions load as binaries; cold placement/catalog imports and the benchmark
CLI pass. Dependency resolution is not certified by these borrowed dependencies.

The latest incoming increment at
`a73741a3360cac24f6ffc4f02b024fca0415d981` contains twelve SHA-verified paths
since 49f16fbb30 and has no intersection with this complete thirty-path change.
The parent 586-row and active 589-row ledgers retain all seventeen unfiltered
changed-incoming import diagnoses. This does not close the integrated G0 audit.
The new exact-head CI remains independently required; parent CI and local
acceptance do not qualify an unpublished commit.


The first published CI also found a QEC thread-context regression in the new
exclusive Runtime allocation. The critical-path document records the correction:
QEC retains its original one-thread scientific contract and restores other
suites' admitted allocation after each case. No product mathematical source,
public contract, scientific assertion or wheel content changes for that repair.
The final wheel remains bound to all current product bytes; source and CI
qualification are tracked separately for each published commit.
