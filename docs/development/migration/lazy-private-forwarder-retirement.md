# Runtime watchdog ownership and private forwarder retirement

Preparation parent: `3316375517aaea6826d01bcccc46afb911f48da1`, PR #1341.
Status: complete source and installed-artifact acceptance passed. The parent
PR #1341 is qualified at the exact preparation commit. The new published-head
CI remains required; no pending CI result is certified.

## Complete private closure

Three Runtime Audit functions import and call existing Release Policy helpers
without adapting arguments. Delete those wrappers and import the same typed canonical
helpers inside the invoking functions. Calls retain their original conditional
positions, and module-level imports do not introduce a circular dependency. Policy implementations, criteria, refusal
messages, schemas, checks, warnings and evidence fields stay with their existing
owners. The differently defined classifier in `validation_helpers.py` has a
separate purpose and is not unified with Release Policy's classifier.

`MPSState._sample_indices` is a fourth private forwarding method. Public `sample`
keeps its request validation and imports/calls the existing numerical owner only
inside the original index-format branch. Bits-format sampling is untouched.
The unsupported-format test patches that numerical owner directly; its refusal
and all original assertions remain. No sampling math or RNG policy moves.

The Audit Engine's generic module `__getattr__` currently reaches arbitrary
private attributes in two other modules. Restrict it to the five existing lazy
public names below. Keep its existing public lookup order and exception behavior;
there is no new module-alias injection, private-helper fallback or package.
The prepared AST proof preserves all original definitions and assertions after
exact boundary edits are normalized. Four checks extend the existing public
boundary test so deleted helpers cannot reappear through generic lookup or
class attributes; no test node or scientific assertion is removed.
The proposed ledger has 583 rows. The historical 102-entry facade inventory stays
unchanged; the newly converted existing watchdog public module is itemized below.

## Obsolete direct-file-loader imports

Remove seven ImportError fallbacks in Engine, Release Policy and MPS Readiness
that duplicate the same canonical relative imports under absolute names. The
normal package branch and binding names remain. Four noncyclic schema/helper
imports move to the ordinary import section; only the actual policy cycle uses
function-local imports. Current
benchmark audit consumers import the canonical package; the former non-package
file-loader path is not a published API or serialization locator. This removes
an internal transition without suppressing a real import failure or changing
policy algorithms. Standalone benchmark scripts keep their existing package
bootstrap and public audit exports.

## Public lazy exports retained individually

The original Engine `__all__` publishes 24 names. Its actual parent installation
provides these five through lazy lookup; preserve their original objects and
qualified owners, along with all nineteen already bound names.

| Existing public name | Defining owner | Concrete basis |
| --- | --- | --- |
| `attach_sharded_optimizer_step_evidence` | `runtime/audit/release_policy.py` | Existing Engine `__all__` and original public-object pickle |
| `audit_distributed_scalability` | `runtime/audit/release_policy.py` | Existing Engine `__all__`, benchmark audit consumer and original public-object pickle |
| `evaluate_mps_backward_readiness` | `runtime/audit/mps_readiness.py` | Existing Engine `__all__`, runtime audit tests and original public-object pickle |
| `require_distributed_scalability` | `runtime/audit/release_policy.py` | Existing Engine `__all__` and original public-object pickle |
| `validate_distributed_claim_evidence` | `runtime/audit/release_policy.py` | Existing Engine `__all__`, benchmark audit consumer and original public-object pickle |

Runtime and Integration own this maintained export boundary. Removing a published
name requires an explicit versioned public API/payload migration. Restricting
unpublished generic fallback does not create a new compatibility guarantee.
The original 24-object reference pickle SHA-256 is
`4df49a0b1d349ae42aeef7189fd82da9cf437f9fb651f5ff5f6243ff1985c23f`.
Its ordered public-name/value snapshot is retained independently and restores
against the independently installed native artifact.

## Other lazy-call candidates retained for concrete responsibilities

Six Compiler lazy callbacks occur in real public `PassSpec` payloads returned by
`default_pass_registry().spec(...)`. Their original pickle SHA-256 is
`ec98b82c94101c6ba894adfc9366845f1400a7552b338d9b65db17d5e60720eb`.
The parent and current installations restore all six identities, uses/preserves
fields and original pickle bytes. These are existing public serialized bindings,
not newly invented private compatibility. Their owner is Compiler/Integration;
removal requires a versioned payload migration and an old-payload reader.

| Existing callback in `compiler/optimization/pass_registry.py` | Public spec name |
| --- | --- |
| `_remove_diagonal_gates_before_measure` | `remove_diagonal_gates_before_measure` |
| `_merge_inverse_pairs` | `merge_inverse_pairs` |
| `_cancel_commuting_self_inverse` | `cancel_commuting_self_inverse` |
| `_merge_commuting_rotations` | `merge_commuting_rotations` |
| `_collapse_one_qubit_runs` | `collapse_one_qubit_runs` |
| `_collapse_two_qubit_blocks` | `collapse_two_qubit_blocks` |

The actual `AnalysisSpec` callback is already itemized in the preceding
[direct-call record](private-direct-call-retirement.md). The full registry's
existing pickle refusal is not treated as a promise about standalone public
spec objects.

The one-call NumPy cosine function in Simulation MPS pullbacks is a real
numerical kernel at its canonical owner. The embedded sampler constructor is a
typed backend construction boundary beside the SDK constructor, with an existing
rejection-before-engine-preparation replacement test. Neither is historical
compatibility forwarding. The twelve-entry local-import scan is bounded and
does not certify every method or all architecture ownership.

## Verification scope and remaining integration

No frozen public API, serialization schema, scientific contract, fixture,
numerical threshold, workload, sample count, dependency or native ABI change is
proposed. Original policy and numerical authority bodies must remain unchanged.
Full source checks, public-payload restoration, metadata and MPS RNG/sample/
refusal comparison, installed cold import and native probes passed as recorded
below. The exact new published-head CI is still required.

Incoming #1084 donor/caller reconciliation remains required. G0 integration must
carry private method/helper deletions, the direct calls and the bounded public
lookup together. This proposal does not close Construction/IR, the directly
related families, native/development distribution, the remaining path audit or
Rust's existing closeout gate.


## Production watchdog returned to Runtime

A complete current-product import scan found one production dependency on
Testing: Statevector training imported its phase watchdog there. Testing's own
README explicitly forbids production imports for execution decisions. Move the
three complete original classes to `runtime/observability/watchdog.py`; no
watchdog math, fields, constructor validation, state transition, cause ordering,
phase budget or refusal changes. Production training, the Testing root and the
correctness scenario import the defining Runtime owner directly. This is the
only actual product import of Benchmarking/Testing outside those two families.

The old public module is an explicit three-type serialization facade. Preserve
its original five-name namespace (three types, the future-annotations binding,
and the dataclass decorator) and published class/method module identities. The
immutable original installed-parent pickle contains 21 actual objects: three
types, three instances, one bound observer and 14 class-owned functions. SHA-256
is `6710083a60095a2e93e65c2cf1f789f2d99602ff1e82149714a407d0b3a48361`.
A separate original snapshot covers 132 complete fixed-timestamp trajectories,
two constructor refusals, all five diagnosis causes and both bounded-phase
edges, including every state field and map order. Its SHA-256 is
`591fb3e5f0828154ec29df9adff3317c3e1da5427665f3a5dba1c4e5d834b23a`.
These controlled fixtures are not performance or scalability evidence.

Runtime and Integration own the retained public watchdog compatibility. Its
scope is the existing three public types and real original serialization
locators, not generic private forwarding or a new public promise. Removal needs
an explicit versioned public-payload migration, old-payload reading and the
release-policy compatibility window; no automatic 0.5.0 deletion is scheduled.
There is one implementation. The historical 102-facade record remains an
immutable historical inventory, with this converted public module documented
separately rather than silently added to that old baseline.

The original watchdog blob is identical in the SHA-verified #1084 full-tree
snapshot and the current parent. Later audited increments do not touch it or
the training caller. Incoming G0 integration must carry the canonical import
and owner move together with previously retired imports. The remaining
Benchmarking/Testing distribution, Native ABI, Construction/IR and Rust closeout
gates remain. Full source, cold Runtime import, original 21-payload restoration,
132-trajectory comparison, installed CLI/native qualification and exact new-head
CI are required before this expanded batch is accepted.


The exact original pickle is retained as a lossless ASCII/base64 test
fixture, with its raw SHA-256 and producer provenance. A permanent backward-reader
case checks all supported Python lanes without assuming canonical pickle byte
encoding across Python releases. Same-interpreter installed qualification also
requires an exact original-byte repickle. No fixture is regenerated from the new
implementation. The expanded complete slice has eighteen paths.


## Source and installed-artifact acceptance

All formal source regressions ran in the actual Git checkout, never a sparse
proposal or an artifact build copy. Counts below describe this migration,
not the earlier parent source qualification.

| Complete source suite | Passed | Existing skips | Pytest seconds |
| --- | --- | --- | --- |
| unit-api-team-final | 19,439 | 958 | 413.80 |
| qec-full | 2,801 | 329 | 84.61 |
| root-local-integration | 1,592 | 223 | 100.17 |
| benchmark-contract-release-consumers | 1,096 | 50 | 24.96 |

The focused closure passed 541 tests. All eleven quality checks, four supplemental
checks and Black 24.1 over the existing CI surface (`flagquantum tests tools`)
passed. Actual task-local optional JAX passed 120 tests with six existing
hardware skips. Independent AST comparison against Git parent 331637551
preserves 162 other definitions, the three complete watchdog class bodies and
all 76 original assertions after the exact import/retirement changes are
normalized; four negative boundary assertions and eleven old-payload assertions
are added. No scientific assertion, numerical body or public contract is changed.

The native wheel SHA-256 is
`6623cf47edd920e0b45e39d226ab0c844d72da59717db2d7a413f99be3dbe757`.
All 1,263 product Python files have identical sets and bytes in the actual
source, wheel and own installed site. Installation borrows the parent's
read-only third-party dependencies; this is artifact qualification, not a new
dependency-resolution certificate. The CPU native and SparseBlossom extensions,
benchmark CLI and no-Torch/JAX cold catalog import pass. Cold import of the
canonical Runtime watchdog additionally loads no Testing package and resolves
the public DTO annotations using only the standard library.

The installed suite passes 722 tests with seven existing skips and one existing
source-only AST case deselected; that case is covered by full source regression.
The ten original installed test files are byte-identical to the parent's
qualified 673-test selection; current sample-validation, correctness scenarios
and the immutable old-payload reader extend that selection. All 508 loaded
product modules originate from the wheel's own installed site. Test-only
watchdog tooling is copied unchanged; no product source directory is supplied.

Twenty-four original Audit exports and six original public PassSpec payloads
restore with the same objects/metadata; all six PassSpecs repickle to their
original bytes. The original 21 watchdog objects repickle byte-identically on
the same Python interpreter, including original bound-method execution. All
204 MPS/Audit scenarios and all 132 watchdog trajectories match the untouched
parent snapshots byte-for-byte, including tensor/RNG bytes, complete state,
map order and refusal behavior. Two original watchdog constructor refusals are
also preserved. Cross-version CI uses the permanent backward-reader test rather
than assuming identical pickle encodings between Python releases.

A full static scan covers all 1,217 product Python files outside Testing and
Benchmarking, resolving import/from bases and members plus fixed dynamic import
literals. It finds zero remaining product consumers of those developer
families. Caller-selected dynamic strings are outside that bounded proof; this
does not certify every path's architectural ownership or close developer-tool
distribution.

Initial installed-harness failure logs remain: three Gloo tests were denied
local loopback communication in the restricted sandbox and the exported test
harness lacked its test-only watchdog tool. Copying that unchanged tooling and
allowing loopback resolves the same tests without any product edit. An extra
Black check over the whole repository found 129 existing differences outside
the established CI surface; those unrelated files are not reformatted here.
The actual-source proof script initially used an invalid working-directory
adapter; correcting the proof script required no product or test source edit.

The first complete benchmark-contract attempt passed 1,089 tests with 50
existing skips but produced seven setup errors from the same recovery fixture:
its two-process local rendezvous exceeded the original 240-second timeout. Both the original restricted run and a communication-enabled run exited with
the same fixture timeout; their logs remain untouched. Live sampling locates
TCPStore connection setup before ranks launch, and an independent controlled
launcher reproduces the timeout with the default hostname while both ranks
launch in 1.04 seconds with the existing `PET_LOCAL_ADDR=127.0.0.1` option.
The qualified complete run uses that task-local launcher setting with loopback
permitted. No product/test/dependency, workload, fixture timeout or criterion
is changed. This is not a numerical performance comparison.
CPU recovery checks establish semantics, not accelerator scalability.

## Retained watchdog compatibility, item by item

| Existing published type | Existing serialized method locators | Concrete basis |
| --- | --- | --- |
| `ProgressSnapshot` | `__init__`, `__repr__`, `__eq__`, `__setattr__`, `__delattr__`, `__hash__` | Original type, populated DTO and all six class-owned functions in the immutable parent pickle |
| `StallDiagnosis` | `__init__`, `__repr__`, `__eq__`, `__setattr__`, `__delattr__`, `__hash__` | Original type, populated diagnosis and all six class-owned functions in that pickle |
| `PhaseAwareWatchdog` | `__init__`, `observe` | Original type, populated observer, bound observer and both class-owned functions in that pickle |

All three keep the exact original `flagquantum.testing.watchdog` locator and
Testing-root identity, while the single implementation belongs to Runtime.
Runtime/Integration own this scope and the versioned removal condition stated
above. The module's existing `dataclass` and `annotations` bindings are kept to
avoid changing its namespace; this introduces no new public API promise.

## Runnable existing public workflow

```python
from flagquantum.testing import certification_matrix, execute_certification_case
from flagquantum.testing.watchdog import PhaseAwareWatchdog, ProgressSnapshot

case = next(
    item for item in certification_matrix()
    if item.backend == "pytorch" and item.operator == "x"
)
result = execute_certification_case(case)
assert result.executed and result.passed

watchdog = PhaseAwareWatchdog(stall_seconds=5)
assert watchdog.observe(ProgressSnapshot(0.0, "input", "read", 1, 100)) is None
diagnosis = watchdog.observe(ProgressSnapshot(6.0, "input", "read", 1, 100))
assert diagnosis.cause == "stalled_input"
```

This existing public workflow executes against the installed wheel. Production
training now imports the Runtime owner directly; extension authors can still
read their original public objects and stored payloads. The fixed timestamps
are deterministic correctness evidence, not measured performance or distributed
capacity evidence. No new runtime manager, dependency or public abstraction is
introduced.


## Incoming reconciliation at publication preparation

Incoming PR #1084 head `009044f67060fedf4b6a3aa9ac3c9d2f5ced4a52`
adds four commits and twenty changed paths over the previously audited 04b72b6c6.
Every changed blob is SHA-verified; none intersects this eighteen-path closure.
The increment changes Register/Core IR implementation, contracts, documentation
and tests; those responsibilities retain their existing G0 closeout gate.
The active 583-row retirement ledger reports the exact same seventeen unfiltered
incoming diagnostics over 3,284 changed Python files, with no additions or
removals. This is an incremental consumer audit, not whole-incoming architecture
acceptance. Parent PR #1341 was externally merged during acceptance; the new
PR can target main without waiting for #1084.
