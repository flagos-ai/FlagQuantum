# Runtime direct-seam retirement

Prerequisite: qualified PR [#1350](https://github.com/flagos-ai/FlagQuantum/pull/1350),
`864605d88a8d9658f0838edbcb2d06ac5442929a`. The main-based carry commit
`376ee2fb2e229af17931095feddf637172cc0121` has exactly that qualified tree.
Status: PR #1351 head `c96e63362807b04adb083d656d8086aea50d8f3f` passed
all three applicable workflows on their first attempts: 432 successful checks
and four existing conditional skips. All 384 CPU partition reports were
retained and verified. Maximum whole-job times were Runtime 531 seconds and
coverage 585 seconds, including preparation and upload but excluding queue.
This qualifies that exact head; later heads require their own CI.

## Canonical callers and deleted delegation

Planning now lives in Runtime. The former Compiler-to-Runtime adapter has no
Compiler caller in either the current product or the audited incoming product.
Delete that internal module and call the existing configuration and planning
owners directly. JAX summaries likewise call their existing audit owners without
an intermediate argument-forwarding function.

| Retired binding | Unchanged defining authority |
| --- | --- |
| `runtime/planner_adapter.py::backend_execution_options` | `runtime/configuration.py::backend_execution_options` |
| `runtime/planner_adapter.py::plan_jax_statevector_training` | `runtime/executors/jax/statevector/training.py::plan_jax_sharded_statevector_training` |
| `runtime/planner_adapter.py::plan_jax_mps_training` | `runtime/executors/jax/mps/planning.py::plan_jax_sharded_mps_training` |
| `runtime/executors/jax/release_policy.py::attach_evidence_contract` | `runtime/audit/statistics.py::attach_distributed_evidence_contract` |
| `runtime/executors/jax/release_policy.py::attach_mps_backward_readiness` | `runtime/audit/mps_readiness.py::_attach_mps_runtime_summary` |

This removes four function definitions and one obsolete module, and routes
fourteen evidence return sites directly to their owners. Backend configuration
was already the same re-exported function. Imports remain local at their original
invocation stage; no audit or JAX dependency is loaded earlier. Distinct statevector and MPS planner variables
preserve their precise return types after removing the old `Any` forwarding.
The remaining statevector claimability function performs real classification and
stays in its existing owner.

The ledger grows from 591 to 596 rows. All five additions are retirement-only,
with no claimed implementation definition move. Existing source-identity maps
remain identical. No old name, replacement wrapper or public compatibility layer
is retained for these internal seams.

## API and preserved evidence

The [public API policy](../../reference/PUBLIC_API_POLICY.md) explicitly identifies
Runtime implementation modules as internal. The old adapter did have an internal
`__all__`; that is not a Stable Core admission. All four removed function names
are absent from the stable snapshot and pre-freeze baseline. Existing public
serialized callbacks, admitted domain exports, signatures and object identities
remain protected; no API snapshot or scientific contract is rebaselined.

The complete original 38-case protocol was captured from the prerequisite's own
native installation before editing. It retains returned values, refusals,
mapping order and complete input pickle bytes before and after each call. The
first original planning call populates the existing Circuit IR cache; that
observed effect is preserved rather than assumed absent. The modified source
reproduces all 38 original rows exactly.

Normalized complete AST comparison preserves all 76 remaining definitions and
methods in the twelve surviving changed product modules. The admitted changes
are the four deletions, direct lazy imports/calls and local planner-variable
renaming. No numerical authority, transport, evidence gate, refusal or optimizer
implementation changes. All existing test, fixture, contract and native paths
remain unchanged.

Fresh original cold-import inventories cover thirteen defining/caller modules.
All twelve surviving modules preserve their heavy-module load state and import
no new product module; neither JAX nor jaxlib is loaded eagerly. The deleted
adapter has no import specification in the modified source.

## Incoming integration and remaining acceptance

Incoming draft/conflicting #1084 advanced through
`8efb2f8e71bb6891420a6e6c5a94e7d0573ddbaf`. All thirteen paths in its latest
four-commit increment are SHA-verified, with no carry or candidate overlap.
The complete incoming product tree has 2,099 SHA-verified Python sources.
Its candidate importers are all in Runtime and both defining files match the
prerequisite. This is a complete static product-consumer review for these named
seams, not whole-branch execution or integrated G0 qualification.

Incoming reconciliation must retain the direct caller changes, rather than
restore retired names. Recompute and retain the unfiltered incoming retirement
diagnostics for the new ledger, including any new obligations. Earlier raw
seventeen-diagnosis records are historical evidence, not a filtered allowlist.
The 55-entry top-level direct-call census is a candidate list; it does not
certify all methods or the remaining package ownership review.

The complete source default selection passes 21,921 tests with 864 conditional
skips in 490 seconds including startup. All 22,785 earlier nodes and outcomes
remain identical; no existing test changes. Real CPU JAX passes all 207 original
cases with six existing device conditions, preserving all 213 selected outcomes.
The independently installed artifact passes those same 213 outcomes, the full
original 38-case protocol and all twelve surviving cold-import comparisons.
All 1,264 product Python files match source, wheel and installation. Both native
extensions and all 507 loaded product providers resolve from that installation.
The deleted planner adapter is absent from source, wheel and installed import
lookup. No optional dependency resolution is claimed.

Wheel SHA-256:
`3129fa93ad34e3c9c335056c33d4e376864eff88e62e577c7d483c5d44267c80`.
All eleven quality gates and the applicable pinned hooks pass. The initial
strict-type diagnostic exposed the old `Any` wrapper's shared planner-variable
binding; distinct, precisely typed locals correct it without suppressions or
contract changes. Its initial log and the passing complete check are retained.
The default suite, actual JAX execution and original protocol use that correction.

The 596-row unfiltered changed-incoming audit retains the same seventeen raw
diagnoses, with none added or removed. The full 2,099-source product-consumer
review is separate from that changed-source scan. Neither certifies integrated
G0. Exact new-head CI remains mandatory. Construction/IR/Registers, directly
coupled Compiler/Simulation,
native/developer distribution and repository-root G0 remain open. Rust remains
deferred until incoming and architecture closeout. No merge or force push.
