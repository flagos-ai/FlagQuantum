# Shared Runtime placement counts

Preparation parent: `800e1ac67f10f6c4e601101ccca998aa69224a6c`, PR #1342.
Status: complete actual-source and installed-artifact acceptance passed.
Parent PR #1342 is qualified at the exact preparation commit. The new
published-head CI remains required; no pending result is certified.

## One backend-neutral owner

The Runtime process-group context, Statevector planner and JAX common module
contain the identical node-count formula. Context additionally wraps its two
existing resolvers with exact-argument private functions. Move the original
`resolve_local_world_size` and `resolve_node_count` bodies unchanged to
`runtime/distributed/placement.py`; delete the two duplicate formulas and two
private wrappers. Rebind all twenty-three current consumer paths directly to
the defining module. The JAX common module only loses a duplicate definition;
it needs no replacement import. Twenty-four existing product Python files change,
with one new concrete owner and no new abstraction, manager, dependency or
public API exports.

The new owner uses only the standard library. Putting it in the existing
Planner package would initialize Compiler and planning dependencies through
that package's initializer; importing Torch's context would activate Torch
for JAX-only metadata. Neither dependency is necessary to count hosts or read
the existing torchrun placement variables. This leaf belongs beside existing
Runtime distributed concerns and is shared by all executor representations.

## Exact retirement and caller closure

| Existing implementation path | Removed internal names | Defining owner |
| --- | --- | --- |
| `runtime/distributed/context.py` | `resolve_local_world_size`, `resolve_node_count`, `_resolve_local_world_size`, `_node_count` | `runtime/distributed/placement.py` |
| `runtime/executors/statevector/planning.py` | `_node_count` | Same existing node-count body at the new owner |
| `runtime/executors/jax/common.py` | `node_count` | Same existing node-count body at the new owner |

The first two Context definitions move; the latter four definitions retire.
No module getter, alias injection, callable wrapper or old internal reexport
is added. Consumers import the defining module and call its actual functions.
The ledger has 586 rows. The historical 102-facade inventory and the separately
justified public watchdog facade from PR #1342 do not change.

All tracked Python imports/from bindings and loaded name/attribute references
were audited across 5,565 files, including tests, tools, benchmarks and examples.
The selected references are direct calls in twenty-three product paths; no
fixed full-path serialization/dynamic-import literal was found. Computed
strings, inherited attributes and unrelated definitions are outside that
bounded proof. Formal actual-source and installed verification are recorded below.

## Existing public boundary

Runtime Executors explicitly implements an internal backend boundary, not a
public backend API. The Distributed package's published Context exports are
`TorchDistributedContext`, `destroy_torch_distributed`,
`init_torch_distributed` and `torch_distributed_is_available`; the moved helper
names are not published in that package's `__all__` or `_CONTEXT_EXPORTS`.
They are absent from the frozen API exports. Non-underscore helper spelling
does not create a new compatibility promise. No old helper is returned in a
public callback/spec or stored in the published context record.

Keep every existing public class, constructor, field, exported lifecycle
function, signature, default, summary/schema and original class/method module
identity. An untouched original installed-parent payload contains the public
`TorchDistributedContext` type, a populated instance and its bound `summary`.
The new installation must restore it and preserve its original bytes and all
summary fields; no public facade or serialization adapter is needed because
this class and method do not move. All existing contract/fixture bytes remain.

## Semantics that stay separate

- `LOCAL_WORLD_SIZE` still takes precedence over `NPROC_PER_NODE`; an empty
  first value still follows the original fallback. Existing int conversion,
  clamping and invalid-value refusals remain, including their evaluation order.
- A subgroup still gets its own world size before either environment variable
  is read. Calls that formerly used the private wrapper use the same canonical
  default `subgroup=False`.
- JAX's explicit device/profile selection in `runtime_environment.py` has a
  different input contract and stays there. Runtime request options and backend
  policy keep their existing precedence. They are not rewritten to torchrun
  environment semantics.
- Node-rank environment overrides and communication-tier implementations keep
  their current owners and evaluation order. No rank, device or group placement
  policy, transport, collective, state partition, numerical algorithm, gradient
  update, optimizer, checkpoint or evidence schema changes.

Original parent qualification saved 3,888 complete node-count/environment
outcomes, including all refusal types/messages and subgroup/precedence cases.
The immutable SHA-256 is
`25dfb136391e4e536b18b72c808099304e749fdfba34d5b19a6cbb25ea2a0318`.
A standalone proposed-owner comparison matches every byte. It proves these
concrete helper outcomes, not complete executor or hardware acceptance. Do not
regenerate this original fixture to accommodate a new implementation.

## Incoming donor and integration obligation

The nontruncated full tree of PR #1084 at
`009044f67060fedf4b6a3aa9ac3c9d2f5ced4a52` contains 2,223 product paths.
All six selected complete definitions are SHA-verified and AST-identical to
current source. All twenty-three current consumer paths exist there, but their
complete modules are not claimed identical. The new owner path is absent.
This current migration needs no new #1084 API or behavior; G0 must carry the
same owner/caller retirement rather than resurrecting the old helpers. Raw
incoming diagnostics and the exact consumer reconciliation remain explicit
acceptance obligations. Existing Construction/IR, related family, Native ABI,
developer distribution, remaining root closure and Rust gates stay in force.

## Ten-minute contributor workflow

The existing user request still plans and runs on one CPU without changing its
circuit or learning about placement helpers:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(1).h(0)
result = fq.run(circuit, outputs=fq.expectation(fq.X(0)))
torch.testing.assert_close(result.expectation(), torch.ones_like(result.expectation()))
```

For a placement-related change, start with `runtime/distributed/README.md`,
then run the existing environment/topology/subgroup workflows:

```sh
python -m pytest tests/test_distributed_backend_policy.py \
  tests/test_distributed_statevector.py tests/test_tensor_network.py \
  tests/unit/test_structure_import_compatibility.py -q
```

The permanent cold-owner test also checks environment precedence and subgroup
resolution without importing Torch, JAX, Compiler or Planner. Complete source,
optional JAX, installed native/CLI/public-payload and exact published-head CI
qualification are required before acceptance. CPU fixtures establish semantics
only and do not certify multi-GPU/multi-node capacity.


## Actual incoming merge-base check

GitHub identifies the actual common ancestor of main 2b985460c and incoming
009044f670 as `e48251b01754cecd783ddf0206e24f75583f7434`, tree
`34e849bb08749911fdadd74269faa153589b5637`. Both recursive trees are
nontruncated. All twenty-four affected existing product files have exactly the
same Git blob on the incoming side as at that common ancestor, and the new
owner exists on neither side. A normal three-way merge therefore preserves
this current-side family migration at that fixed incoming head. It does not
need a speculative old-path compatibility layer or a rewrite of untouched
upstream files. Recheck later incoming heads and the final integrated tree.

The proposed 586-row guard against the 3,284 changed-incoming Python files
retains exactly the same seventeen unfiltered diagnostics, with zero additions
or removals. Those changed files do not include this unchanged twenty-four-file
family. A standalone old incoming snapshot is not a simulated merge result;
the actual common-ancestor equality is the independence evidence. None of this
closes the remaining G0 integration, full architecture audit or Rust gates.


## Latest incoming increment

The next audited incoming head is
`48d4ef17c855b182199f05a1679b1650fee19269`: eight commits and twenty-three
changed paths over 009044f670. Every blob is SHA-verified; none intersects this
thirty-one-path proposal. The actual merge base stays e48251b017 / tree
34e849bb08, so the selected twenty-four family blobs remain unchanged on the
incoming side. Both the current 583-row and proposed 586-row ledger retain the
same seventeen unfiltered diagnostics over the latest changed-incoming Python
set. Cirq/IR changes and the existing Cirq root checker remain in their directly
coupled G0 scope; this does not move or alter them early.


## Source and installed-artifact acceptance

The complete source suites ran in the actual Git worktree on
`refactor/structure-runtime-placement-counts`, never in the sparse proposal or
artifact build copy. Parent PR #1342 passed all applicable checks first, with
243 successful checks and its three existing conditional skips. The new
published-head CI must qualify independently.

| Complete source suite | Passed | Existing skips | Pytest seconds |
| --- | --- | --- | --- |
| unit-api-team-final | 19,440 | 958 | 437.17 |
| qec-full | 2,801 | 329 | 86.20 |
| root-local-integration | 1,592 | 223 | 90.83 |
| benchmark-contract-release-consumers | 1,096 | 50 | 23.75 |

Focused distributed/backend/topology, MPS/TN and public import workflows passed
430 tests with one existing skip. Real task-local optional JAX passed 120 tests
with six existing hardware skips. All eleven quality checks, four supplemental
checks and both Black 24.1 and current Black over the established CI surface
passed; strict typing covers all 1,264 product Python files. The first Ruff
attempt failed solely when trying to create its cache outside the writable
sandbox. Its original log/results remain; the identical full rules/surface
passed with `--no-cache`. No source, rule, exclusion or pass criterion changed.

Whole-module AST comparison against actual Git parent 800e1ac67 preserves the
two complete canonical definitions and all 305 other definitions after exact
retirement/import binding changes are normalized. All 45 original test
assertions remain; six absence checks and one permanent cold-owner scenario
are added. No test body, scientific assertion, contract, fixture, numerical
formula, threshold, dependency or ABI is rewritten to obtain these results.

The native wheel SHA-256 is
`48f506b5653db0b757f99579d84ef68788bce5e489ee9421a421d86bd051b579`.
All 1,264 product Python files have the same sets and bytes in actual source,
wheel and own installation. Third-party dependencies are borrowed read-only
from the qualified parent; this is artifact qualification, not a fresh
dependency-resolution certificate. Both CPU native and SparseBlossom imports
are verified actual extension binaries, with all loaded product modules from
the new installation. The benchmark CLI and cold catalog pass. Cold placement
also confirms original environment precedence/subgroup behavior without
importing Torch, JAX, Compiler or Planner.

The installed workflow suite passes 724 tests with seven existing skips and
one existing source-only AST case deselected; the complete actual-source suite
covers that case. Ten original test files are byte-identical to the previously
qualified parent selection. Current sampling/correctness and the unchanged
watchdog backward reader are retained; the cold placement and old-namespace
absence cases extend that selection. All 518 loaded product modules originate
from the wheel's own site. Test-only watchdog tooling is copied unchanged and
its harness path is checked; no product source directory is supplied.

All 3,888 complete original placement/environment outcomes and refusals match
the original parent snapshot byte-for-byte, including error types/messages and
fallback/subgroup cases. The original public Context type, populated instance
and bound summary restore and repickle to their exact original bytes; the
summary preserves all fields. Their original SHA-256 is
`b5ab22eae64991db199697bc3e07e7ab1862a740490bdcb824e0a65418055b22`.
All six retired internal names are absent in the installed artifact.

A separate installed replacement check supplies a valid fixed single-rank
count implementation at the canonical owner, calls the existing public
`init_torch_distributed` without changing its consumer, and restores the
original implementation in a `finally` block. The supplied implementation is
actually called with `(1, 1)`, and the resulting public Context reports one
rank/host without initializing a process group. This proves replacement at the
shared boundary without copying the formula, adding an adapter or manufacturing
hardware evidence. CPU checks here establish semantics, not accelerator or
multi-node scalability.

The initial temporary-proposal builder incorrectly demanded a surviving call
in the JAX common owner-only file after deleting its duplicate definition.
That stopped before actual source application; its partial output and script
are retained. Allowing an owner-only deletion removes the unnecessary import,
without changing original bodies or parent snapshots. Formal source and native
artifact acceptance above use the real Git implementation.


## Publication preparation at latest incoming head

The next incoming audit covers
`7a62bf0989c4dec1af89ca8950dc9aa945c0d93e`: seven commits and nineteen
SHA-verified changed paths over 48d4ef17c8, with no overlap with this thirty-one
path closure. The actual common ancestor with the newly merged main remains
e48251b017 / tree 34e849bb08, extending the selected-family independence proof.
Both the parent 583-row and active 586-row ledger retain the exact same seventeen
unfiltered diagnostics over 3,289 changed-incoming Python files. The newly
modified multi-level IR root checker remains in its existing G0-coupled family.
Parent PR #1342 was externally merged after exact-head qualification and being
marked ready; its merged main tree equals the qualified parent source tree.
The new PR can target main. No merge or force-push was performed by this task.
