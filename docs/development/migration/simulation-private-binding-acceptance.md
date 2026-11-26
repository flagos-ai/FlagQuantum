# Simulation private-binding acceptance

This complete batch follows CI-qualified PR1335 head
`f986fb4026b46a585a165becaaae52f8f3bce6cc`. It retires 29 unused private
aliases from Dynamics, Statevector and Stabilizer, plus six unused Dynamics
imports. All production, test and benchmark-producing consumers use existing
defining modules directly; no forwarding is added. The seven defining owner
files remain byte-identical. The cumulative migration ledger has 545 rows,
and the itemized public-entry inventory remains 102 with no additions.

## Source and regression evidence

All 27 changed Python ASTs preserve behavior after imports and known owner
bindings are normalized: 280 product definitions (including nested definitions),
35 benchmark definitions and 614 test assertions. All 644 audited frozen
contract, scientific-result and approval blobs remain byte-identical.
Public signatures, precision, RNG order, dispatch, error behavior and evidence
are unchanged. Core circuit lazy imports remain lazy; its 1248 lines remain
within the unchanged 1250-line guardrail.

Focused preview scenarios pass **811 tests**, with 44 existing conditional
skips. The formal full suites ran independently in the actual Git checkout:

- unit-api-team-final: **19,420 passed / 957 skipped**, 441.05 seconds.
- root-local-integration: **1,592 passed / 223 skipped**, 87.95 seconds.

The affected QEC Stim bridge, measurement conversion and binary-record
scenarios additionally pass **34 tests / 20 existing conditional skips**.

Existing optional/hardware skip policies and the credential-dependent Kaiwu
live-smoke exclusion are unchanged. Eleven source quality gates pass,
including strict mypy over 1262 sources. Black 24.1 and 26.10 both pass all
3663 Python files. The operator manifest, runtime schema, legacy-root usage
and correctness certification checks pass. Applicable pinned source hooks and
final documentation checks must pass before committing.

## Native wheel and historical payloads

The qualified native wheel SHA256 is
`fdb0010bff2f1a25f65343824faf6c0442da228a7332a46db48b701239f2501d`.
All 1262 product Python files match source, wheel and clean installation
byte-for-byte. The formal source matches that qualified source; the final Git
commit must bind those same blobs. Native CPU and SparseBlossom extensions load
from the clean installation. Cold kernel metadata imports load neither Torch
nor JAX; the installed benchmark CLI exits successfully.

Twenty-four installed public flows and the original coarse-grid physical-state
refusal match the parent byte-for-byte, including values and metadata. Their
snapshot SHA256 is
`f4c25e1302bd10c8ebb23b9a3bd6c73e5e0117ef10b069dae99289c38b4a5047`.
The unchanged original parent pickle restores three Evolution objects and seven
public references with equal resolved field hints and tensor-byte hashes.
A separate original parent pickle restores all 29 retired-alias function/class
references through their unchanged actual owners and a public Observable with
its existing `_PauliTerm` payload. The restored Observable's public expectation
matches the parent byte-for-byte. `_PauliTerm` remains the required historical
public payload locator; no new public compatibility is introduced.

The macOS arm64 artifact establishes this platform's installed behavior.
It does not establish multi-GPU/multi-node scalability or certify other native
platforms; the exact pushed-head workflow matrix remains mandatory.

## Incoming integration boundary

The changed incoming-source audit reaches draft/conflicting PR1084 head
`a4fcc985733c867ba67bcd1b1a17e1b708cb88d5`. Its last two increments have 20 and
13 changed paths, independently blob-verified, with no candidate-path
intersection. The audit covers 3259 changed incoming Python source trees,
not every unchanged file on that branch.

The raw new-retirement diagnostic contains one old diagonal import in Runtime
`dynamic/execution.py`. Its only name load is inside `_apply_instruction`, whose
AST is identical to the qualified pre-retirement donor. The qualified parent
already moved that definition to Simulation and retired the donor. Joint
integration must carry that parent donor retirement and remove the orphaned
import; it must not restore the Operations alias. There are zero additional
independent feature consumers. The original diagnostic is retained rather than
rewritten as a false zero. A separate full parent-ledger audit retains fifteen
existing incoming old-import obligations.

Construction/IR, directly dependent compiler work, native ABI/development
packaging and Rust retain their existing PR1084 closeout gates. This batch
neither certifies joint integration nor completes the whole restructuring goal.
