# Dependency policy location

Integration declaration was committed before implementation. Owner: Integration.

Move `dependency-policy.toml` unchanged to `config/dependency-policy.toml` and
remove the root file. Keep one authoritative file, without a copy, symlink,
loader relay or old-root fallback. This changes repository organization, not
dependency versions, optional extras, import rules or certified version lanes.

Original SHA-256: `b7f3eca5c0605bcbe0dfae68d4f7fb1dc28189fdb9b4871feea2d71c3feab532`.

## Source and dependency boundary

The prerequisite is fully qualified PR #1358, head
`cb1938232d9ebaa80f51b1a099ec2f41608899d7`, tree
`bcb7020c9d999779ccb51e38c6df482cf95f6d55`. Its exact tree is carried on main
`29afa414a6362d381cb4c2992cd19fd024761f83`; its CI does not qualify this new head.

Incoming #1084 at `1f085a9af3ca424fee329a340e0325f9a69cb031` has no target
collision. Its nine changed referencing files were blob-hash verified and their
Python ASTs inspected. Physical default readers can move independently of its
FTQC implementation. Preserve main's complete current policy: incoming old-base
data lacks main's chemistry/OpenFermion declarations and import restrictions.
The incoming isolated `test_environments` additions remain a separate semantic
contribution to reconcile during final #1084 integration. Do not overwrite
either branch's valid additions or claim that integration has already passed.

## Complete consumer closure

Update the standalone dependency and Cirq checker defaults, five canonical
validation defaults, direct policy-reading tests and the Qiskit benchmark lane
reader. Repository-relative scans, manifests and contract inputs remain rooted
at the repository, not the policy file's new parent. The standalone checkers
are substantive implementations and must not be deleted as forwarders.

Update the Qiskit wheel-cache `hashFiles` input and its existing assertion.
Move the OpenFermion workflow path filter; both original Python 3.10 and 3.12
SDK lanes must execute for this PR. Synchronize current governance, security,
dependency, interoperability, testing and roadmap references, configuration
README, ownership and one exact non-Python whole-file relocation row.

Keep `contracts/cudaq-parity-matrix.toml` byte-for-byte unchanged. Its narrative
policy names, diagnostic basenames and exact original error assertions are not
physical file loaders. Preserve historical approval/audit/scientific records
and the arbitrary explicit temporary policy fixture in the Qiskit lane tests.
No new contract, registry or generalized path resolver is required.

## Existing contributor workflow

From the repository root:

```bash
python tools/architecture/check_dependency_policy.py
python tools/architecture/check_dependency_policy.py --print-floor
```

The first command validates the same declared extras and source import rules
and prints `dependency policy passed`. The second prints the same existing
pinned development floor. Neither installs software, changes environments or
submits a remote task. The configuration README also lists the first command
as the shortest contributor validation path.

## Acceptance

Compare policy and frozen-contract hashes. Verify each Python AST differs only
by declared physical paths; preserve explicit fixture paths and diagnostic
strings. Replay all eight original CLI stdout/stderr/exit codes exactly and
run all original positive and refusal consumer tests. Run ownership,
architecture, hygiene, documentation-source and link checks, followed by the
complete default suite and original node/outcome/multiplicity comparison.

Qualify the actual committed CI source/tree and all applicable workflows,
including both OpenFermion lanes, source/archive/distribution behavior, all
384 original Runtime/coverage reports, original scientific call phases and
admitted resources. Target every executed job below 600 seconds, including
preparation/upload and excluding queue. Retain failed and over-target
observations; do not cut original workloads or substitute favorable reruns.

The connected #1084 families, native ABI relocation, independent developer
distributions, remaining bootstrap closure and final integrated G0 remain
unfinished. Rust, merging, force pushing and runner administration are outside
this work package.

Implementation is present; local and own exact-head CI qualification remain
separate acceptance requirements, not claims established by this declaration.
