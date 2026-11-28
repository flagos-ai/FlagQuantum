# CI data location

Integration declaration before implementation. Owner: Integration.

The three remaining root CI JSON files belong beside the CI tooling. Move their
original bytes, remove the root predecessors and empty directory, and update real
default readers and current physical references. Do not introduce a copy,
symlink, loader wrapper, new registry or root-path fallback.

| Current source | Unique target | Original SHA-256 |
| --- | --- | --- |
| `ci/pr-gate-main.json` | `tools/ci/pr-gate-main.json` | `730c1413af41e3dade4b3abfe645ee8921c12bd4e84412f0942817b598508b8b` |
| `ci/flagos_cuda_reference.lock.json` | `tools/ci/flagos_cuda_reference.lock.json` | `d48bd5750094965cdfeb91f297c32aea054ffefa0ae8e85918b63a759bad81ed` |
| `ci/domestic_single_card_attestation.template.json` | `tools/ci/domestic_single_card_attestation.template.json` | `d2f8a2c749b20cddb1a29de344b6c04868aee71505667bf77d4804ae68ac1f6c` |

The exact prerequisite is qualified PR #1357, head
`7d350a1589172887182af9d7a6beeb60ab4adbd3`, tree
`5456dd16bc4d12abd1701337d148d4a20b92c27c`, carried on main
`29afa414a6362d381cb4c2992cd19fd024761f83`. Its checks do not qualify this new head.

Incoming #1084 at `1f085a9af3ca424fee329a340e0325f9a69cb031` leaves all three
files identical to current and common-base bytes, with no destination collision.
All 4,685 earlier changed blobs and 4,674 text blobs were checked for these
filenames; the subsequent six-file/four-Python increment preserves the complete
input and consumer review. This scoped independence review does not qualify the
incoming program/type implementation or final integrated architecture.

## Frozen data, contract and administration

Preserve the complete domestic-single-card contract and its historical template
identifier. Its existing checker already uses `source_path` for every declared
verification path and the template open. Add three exact whole-file data
relocation rows to that existing mapping; do not modify the contract or widen
mapping rules. Existing `tools/**` ownership protects the targets; remove the
obsolete root `ci/**` rule.

The lock's versions, image digest, build settings, device count, budgets and
semantics stay unchanged. Keep both original A100/A800 recorded observations
byte-for-byte. Their current offline acceptance baseline is explicit: the A800
record passes, while the older A100 record is rejected by the current schema,
environment and recorded-source rules. Preserve both stdout and exit status;
do not alter old evidence or relax checks to make every historical record pass.
New generated evidence continues to state its actual lock path and bytes hash.

Moving the ruleset JSON changes only its repository location. No remote ruleset,
branch protection, auto-merge, queue, quota, runner or merge setting is changed.

## Existing contributor workflow

From the repository root:

```bash
python tools/validation/check_domestic_single_card_contract.py
python tools/validation/validate_flagos_reference_evidence.py artifacts/flagos_cuda_reference_a800_20260824.json
```

Both existing commands validate the same unchanged contract/template and recorded
evidence and exit successfully. They do not probe hardware or submit a job.
The older A100 record retains its original rejection under the same validator.

## Complete closure

Update both real validator defaults, the three direct test readers, current
Markdown references, CI README, ownership and the three non-Python move rows.
Stage the full rename before ownership checks. Check exact data/contract/original
evidence hashes, source-mapping behavior, all original positive and negative
acceptance cases and the complete unchanged default selection. Verify actual
source/archive/distribution impact and every applicable exact-head CI workflow.
Preserve source selections, scientific call phases and admitted resources; target
each executed job below 600 seconds including preparation/upload and excluding
queue, retaining failures and misses. Never inherit predecessor green checks.

Remaining dependency/bootstrap, #1084 connected domains, native ABI, developer
installations and final G0 are unfinished closures. Rust, merging and force
pushing remain outside this work package.
