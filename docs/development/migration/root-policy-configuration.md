# Root policy configuration

Integration declaration, committed before implementation. Owner: Integration.

Capability maturity and evidence-revision origins belong with repository policy
under `config/`. Their existing data, evidence, origin declarations and rejection
semantics remain unchanged. This relocation introduces no runtime configuration
API, loader manager, fallback, second file or historical-root facade.

| Current source | Unique target | Original SHA-256 |
| --- | --- | --- |
| `capability-maturity.toml` | `config/capability-maturity.toml` | `0bac21b651931d82ce48a8a5de20cf2bf548c6484b24285e93597627c325857f` |
| `evidence-revision-origins.toml` | `config/evidence-revision-origins.toml` | `16380903835569f4890e4dbc59f80fbf2f33a5e009032c6f66f367c55efa7005` |

## Source and protected boundaries

The prerequisite is fully qualified PR #1356, head
`7c737f2237a7d45e316768c375e38b1fa6f9b324`, tree
`9bf41983b7e43c7740e591ff55ea42282abaf28f`. Its exact tree is carried on main
`29afa414a6362d381cb4c2992cd19fd024761f83`; its CI does not qualify this new head.

Incoming #1084 at `66a2b1528a49417d2f78f3cd8950810d1fbbd2a0` leaves these two
policies at their old common-base bytes. Main has evolved both. Keep main's
complete current data rather than restoring the incoming old-base versions.
The latest three-commit, eighteen-file increment is fully hash-verified, its
fifteen Python files AST-parsed, and all eighteen changed texts checked for
these policy references. It changes neither the policy/authority inputs nor
the proposed destinations. This is scoped independence evidence, not merged
architecture or whole-#1084 acceptance.

Real default readers must use the new paths: maturity validation, parity
matrix input, QEC alignment, evidence-origin validation and documentation
authority input. `docs/source_of_truth.json` is an actual reader locator and
must change. Update current evidence-retention consumers and valid generated
links as well. Repository-relative evidence, examples and source paths remain
rooted at the repository, not at the policy file's new parent.

Two versioned authority identifiers stay byte-for-byte unchanged:

- `contracts/cudaq-parity-matrix.toml::source_of_truth` names
  `capability-maturity.toml`; its loader uses the separate `MATURITY_PATH`.
- `contracts/research-capability-services-v1-candidate.json` names the same
  `authorities.product_maturity`. The approved external-adapter candidate and
  its exact original authority assertion remain unchanged.

Preserve both complete contracts and their API/schema/approval status. The
basename remains the authority's name while physical default readers move.
This creates no old-root file or loader fallback. Preserve diagnostic basenames
and arbitrary explicit declarations-file arguments, including the synthetic
evidence tests that supply their own paths.

## Existing contributor workflow

From a full-history repository checkout:

```bash
python tools/validation/check_capability_maturity.py
python tools/release/check_evidence_revisions.py
```

These existing commands continue to validate the same maturity and evidence
declarations and exit successfully for the accepted repository data. Their
defaults read `config/` after implementation. They do not execute a quantum
workload, change recorded maturity, fetch missing evidence or submit a cloud job.
History-limited skips do not certify the full-history provenance gate.

## Complete implementation and acceptance

Move both TOML files unchanged and remove their root predecessors. Update all
real readers, authoritative documentation input, current retention consumers,
relevant tests, source/archive/install selections and current physical links.
Use existing `config/**` Integration protection; remove obsolete root rules.
Update the two non-Python relocation rows without claiming moved definitions.
Retain historical scientific records and exact original API expectations.

Stage the complete rename before tracked-file ownership checks. Run the
unchanged focused policy, documentation, contract and numerical consumers,
then the complete default selection. Check both policy hashes, both untouched
contracts, original node/outcome multiplicities and all refusal behavior.
Qualify the actual committed PR source, installed/distribution impact and all
applicable CI with original workloads/resources. Target every executed CI job
below 600 seconds, including preparation/upload and excluding queues; preserve
all failed and over-target observations rather than reducing original work.

The #1084 connected families, native ABI relocation, independent developer
distributions and final integrated acceptance remain separate unfinished
closures. Rust, merging and force pushing are outside this work package.
