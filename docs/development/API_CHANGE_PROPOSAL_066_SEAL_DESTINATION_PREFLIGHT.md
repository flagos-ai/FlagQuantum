# API Change Proposal 066: The seal destination, not the release directory

## Status

**Approved by the repository owner (selected as the recommended remedy when the
campaign was stopped by the defect) and implemented by the integration lane.**

The change is confined to `/nfs`-external release tooling under `tools/`, which is
a protected integration surface rather than a Stable Core contract. It adds no
root export, no function signature, no plan property, no result field, and no
serialization schema version, and it regenerates no public-API snapshot.

## Problem

`tools/check_release_evidence_environment.py` is the preflight the sealer runs
before it writes anything. One of its checks was:

> `scalability directory already contains promoted JSON; seal into the candidate
> directory and promote only after strict audit`

The check read `SCALABILITY_ROOT.rglob("*.json")` and refused when the result held
any file other than the audit summary. That is a proxy for the invariant the
campaign needs, and it is wrong in both directions.

**It fails open.** `benchmarks/results/scalability/` is empty before the first
campaign seals. In exactly that state the check admits any destination, including
a destination inside the release directory. The first campaign in a fresh checkout
could therefore seal straight into the release directory and pass its own
preflight, which is the seal the message exists to prevent. What makes an
artifact release evidence is the promotion step that re-runs the gate over the
sealed set; a seal written directly into the release directory skips it.

**It fails closed forever.** A release directory is not emptied once a campaign
has been promoted, so from the first promotion onward the check refuses every
later seal. This was measured on the campaign branch rather than argued: with the
five signed statevector envelopes promoted, sealing the MPS single-device
capacity baseline was refused with

```
release evidence preflight failed: scalability directory already contains
promoted JSON; seal into the candidate directory and promote only after strict
audit
```

The destination in that invocation was
`benchmarks/results/smoke/release_candidates/mps_single_gpu_capacity/` -- a
candidate directory, which is what the message asks for. The rule was reading the
state of a directory the command does not write to.

## Decision

### 1. The preflight is given the seal destination and checks that

`readiness_errors` takes `seal_destination` and `release_root`, and refuses a
destination that resolves inside the release root. `environment_errors` passes
the repository's release root and forwards the destination from its caller;
`tools/seal_runtime_evidence.py` passes `args.output`. The standalone
`--seal-destination` option lets a launcher ask the same question before it
measures anything.

### 2. The check resolves both paths before comparing them

The answer has to be about the file the seal would create rather than about the
spelling of the argument. `benchmarks/results/scalability/../scalability/leg.json`
and a symlink into the release directory are both inside it, and a lexical prefix
test accepts both; conversely
`benchmarks/results/scalability_candidates/leg.json` is not inside it, and a
comparison without a path separator would call it one. Both cases have a test.

### 3. `promoted_json` is removed rather than kept beside the new check

Keeping it would preserve the fail-closed-forever behaviour the change exists to
remove, and the new check subsumes the case the old one was reaching for. The one
caller that passed it, `environment_errors`, is the one that now passes the
destination.

### 4. The property is stated directly and holds in both directory states

"No seal is written into the release directory at all" is true before the first
promotion and after every promotion, so one rule covers the campaign's whole
life. That is the difference from the proxy, which was true only in the interval
between two promotions.

## Required evidence

- [x] A destination inside an **empty** release directory is refused:
      `test_a_destination_inside_the_release_directory_is_refused_when_it_is_empty`
- [x] A candidate destination is accepted while the release directory is empty:
      `test_a_candidate_destination_is_accepted_when_the_release_directory_is_empty`
- [x] A candidate destination is accepted while the release directory holds
      promoted JSON:
      `test_a_candidate_destination_is_accepted_when_the_directory_holds_promoted_json`
- [x] A destination spelled through the release directory is refused:
      `test_a_destination_spelled_through_the_release_directory_is_refused`
- [x] A destination in a neighbouring directory whose name starts with the
      release directory's is accepted:
      `test_a_destination_that_merely_starts_with_the_release_directory_name_`
      `is_accepted`
- [x] A destination that resolves through a symlink into the release directory is
      refused:
      `test_a_destination_that_resolves_into_the_release_directory_is_refused`
- [x] The standalone environment check invents no destination error:
      `test_the_environment_check_alone_names_no_destination`
- [x] `tools/seal_runtime_evidence.py` passes the destination it will write:
      the sealer's own call site, exercised by the capacity-baseline seal below
- [x] The check was exercised against the real release directory at the real
      revision: `python tools/check_release_evidence_environment.py
      --world-size 1 --seal-destination
      benchmarks/results/scalability/leg.json` exits 2 naming the destination,
      and the same command with a candidate destination names no destination
      error
- [x] `ruff check` passes on the three touched files
- [x] `tools/check_team_scope.py --team integration --files` passes on the three
      touched files

## Non-goals

- The promotion gate is not relaxed. `tools/promote_release_candidates.py` still
  re-runs the capability's gate over the candidate set with a signing key and
  moves nothing unless it passes, and `benchmarks/audit_results.py
  --require-scalability` still re-validates every promoted payload.
- The sealer's other preflight conditions are unchanged: a full 40-character
  commit, a configured signing key, and at least `world_size` device UUIDs are
  still required.
- No capability's release contract is touched.

## Why this is an integration change rather than a lane change

`tools/**` is a protected integration surface, and the defect is in the shared
release path rather than in the MPS, tensor-network, or statevector lane: the
sealer, the preflight, and the promotion tool are the same three files whichever
capability is being released. Every capability would otherwise have had to carry
its own copy of the rule, which is the duplication the protected surface exists
to prevent.
