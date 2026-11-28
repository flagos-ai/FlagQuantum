# Ci tools

Owns test selection, coverage, bounded subprocess execution and exact source restoration. Product behavior belongs in the owning FlagQuantum domain; these
tools inspect, validate or package it. Optional SDKs and live hardware remain
explicit prerequisites of the existing individual commands.

Canonical Python imports use `tools.ci`. Invoke scripts directly in
this directory or through their canonical module names. Historical root command
wrappers have been removed; CI, hooks and callers use these implementations.
See the [retirement inventory](../../docs/development/migration/internal-forwarder-retirement.md)
for removed paths and the separately retained #1084 integration inputs.

For a ten-minute change, inspect one command and its existing failure fixtures,
change its owning implementation, then run from the repository root:

```sh
python -m pytest tests/unit/test_ci_source_archive.py -q
python tools/architecture/check_architecture.py
```

The CI ruleset template, locked FlagOS CUDA-reference environment and unverified
domestic-card attestation template live here as JSON data. Their contents do not
authorize remote settings or hardware claims. Keep historical contract identifiers
resolved through the existing source-relocation ledger.

For the data-only acceptance path, run from the repository root:

```sh
python tools/validation/check_domestic_single_card_contract.py
python tools/validation/validate_flagos_reference_evidence.py artifacts/flagos_cuda_reference_a800_20260824.json
```

Both commands validate existing data without measuring hardware or submitting jobs.

The remaining root-tool implementations have moved to their owners. The source bootstrap remains a standalone standard-library program. Pre-checkout downloads and incremental packs include the complete root/tools/ci ancestor chain even when the ci subtree is unchanged versus the pack base.

```sh
python -I tools/ci/ci_source_archive.py --help
```

See the [root-tool Integration declaration](../../docs/development/migration/root-tool-ownership.md). Incoming #1084 semantic changes remain separately unqualified.

Coverage aggregation uses `ci_coverage_artifacts.py` on the host interpreter to
start at most eight REST artifact transfers. It publishes only the exact current
run/source shard set after SHA-256 and safe ZIP verification; the original
SQLite integrity, combine and coverage-policy steps remain mandatory. The
locked offline Python wrapper continues to perform coverage computation without
network access. Inspect options without downloading anything:

```sh
python3 -I tools/ci/ci_coverage_artifacts.py --help
```

See the [transport repair declaration](../../docs/development/migration/ci-coverage-artifact-transport.md).
