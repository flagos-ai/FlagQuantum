# Development artifacts

This directory contains non-benchmark development evidence and provider task
records retained by existing capability work. Files here are not release-grade
performance evidence unless a capability manifest explicitly binds and scopes
them.

Root-level JSON files are reserved for current capability records referenced by
the capability matrix or release documentation. Historical Quafu batches and
task fragments live under [`legacy/`](legacy/README.md); new compact,
non-release outputs use [`development/`](development/README.md). Place raw
responses, repeated task payloads, and bulky intermediates in the team evidence
archive.

Release benchmark evidence belongs under
`benchmarks/results/scalability/` and must pass the release audit. See
[repository governance](../docs/development/REPOSITORY_GOVERNANCE.md) for the
retention and migration policy.
