# Development artifacts

This directory contains non-benchmark development evidence and provider task
records retained by existing capability work. Files here are not release-grade
performance evidence unless a capability manifest explicitly binds and scopes
them.

Existing root-level JSON files and provider batch directories are legacy
holdings. Do not add new dated or batch directories at this level. New work
should keep only a compact, provenance-bearing summary in the source repository
and place raw responses, repeated task payloads, and bulky intermediates in the
team evidence archive.

Release benchmark evidence belongs under
`benchmarks/results/scalability/` and must pass the release audit. See
[repository governance](../docs/development/REPOSITORY_GOVERNANCE.md) for the
retention and migration policy.
