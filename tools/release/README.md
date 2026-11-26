# Release tools

Owns artifact manifests, evidence promotion and installed-distribution verification. Product behavior belongs in the owning FlagQuantum domain; these
tools inspect, validate or package it. Optional SDKs and live hardware remain
explicit prerequisites of the existing individual commands.

Canonical Python imports use `tools.release`. Invoke scripts directly in
this directory or through their canonical module names. Historical root command
wrappers have been removed; CI, hooks and callers use these implementations.
See the [retirement inventory](../../docs/development/migration/internal-forwarder-retirement.md)
for removed paths and the separately retained #1084 integration inputs.

For a ten-minute change, inspect one command and its existing failure fixtures,
change its owning implementation, then run from the repository root:

```sh
python -m pytest tests/unit/test_source_relocations.py -q
python tools/architecture/check_architecture.py
```
