# Docs tools

Owns source-based catalogs, operator documentation and link validation. Product behavior belongs in the owning FlagQuantum domain; these
tools inspect, validate or package it. Optional SDKs and live hardware remain
explicit prerequisites of the existing individual commands.

Canonical Python imports use `tools.docs`. Existing `python tools/<name>.py`
commands remain forwarding entry points for current CI and #1084 consumers.
Both paths execute one implementation. New commands belong in this directory.
See the [migration ledger](../../docs/development/INDEPENDENT_STRUCTURE_MIGRATION.md)
for the owner and retirement gate of old command entries.

For a ten-minute change, inspect one command and its existing failure fixtures,
change its owning implementation, then run from the repository root:

```sh
python -m pytest tests/unit/test_docs_source_of_truth.py -q
python tools/check_architecture.py
```
