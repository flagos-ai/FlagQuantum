# Package version release identity for 0.3.0rc2

## Authorization

The API owner explicitly authorized preparing FlagQuantum `0.3.0rc2` on
2026-10-10. This release candidate supersedes `0.3.0rc1` without modifying the
existing tag or published artifacts.

## Contract boundary

`fq.__version__` remains a stable root export and a string. Only its
release-specific literal changes from `0.3.0rc1` to `0.3.0rc2`; its name, type,
import path, and purpose remain unchanged.

`flagquantum.version.__version__` remains the single version source. Setuptools
reads it for package metadata, and distribution verification checks the built
wheel and source archive. This decision does not authorize any other Stable Core
change or regeneration of the historical baseline.

## Compatibility scope

Official `0.3.0rc2` wheels target the PyTorch `2.10.x` native ABI and Python
3.10 through 3.12. This scope aligns the release with Torch-FL's supported
PyTorch line and does not claim certification on Torch-FL hardware.
