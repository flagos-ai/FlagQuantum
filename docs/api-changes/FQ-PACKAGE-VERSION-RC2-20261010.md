# Package version release identity for 0.3.0rc2

## Authorization

The API owner authorized preparing FlagQuantum `0.3.0rc2` on 2026-10-10. This
release candidate supersedes `0.3.0rc1` without modifying its existing tag or
GitHub prerelease.

## Contract boundary

`fq.__version__` remains a stable root export and a string. Only its
release-specific literal changes from `0.3.0rc1` to `0.3.0rc2`; its name, type,
import path, and purpose remain unchanged.

`flagquantum.version.__version__` remains the single version source. Setuptools
reads it for package metadata, and distribution verification checks the built
wheel and source archive. This decision does not authorize any other Stable Core
change or regeneration of the historical baseline.

## Compatibility scope

Official `0.3.0rc2` wheels are compiled against the PyTorch 2.10 Stable ABI and
support PyTorch `>=2.13,<2.15` at runtime on Python 3.10 through 3.12. This scope
does not claim certification on Torch-FL accelerator hardware or vendor
runtimes.
