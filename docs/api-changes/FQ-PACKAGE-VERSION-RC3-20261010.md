# Package version release identity for 0.3.0rc3

## Authorization

The API owner authorized preparing FlagQuantum `0.3.0rc3` on 2026-10-10 after
the cross-platform release-wheel corrections passed on `main`. This release
candidate supersedes `0.3.0rc2` without modifying its existing tag, GitHub
prerelease, or PyPI distribution.

## Contract boundary

`fq.__version__` remains a stable root export and a string. Only its
release-specific literal changes from `0.3.0rc2` to `0.3.0rc3`; its name, type,
import path, and purpose remain unchanged.

`flagquantum.version.__version__` remains the single version source. Setuptools
reads it for package metadata, and distribution verification checks the built
wheel and source archive. This decision does not authorize any other Stable Core
change or regeneration of the historical baseline.

## Compatibility scope

Official `0.3.0rc3` wheels are compiled against the PyTorch 2.10 Stable ABI and
support PyTorch `>=2.13,<2.15` at runtime on Python 3.10 through 3.12. The
release pipeline verifies Linux x86_64, macOS arm64, and Windows amd64
`cp310-abi3` artifacts. This scope does not claim certification on Torch-FL
accelerator hardware or vendor runtimes.
