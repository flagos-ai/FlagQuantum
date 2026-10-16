# Package version release identity for 0.3.1

## Authorization and scope

The API owner requested preparation of a distinct `0.3.1` distribution on
2026-10-16 after the approved research services in #632 merged later than the
published `0.3.0` artifact. This is release preparation, not authorization to
overwrite an existing artifact or bypass the release checks. Issue #633 tracks
the exact release decision and installed-artifact verification.

The service API was reviewed and approved in the research-capability-services
change. This version change does not alter its signatures, schemas, or maturity.
It makes those additive services available in a new backward-compatible
distribution while preserving the existing 0.3.0 API.

## Contract boundary

`fq.__version__` remains a stable root export and a string. Only its
release-specific literal changes from `0.3.0` to `0.3.1`; its name, type,
import path, and purpose remain unchanged.

`flagquantum.version.__version__` remains the single version source. Setuptools
reads it for package metadata, and distribution verification checks the built
wheel and source archive. This decision does not authorize any other Stable Core
change or regeneration of the historical baseline.

## Release gate

The exact `0.3.1` main revision must pass every required check before tagging.
The release pipeline must verify the supported Python 3.10–3.12 and Linux
x86_64, macOS arm64, and Windows amd64 `cp310-abi3` artifacts, including the
approved research service modules and public exports. A locally built wheel or
source checkout alone is not evidence that a registry distribution is usable.
