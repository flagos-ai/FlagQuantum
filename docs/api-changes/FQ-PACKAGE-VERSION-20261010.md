# Package version release identity

## Authorization

The API owner explicitly authorized publishing FlagQuantum `0.3.0rc1` on
2026-10-10. The release changes the value of the existing stable
`fq.__version__` export without changing its name, type, import path, or purpose.

## Contract boundary

`fq.__version__` remains a stable root export and a string. Its literal value is
release metadata and must change when a new distribution is cut, so the
immutable v0.2 migration baseline cannot require every later release to report
`0.2.0`. The Stable Core checker excludes only this release-specific literal;
all other retained exports continue to be compared with the reviewed baseline.

`flagquantum.version.__version__` remains the single version source. Setuptools
reads it for package metadata, and distribution verification checks the built
wheel and source archive. This decision does not authorize any other Stable Core
change or regeneration of the historical baseline.
