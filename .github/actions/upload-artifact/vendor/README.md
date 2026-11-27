# Pinned upload action

This directory contains the existing official `actions/upload-artifact`
implementation at commit `ea165f8d65b6e75b540449e92b4886f43607fa02`.
The integration owner retains its exact `action.yml`, MIT `LICENSE` and bundled
upload entry point. `PROVENANCE.json` records their Git object IDs and SHA-256
values. This is the same SDK previously resolved by `@v4`, with no API, runtime,
retry, artifact format or dependency-version change.

The parent bounded-retry action uses these local bytes so each test partition
can upload without fetching another action archive from `codeload.github.com`
after its scientific tests finish. The source artifact's existing digest and
Git-tree checks also bind this dependency. Artifact data still uses the original
GitHub service and upload protocol. Persistent failures still fail the last
attempt; job ownership, hidden files, retention and input/output handling stay
with the existing wrapper.

Refresh only from an explicitly selected upstream commit. Verify every raw
file's Git object and SHA-256, retain the original license and metadata, update
the provenance and exact-byte tests, and run the wrapper regression tests plus
real CI upload/download and source-binding checks. Do not edit or reformat the
bundled upstream files. This CI-only dependency must not enter Python wheels
or source distributions; inspect both artifacts when packaging changes.

The official bundle is 5,051,718 bytes. The repository hygiene policy permits
only this exact path up to that size; the artifact tests additionally enforce
its recorded upstream Git and SHA-256 identities. This retains the upstream
implementation rather than rebuilding, trimming or modifying its protocol code.
The integration owner must review both the byte ceiling and provenance on an
upstream refresh. Other files retain the repository's ordinary size ceiling.
