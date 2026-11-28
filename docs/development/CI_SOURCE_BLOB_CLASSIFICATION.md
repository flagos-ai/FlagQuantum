# Nullable classification of truncated GitHub source blobs

## Failure and required behavior

PR #1317 introduced a 12987160-byte compressed raw-evidence archive. GitHub's
GraphQL Blob response for its Git object was observed as:

```json
{"byteSize": 12987160, "isBinary": null, "isTruncated": true}
```

The old metadata validation required a Boolean binary classification, so API
source restoration refused the object. The GPU gate's fallback Git fetch then
reported a connection reset, an SSL timeout and cancellation at its fifteen-minute
job budget. Main CI source preparation separately succeeded. This is a source
transport failure before tests, not a compiler verdict or a passing GPU gate.
The original failed/canceled workflow is retained; no blind retry is evidence.

The integration-owned downloader now accepts an explicitly null classification
only when truncation is explicitly true. Such a blob cannot enter the small-text
GraphQL optimization: it goes through the existing REST raw-blob path. The
existing base64 decoding and requested Git object SHA check remain authoritative,
followed by complete tree/commit verification. The change does not infer that
null means text, accept truncated source bytes or relax any identity check.
Missing fields, null classification without truncation, wrong field types and
altered object bytes still refuse.

This small repair adds no dependency, public API, schema, workflow, timeout,
benchmark threshold, test skip or numeric support claim. It supports the existing
user development workflow of submitting source/evidence changes through a PR:
CI must restore the exact requested revision before testing it. It changes no
compiler or numerical runtime behavior and makes no distributed scaling claim.

## Verification and workflow

The new regression first failed twice on the original downloader, for the two
existing signed-commit representations. After repair the complete source-archive
suite passes forty tests. The scenario covers exact file contents, binary bytes,
symlinks, executable mode, commit/tree identity, clean checkout, cache reuse and
restoration after removing one cached blob. Corrupting the REST bytes under the
same recorded SHA still fails. Six malformed-metadata controls retain refusal.

Run the existing contributor path:

```sh
python tools/ci/ci_source_archive.py --help
python -m pytest tests/unit/test_ci_source_archive.py -q
```

Expected: the existing CLI help, then forty source-restoration and rejection
checks pass without network access. The real GitHub metadata observation above
is diagnostic evidence, not a required external dependency for those tests.
Fresh full exact-head CI remains required before integration. The canceled
historical gate must not be represented as having passed this corrected code.

## Local default-tier result

`python tools/ci/ci_tier.py pr-default` completed without timeout: 21561 passed,
796 skipped and two failed in 1154.72 seconds. It is not a full-tier pass.
The first attempt could not collect tests because the shared development venv
lacked declared PyYAML. A separate temporary overlay supplied PyYAML 6.0.4;
no dependency manifest or shared environment was changed.

The QEC retirement subprocess then imported an installed older FlagQuantum
layout because the overlay path omitted this source root. With the source root
explicitly prepended, the unchanged QEC check and all forty source-archive tests
passed together (41 tests, 13.49 seconds). The other full-tier failure reports
historical origin declarations for commits now obtainable from local Git
history. It remains unresolved here; no evidence-origin policy, declaration,
test or API snapshot is changed by this source-transport repair. These local
observations cannot substitute for complete exact-head CI.
