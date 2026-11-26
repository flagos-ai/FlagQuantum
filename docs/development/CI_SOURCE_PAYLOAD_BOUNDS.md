# Bound CI source object responses

CI source preparation can exhaust its job budget before compiler tests start.
For PR #1298 head `a7a89dce53a4fa89b9a2e4954d1e20b32c313c54`,
[run 45006524928](https://github.com/flagos-ai/FlagQuantum/actions/runs/45006524928)
spent ten minutes restoring API objects. The fallback Git fetch then reported
TLS disconnects and the job was canceled at fifteen minutes. Its dependent
aggregate jobs failed without testing the compiler. The
[retained record](evidence/ci-source-payload-bounds.json) includes the relevant
log lines and immutable source probes. This is not evidence of a compiler
semantic failure, nor proof that payload size caused every transport failure.

## Source restoration change

The existing GraphQL request included full text for as many as eight missing
objects. A batch of eight large retained evidence files could therefore carry
several megabytes of text through one response, with every transient failure
retrying that response. The fallback also fetched objects sequentially after a
failed batch.

The Integration-owned helper now reads object metadata first. Nonbinary,
untruncated blobs of at most 32 KiB remain eligible for a second batched text
request; at most eight contribute 256 KiB of UTF-8 text. JSON transport overhead
is additional, so this is a text-content bound rather than a total wire-size
bound. Larger blobs, binary blobs and truncated blobs use independent REST
requests with at most eight workers. Transport fallback also uses independent
concurrent requests. Mutable response metadata is rejected between the size and
text requests. No application cache or new dependency is introduced.

The existing per-request timeout, maximum two retries, job timeout, authentication
refusal behavior and exact SHA checks remain unchanged. Tree reconstruction,
commit reconstruction, executable modes, symbolic links and clean checkout still
use their existing verification boundaries. Individual blob responses are not
newly size-capped; this change bounds batched text rather than claiming to bound
all possible source files or total restore wall time. Source acquisition can
still fail, and tests must not be reported as passed when it does.

## Verification and practical limits

All 132 focused source-object, archive, transport, unsigned/signed commit and
artifact tests pass. The extended restoration scenario includes 96 small texts,
four large texts, a binary file, an executable and a symbolic link. It verifies
complete content and Git identity, proves that no large text enters GraphQL,
and checks the 256 KiB text limit. It rejects tampering and changed metadata.
All four new workflow/parallel/metadata controls fail against the exact baseline
method and pass with the candidate.
A bounded two-request barrier makes sequential fallback fail and confirms that
independent requests run concurrently while preserving result order.

The old count assertion allowed 15 GraphQL queries for a different request shape.
Metadata preflight deliberately adds a second bounded query, so the new complete
scenario permits at most 28 GraphQL queries and requires exactly five individual
binary/large-blob fetches. This replaces a request-shape assertion with explicit
request-count and payload bounds; it does not change CI timeouts or source
identity acceptance. The initial implementation passed 100 tests and failed that
old count assertion; the failure is retained in the evidence record.

One read-only probe selected the eight largest evidence blobs added or changed
by the affected branch, totaling 10,931,815 bytes. The GitHub CLI transport
completed in 2.583 seconds; the helper's native urllib transport completed in
3.231 seconds on the macOS development host. Both independently recomputed all
eight Git blob SHA-1 values. Exact source hashes, helper text and drivers are
retained. Blob identifiers are explicitly qualified as `git:blob:<sha1>`;
reconstruction instructions and original report SHA-256 digests preserve the
probe representations without mislabeling blob hashes as commit revisions.
These are single probes through different transports, not a speedup
comparison, the affected self-hosted Linux runner, or complete CI qualification.
No machine, service, driver or kernel was restarted or changed.

Black, Ruff and strict Linux-target mypy pass the complete changed Python files.
All 17 focused repository hygiene and evidence-vocabulary tests also pass.
The first mypy run also exposed an existing unvalidated `Request.data` type in
the test bootstrap; an explicit bytes assertion fixes that typing issue without
changing the bootstrap's acceptance. The required full `python tools/ci/ci_tier.py pr-default` attempt reached its
300-second supervisor budget at approximately 41 percent and was terminated;
it is not a passing full-suite result. Exact-head GitHub CI remains required
before integration.

## Existing workflow and reproduction

This tool supports ordinary development and CI for existing public workflows:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

result = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert result.metadata["routing"]["mapping_restored"] is True
```

Expected: equivalent, topology-legal compilation with restored placement. The
fix changes how CI obtains the exact source for testing this workflow; it does
not modify compilation or SDKBench acceptance.

```sh
python -m pytest tests/unit/test_ci_source_graphql.py \
  tests/unit/test_ci_source_archive.py tests/unit/test_ci_source_network.py \
  tests/unit/test_ci_source_unsigned.py tests/unit/test_ci_source_artifact.py -q
```

Expected: 132 passing tests without credentials, remote jobs, containers or
hardware. Live probe drivers have local development paths and require existing
GitHub read access; they are diagnostic scripts, not the ordinary test command.
