# Isolated CI source-pack staging

The source producer now creates one private temporary root for each invocation
with `mktemp -d`. Full and incremental packs occupy separate subdirectories;
the upload steps read their paths from the producer's outputs. Artifact names,
producer-attempt keys, consumer download paths, Git object verification and
source admission remain unchanged. A final `always()` step removes only the
private root, including when packing or upload fails.

## Failure evidence

PR #1281 run `44963275312`, prepare-source job `137026741978`, restored and
verified workflow commit `e4e3b8c5fc1aa6495d2b6e8458b625f2eb5c2081` on runner
`fq-bw1100-16-cpu11`. At `2026-11-21T23:45:45Z`, packing failed before tests:

```text
mkdir: cannot create directory '.../_work/_temp/ci-source': File exists
```

The fixed producer path could already exist after a previous producer or
consumer job. Reusing that directory with `mkdir -p` would risk mixing its
contents with the new upload. The private root avoids that collision and keeps
previous data intact. Both CI and local-GPU workflows contained the same
producer script and receive the same correction. No host directory is removed
as a repair action; no machine or service is restarted.

Complete downloaded job-log SHA-256: `78e563e7ed168110eb5f08fcbf88c3cc7ad195bb838ce35dc27c96ed3b4b9478`.

## Behavioral verification

```bash
python -m pytest tests/unit/test_ci_source_artifact.py -q
```

Expected: 27 passes. Sixteen parameterized complete source workflows execute
the actual packing and consumer shell scripts with real local Git objects.
They begin with retained shared directories, run the producer twice, verify
that the two roots differ and the first pack is untouched, then restore binary,
symlink and ordinary source files into clean and dirty consumer checkouts.
Both the incomplete-cache probe and complete-cache path run; a mismatched
commit remains rejected. Successful and deliberately failed producer roots are
cleaned while the pre-existing directories remain intact.

The first local test collection lacked PyYAML; adding the existing cached test
dependency resolved it without changing dependencies. The first sparse default
tier selected 213 tests: 205 passed and eight failed because the tracked
`contracts/pytest-duration-hints.json` input was absent. That unmodified input
was restored before rerunning; all 213 selected tests then passed. This is a
sparse tooling selection, not full-repository default-tier certification. Strict checking initially found missing PyYAML
stubs and one existing unannotated test; cached stubs and a `None` return
annotation resolve both without suppression. Black, Ruff and strict mypy pass
for the changed Python test file. Full exact-head Linux CI remains necessary.

This supports the existing local compiler workflow:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

CI must first deliver the exact source that tests that workflow. This change
makes no compiler-quality, timing, resource, scalability or SDK comparison claim
and does not waive any test or hardware performance gate.

Completed sparse default-tier log SHA-256: `e0f85f5c98b0ebc7dae2fea3e774ffab5294b1910879dee4123bd611f24de774`.
