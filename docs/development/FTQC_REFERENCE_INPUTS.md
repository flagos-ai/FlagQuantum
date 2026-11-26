# Frozen reference inputs for local and hosted tests

Integration owns source acquisition and validation through
`tools/ftqc_source_inputs.py`. Verification consumers read the existing native
`docs/development/ftqc_source_inventory.csv`; it is the sole source of file IDs,
paths, revisions and Git blob identities. The personal migration-progress CSV is
not a test dependency. Its 1,280 ordered ID/path/blob rows compare exactly with
the native inventory before consumer migration.

## Interface and ownership

`read_inventory` validates every declared source record without filtering on the
historical availability labels. `SourceFile` is an internal validated CSV row,
not a runtime or public program contract. `prepare_sources` consumes one source
archive per declared revision and a destination. `verify_sources` requires every
declared file's exact Git blob. File bytes and source executable bits come from
the original archive; no reference code is patched or normalized.

The two source revisions are Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` and its later Kickmix subtree at
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. Both archives come from the public
`quantumlib/Qualtran` repository. Downloading uses no credentials, a bounded
request and a 64 MiB archive limit. Staged declared sources have a 512 MiB limit.
Final copying must leave at least 1 GiB free. Existing destinations are verified
read-only; changed or incomplete ones refuse rather than being repaired.

New snapshots are checked in private staging, copied into an exclusively created
destination and verified again. A failed new copy removes only that newly owned
destination. Unsafe paths, duplicate identities/archive entries, nonregular
source members and changed hashes refuse. Archives are read as data; no archive
member executes during preparation. Standard-library code keeps this step
independent of Torch, Qualtran, Kickmix and optional developer SDK installation.

From a prepared source checkout, the standalone command is:

```console
python tools/ftqc_source_inputs.py \
  --archive-cache /tmp/fq-reference-archives \
  --destination /tmp/fq-reference-sources
```

It prepares source files only. Reference tests' normal sibling directory is
`../ftqc-reference-env`; use that destination only for a fresh CI checkout.
The existing local G2 reference environment contains independently owned SDKs
and evidence and must not be replaced. Standalone C++ oracle generators use
`--inventory docs/development/ftqc_source_inventory.csv` where they accept an
explicit inventory. They retain their direct-script invocation and numerical
source bodies.

## Acceptance boundaries

This source owner supplies the complete declared source snapshots. It does not
install a scientific environment or create a substitute for missing Jupyter
templates, multiple SDK profiles, compiler toolchains or local helper artifacts.
Those consumers must be audited and supplied before the complete nightly and PR
parents Q-0009/Q-0011 can close. Missing references must not be hidden by marker
exclusions or silent skips. No hosted workflow, notebook refresh, security scan,
upload or G3 gate is run by source preparation.

The CSV migration changes only equivalent provenance inputs. Original reference
bytes, mathematical test bodies, gate/operand/feedback assertions and stored
numerical corpora remain unchanged. Source comparison imports use each object's
defining module with runtime identity checks against the original reexports.
Tests that intentionally exercise runtime acceptance beyond external annotations
retain those original inputs at an explicit dynamic SDK boundary. Concrete
constructor unions and shape/type assertions preserve exact oracle calls.
Tests exercise snapshot verification,
unchanged existing trees, late corruption, refusal/rollback, exclusive ownership,
download limits and low-space refusal. Source bytes and operational environment
qualification remain distinct evidence.

The module has no quantum registers, bit order, complex amplitude, primitive
stream, analytical gate model, measurement history or work-qubit ownership.
Algorithm and runtime owners retain those responsibilities and their evidence.

## Installed notebook template data

`tests/unit/notebook_data.py` owns the three test consumers' template location.
It reads the installed nbconvert distribution's RECORD and resolves its declared
Markdown template root, covering both pip prefix and target layouts. Reading
RECORD directly preserves evidence of missing declared files that
`Distribution.files` can filter out. Missing, absent or ambiguous declarations
refuse; templates are neither copied nor synthesized.

Notebook execution, command-line export and widget conversion use this same
installed data owner. Existing kernel isolation, cleanup, subprocess deadlines,
complex-state/append checks and local documentation rebuild consumers remain
unchanged. This establishes data location, not a substitute SDK version/profile
qualification. No scientific packages, hosted job or ReadTheDocs build are
installed or launched by this helper.
