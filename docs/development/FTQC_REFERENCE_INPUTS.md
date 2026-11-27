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

## Isolated source format and lint profiles

The existing dependency policy owns `ftqc-reference-format` and
`ftqc-reference-lint`. Their exact versions reproduce the already qualified
source-tool consumers: Black 24.8.0, flynt 0.78, isort 5.10.1 and astor 0.8.1;
Pylint 3.3.9 with astroid 3.3.11, dill 0.4.0, mccabe 0.7.0, platformdirs 4.5.1,
tomlkit 0.13.3 and isort 5.10.1. Installed license notices were inspected for all
eleven profile entries. These are test tools, not runtime dependencies or new
production extras. Upstream format and Pylint requirement bounds and locked
versions remain unmodified. Existing complete command/diagnostic comparisons
qualify these selected tools; this does not install the entire upstream
scientific `format.env.txt` or `pylint.env.txt` environment.

The reusable `prepare-ftqc-reference-inputs` composite action first prepares the
verified 1,280-file source snapshot, then installs the two small binary-only tool
profiles into separate sibling targets. Both the FTQC comparison lane and CPU
nightly caller use it after their native base dependencies. Explicit `--no-deps`
is limited to these enumerated test targets, with common utilities supplied by
the qualified CI interpreter. It never replaces the primary quality toolchain,
uses source builds or changes production dependency installation.

The existing source-aware consumers read `format-command-deps` and `lint-deps`
and retain actual check/apply output, status propagation, failure combinations,
source copyright edge behavior and original diagnostic equality. Other SDK
profiles, wheel/cache inputs, external graphics tools and complete hosted-suite
execution remain separate prerequisites before Q-0009/Q-0011 can close.

## Open optional-reference collection prerequisite

The complete native dependency/CI regression also runs the existing optional
reference guard. It reports 759 unguarded optional imports across the current
checkout after removing three needless NumPy imports from oracle shape checks.
The same 759 findings occur under the preceding dependency policy: declaring the
two new tool profiles neither introduces nor exempts these findings. The guard,
its test, marker selections and refusal rules remain unchanged. This is an open
G2 integration check, not a passing hosted-suite result or a grandfathered
exception. Full Q-0009/Q-0011 and G2 acceptance require its actual resolution.

## Missing-dependency contract for collected comparisons

`tests/_reference_dependencies.py` is a test-only boundary for literal optional
imports. A guarded import catches only `ModuleNotFoundError` and delegates the
same error object to `unavailable_reference`. Ordinary optional collection
explicitly reports the unavailable dependency as a module skip. When
`FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES` is set to any value other than `0`, the
original error and cause are raised. An unset value is equivalent to `0`.
Unrecognized settings therefore cannot silently enable optional behavior.
Other import errors, numerical errors and source failures are not caught.

The shared FTQC input action sets mandatory mode for both reference CI and CPU
nightly. New absence guards must use this boundary, retain their original
imports/types/math and prove that available SDKs execute unchanged. No marker
selection, source bytes or dependency checker rule is relaxed. Real subprocess
checks cover both presence/absence modes: a required missing SDK fails collection,
while a present SDK runs the comparison in both modes without skips. Migration
of the 759 actual guard findings remains required; this contract alone does not
close that check or certify unrelated historical skips.
