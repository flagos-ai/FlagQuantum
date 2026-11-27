# Observable numerical ownership acceptance

Status: formal source and native installed artifact qualified. This child still
requires its own exact pushed-head combined CI. Parent #1338
`d1d1d13b0cabab04b64250d57cd2772457eadb26` passed all applicable checks
(243 successful and 3 existing conditional skips).
Both parent workflows passed on their first attempt.

## Source and initial checks

Nine product files implement five extracted numerical helpers. Seven normalized
AST donor comparisons cover both JAX result families, physical outputs and the
native/eager weighted route. All 82 original function definitions are reviewed;
non-target definitions, public signatures/annotations and all nine product
assertions remain. Existing tests and frozen contracts are unchanged.

The initial focused source run passes 683 tests with 127 existing conditional
skips; eleven new replacement cases pass separately. Initial quality has ten
passing gates and one Ruff failure for two import blocks. Its raw diagnostic
is preserved; import ordering is corrected without changing numerical bodies.
Final preview quality passes all eleven gates: strict mypy covers 1,262 sources,
Black covers 3,665 Python files, and local links resolve across 1,915 Markdown
files. The final related run passes 694 tests with the same 127 conditional
skips, including all eleven replacement cases. Formal regression remains due.

## Original installed-parent differential

The immutable parent contains 222 encoded scenario entries, SHA-256
`72cb205e7836ecc522795ecb96d530eb8cb6e0f925aaf24e6614b52ef0374f76`:

- 192 JAX cached-result cases span both families, real precisions, batch sizes,
  qubit counts, normalization, strided storage and gradient-enabled/disabled
  inputs. Raw probability/Z values, gradients, flags, cache identity, public
  method signatures, annotations and invalid-index refusals are captured.
- Four explicit simulated Torch-admission probes retain dependency refusal
  before the invalid-index check. These are not an uninstalled-dependency test.
- Sixteen real cached weighted-readout cases cover both dtypes, batches,
  native boundary enabled/disabled and cache enabled/disabled, including repeat
  reuse, changed terms and direct native-provider observations.
- Four Ising and six Rydberg presence cases use real public sealed plans and
  replay. Complete physical result fields and plan JSON are captured without
  omitting fields; mapping and dataclass field order is preserved.

The first source-preview comparison has sixteen differences, confined to native
availability and direct native-provider observations. Its mathematical output,
weights and all other entries match. The installed parent has `_C.abi3.so`;
neither the actual source checkout nor its plain copy contains that binary, so
source native availability is false. The original snapshots, full failure and
leaf diagnostics are retained. This does not qualify strict installed native
behavior; availability/path fields are not removed from comparison.

The fingerprint harness initially rejected a one-element strided real Tensor
view when converting its dtype to bytes. It now records original stride/flags
and encodes logical C-order bytes through NumPy. No original baseline output
existed before that harness correction; the first raw failure remains.

## Complete formal source regression

Three phases ran in the actual Git checkout, with four workers and unchanged
optional/hardware policy and credential-dependent Kaiwu live-smoke exclusion.
The source-tool test file is included in this complete source run.

| Phase | Passed | Existing skips | Pytest seconds |
| --- | ---: | ---: | ---: |
| unit-api-team-final | 19425 | 957 | 416.62 |
| qec-full | 2801 | 329 | 94.17 |
| root-local-integration | 1592 | 223 | 106.99 |

All eleven formal quality gates and five additional gates pass. The commit gate
also requires all five pinned source hooks, four final documentation/diff gates
and exact 1,262-product-blob/wheel equality. Existing scientific assertions, all
frozen contracts/fixtures and original evidence remain unchanged.

## Native installed artifact

The fresh wheel SHA-256 is
`ea282c5068daac69ab8776747651f911a9620c31ca9a7c8ce22074f2820d09ee`.
All 1,262 product Python file sets and bytes match source, wheel and installation.
Cold catalog import does not load Torch/JAX. Native CPU/SparseBlossom and the
installed benchmark CLI pass from the installation's own site. Third-party
packages use the parent's borrowed dependency path; this is not a new independent
dependency-resolution certification. Native ABI and source are unchanged.

The same strict **222-entry comparison now passes byte-for-byte**, including
native availability and provider observations. Its SHA-256 remains the original
`72cb205e7836ecc522795ecb96d530eb8cb6e0f925aaf24e6614b52ef0374f76`.
No field, path-selection check or tolerance was removed to resolve the plain
source environment mismatch.

Original parent pickle SHA-256
`bd912e9c53091cbe4dfb055d31f7d1be5df810ede90e2ff1563c3822ba6819e1`
restores two cached public JAX record fixtures, six bound methods, two real
public sealed plans and the original canonical native function reference.
The minimal cached JAX records are interface fixtures, not distributed execution
or scalability evidence. Full plan replay and subsequent readout snapshots match
parent bytes, SHA-256
`3f32e2c975cb41c235905a207c49c127705683c44fcc62edbe2f302dbe027102`.
The old Runtime native alias is absent in this artifact while its original
function pickle still resolves to the unchanged public native owner.

Outside a product checkout, **759 original/replacement tests pass**. All 265
loaded FlagQuantum modules resolve to this artifact's own installed site. The
source-tool file `test_ising_braid_source.py` is retained for complete source
regression. The first installed invocation lacked the source-free fixture root
on the import path during integration-test collection. Its raw log remains;
the corrected launcher adds only the original tests/examples/contracts root,
asserts that it contains no product directory, and validates every loaded
product module after the run. Tests and assertions are unchanged.

## Published-head acceptance

The child requires its own exact-head combined CI. Publication additionally
binds the formal commit to the qualified artifact and enforces final source
hooks/document gates. A green parent or local qualification does not replace
the published-head check.
