# Complete circuit test helper source audit

## Ownership and interface before implementation

Frozen Kickmix util/circuit_testing.test.cc and .h belong to K-0293/K-0294.
They are test support owned by Verification. Existing Compiler KmxCircuit owns
parsed operations and binary encoding. Existing Drawer rendering remains the
single diagram implementation. Native test helpers consume those owners.

The helper group includes indentation normalization, newline-only splitting,
bytewise mismatch diagnostics, the complete all-operation circuit factory and
its two original test scenarios. Native assertions raise AssertionError with
the original diagnostic bytes instead of reporting a nonfatal GoogleTest part.
This ownership adaptation preserves bytewise evidence, including UTF-8 fragments.
No production API, quantum execution route or resource model is added.

Source/header/dependency blobs must be verified before compiling unchanged
original definitions and original tests. Existing diagram/KMB execution evidence
is reused after verifying the complete helper's actual circuit payload.

## Complete definition disposition

| Original definition | Native owner and verification |
| --- | --- |
| static undent | tests.unit.circuit_diagram_fixtures.undent retains newline trimming, space-only indentation and carriage-return handling |
| static split_lines | split_lines uses newline-only splitting, including trailing empty lines |
| expect_circuit_has_text_diagram | Existing render_circuit_text plus bytewise diagram_failure and a scoped native assertion |
| circuit_with_every_operation | circuit_with_every_operation constructs a fresh immutable KmxCircuit from the verified original payload |
| Original unindent test | All original cases and exhaustive small whitespace strings agree |
| Original operation-coverage test | All 39 effective opcode bytes, guarded variants, six debug forms, nested scopes, CSWAP and exact phases are present |
| Header declarations | Both declarations map to reusable native test helpers with no production exports |

All 139 source/dependency blobs are verified before compilation. The wrapper
includes the original .cc unchanged, preserving both original registered tests
and its private static functions. Both original tests pass. All 1,081 source
observations compare normalization, newline splitting, successful assertions,
complete mismatch bytes and the actual whole-operation factory. Its complete
ASCII diagram and 746-byte KMB encoding match native output. The source corpus
regenerates byte-for-byte. Existing ASCII diagram consumers use the new helper,
with direct rendering retained for the original defect described below.

The complete 174-test diagram/object/KMB/command concentrated batch passes.
The helper owner has 100% statement and branch coverage. Strict mypy passes all
four changed Python files. Black, Ruff, Verification scope, architecture,
internal IR, protected API baseline and whitespace checks pass. Prior 55 local
register integration tests and whole-repository dependency findings are reused
from the immediately preceding production batch. No production dependency or
numerical execution code changed in this batch.

## Original defect and native adaptation

The original expectation helper normalizes a diagram by removing its minimum
indentation. A purely classical circuit renders with a mandatory leading space.
Passing that exact diagram as its expected value still fails after normalization.
The original wrapper reproduces this with BIT_STORE0, guarded BIT_INVERT and
PUSH_CONDITION/POP_CONDITION. Native raises with the same diagnostic payload.
The existing direct diagram comparison continues covering this case.
**K-0293 stays open** for this original normalization defect. K-0294's complete
declaration scope is accepted. The frozen original sources remain unchanged.

GoogleTest's failure-expression prefix and final separator newline belong to
its reporting framework. Native AssertionError carries the exact underlying
failure bytes. Original nonfatal test-part reporting becomes a raised native
assertion. Byte storage deliberately preserves source UTF-8 diagnostic behavior.
Source parsing, rendering and C++ dependency implementation breadth retain their
separately inventoried parents.

The helper emits zero quantum primitives, measurements and work bits. Its factory
is a test circuit containing all physical opcode forms. Exact primitive records
and encoded bytes are separate from the circuit's existing analytical resource
models. Full complex amplitudes, measurement feedback and clean-work behavior
reuse the already accepted physical instruction and register integration evidence.

## Reproduction

```bash
python tests/unit/fixtures/generate_circuit_testing.py \
  --source-root /path/to/frozen/kickmix/src \
  --gtest-root /path/to/frozen/googletest \
  --output /tmp/circuit_testing.json
cmp tests/unit/data/circuit_testing.json /tmp/circuit_testing.json
python -m pytest tests/unit/test_circuit_diagram_fixtures.py \
  tests/unit/test_kmx_circuit_text.py tests/unit/test_kmx_circuit_object.py -q
```

The generator targets the measured macOS/clang host and uses linker dead stripping
for unused CircuitBuilder dependency routes. Raw source-run, build, regeneration
and regression receipts are retained alongside the main ledgers under
ftqc-reference-env/g2-circuit-test-helper-*20261122.*.
