# Repository test configuration and reference-suite ownership

Owner: Integration. Commit this declaration separately before implementation.
This connected root-ownership slice belongs in the existing root-tool PR #1360,
not a separate PR for each configuration or directory. The resulting head still
requires independent CI/source/scientific/resource qualification.

Consolidate every original pytest/Coverage value in pyproject.toml and remove
pytest.ini and .coveragerc. Preserve all six effective pytest values, marker
ordering/descriptions, coverage sources/omissions/exclusions, default ordered
node multiset and original outcomes. Synchronize every actual configuration
consumer; retain unrelated synthetic INI fixtures and historical evidence bytes.

Move the single early environment authority from root conftest.py to
tests/conftest.py, retaining all six setdefault calls before optional-library
imports and PYTEST_DEBUG_CONFIG behavior. Move the nine explicitly invoked
Qualtran parity test modules and their shared counter into
 tests/reference/qualtran/ (without the leading space). Keep that suite explicit
using pytest's collect_ignore for reference/qualtran: default collection must
remain exact, while explicit directory and file commands must still execute.
No root forwarder, second environment authority or product import side effect.
The historical artifacts/development/torch_training_cpu_20261103 identity fake is
review-only evidence: its documented replay commands run verify.py/reproduce.py,
not pytest on the fake module. Preserve that original artifact unchanged.

The default-CI reachability guard must continue checking every original default
test. Distinguish only the existing explicit reference suite, verify its ignored
path against the actual conftest declaration and require each such test's whole
relocation from its previous tools/qualtran_parity source in the existing ledger.
Do not give old CI tests, arbitrary directories or unknown files an exemption.
Keep every reference test body/input/clock/seed/precision/budget unchanged except
for the two internal shared-helper imports. Compare complete direct-reference
collection and default selection independently; do not claim scientific parity
from collection, missing optional dependencies or unit-only results.

Move its two actual benchmark scripts and README to benchmarks/qualtran and
its two raw JSONL records byte-for-byte to artifacts/benchmarks/qualtran.
Synchronize real docs/imports and the existing source relocation ledger. Delete
tools/qualtran_parity completely; retain no executable aliases or copied bodies.
Historical records retain their original commit/source identity and bytes.
Admit only the approved artifacts/benchmarks canonical container in the layout
guard; preserve every file-size, forbidden-file and original285-result corpus
ratchet value. The two existing Qualtran records were outside that original
corpus before and remain separate; generate no new benchmark data.

No product/native/dependency/runner/resource change is included. Preserve frozen
contracts and all other scientific selections. Require focused original consumer
cases, API comparisons, all static gates, complete original default outcomes,
explicit-reference execution under its stated environment and new-head source,
installed-distribution and whole-job CI evidence. No merge, force push, Rust or
runner administration. This preview does not complete remaining artifacts,
developer distributions, #1084 domains, native ABI or integrated G0.
