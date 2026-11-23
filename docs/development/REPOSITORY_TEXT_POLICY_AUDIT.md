# Complete repository text-policy ownership

The five complete policy files below come from Qualtran v0.7.0 at revision
bcb17bb25360ed68ce3bc084645db3e929799389. Their original declarations retain
Google LLC copyright and Apache-2.0 notices in the frozen reference tree.
The adapted editor configuration retains its original notice. All source Git
blobs are verified before their declarations or Git behavior are exercised.

| Parent | Complete source | Git blob | Native owner |
| --- | --- | --- | --- |
| Q-0001 | .editorconfig | 27247697fffdf501f7701d89054270ac1e0cd4e4 | Root .editorconfig and existing Python formatter configuration |
| Q-0002 | .gitattributes | 6b61e741eb3fc17b850369eff1fc3cdd73bdc45a | Root Git attributes and existing source archive checks |
| Q-0012 | .gitignore | 339809b29bd7be6f2973a6ec1625aa12a72d5e78 | Root and documentation Git ignores, repository hygiene and distribution policy |
| Q-0013 | .markdownlintrc | bbf11d7d4f04eedab5920f66ea287a9b0a611b19 | Existing documentation links, source-of-truth, rendering and whitespace owners |
| Q-0015 | .yamllint.yaml | a85c84e188c4188e85c146ff2c2cb22c64fc9c80 | Existing check-yaml hook and domain-specific configuration consumers |

## Interfaces and responsibility

These files consume repository-relative paths and text. Their outputs are editor
properties, Git selection and attributes, or contributor diagnostics. They have
no register interfaces, bit order, quantum outputs, measurement branches or work
ownership. Actual added primitive gates, measurements and workspace are all zero.
They declare no quantum resource model. Existing algorithm amplitude and
resource receipts remain attached to their algorithm owners.

The integration team owns root configuration. Tooling tests exercise consumers
in disposable repositories, with global and system Git configuration disabled.
Reference configuration remains outside the runtime package. The user workflow
continues through existing circuit.append(op) and development commands.

## Complete disposition

### Editor properties

The source UTF-8, spaces, final-newline, English spelling, trailing-whitespace
and 100-column defaults are retained. All Bazel, C++, JavaScript/TypeScript,
JSON, Python, reStructuredText, shell and YAML indentation sections are retained.
Python adds an 88-column override matching the existing Black/Ruff owner.
Existing evidence and byte-bound drawer license/runtime exclusions are represented
in the editor configuration as well as the existing pre-commit hooks.

The test parses the complete source and native declarations and compares every
source section and property. This is declaration verification. Editor-plugin
behavior remains dependent on each editor's EditorConfig support. The existing
formatter orchestration tests execute the actual native Ruff and Black path.

### Git attributes

The generated Python protocol implementation and stub patterns now set
linguist-generated=true. Existing automatic text normalization remains the native
owner. Actual git check-attr and index staging prove generated classification
at root and nested paths, authored-code nonclassification and LF normalization.
The original generated-only attributes retain CRLF in the isolated source
fixture. This deliberate native normalization difference is recorded explicitly.

### Ignored products and publication

Every active original ignore directive is exercised through actual Git using a
matching concrete path. This covers Python/extension products, installers,
coverage, translations, web-framework outputs, notebooks, interpreter state,
environments, IDEs, generated documentation and the original generated test
assets. Commented Pipfile.lock remains authored input. Original paths are
validated in the frozen policy fixture, not copied wholesale into the product.

Native cache, distribution, execution, private research and scratch ownership
stays with existing ignores and repository hygiene. The isolated native consumer
checks ignored artifacts alongside retained environment templates, recorded raw
evidence logs, source, documentation and JSON fixtures. Explicitly tracked
content remains tracked even when its filename matches an ignore rule. The
existing documentation output and actual source archive tests verify publication
and byte-preserving restoration separately. Original application-specific output
names have no native producers and their patterns are retired.

### Markdown and YAML contributor conventions

The complete original JSONC is parsed with the existing reference JSON5 consumer,
including schema URLs containing double slashes. The capture retains all ten
configuration keys, ATX headings, first-line-heading exemption, punctuation
selection, zero trailing spaces, 100-column exceptions, list-marker spacing,
four-space nested indentation, bare URLs and all 56 allowed HTML elements.
The complete YAML declaration retains the 100-column rule and both unbreakable
word/mapping exemptions.

The native repository keeps its existing Markdown layout and YAML conventions.
The upstream list spacing, punctuation, HTML allowlist and 100-column lint rules
are retired as product gates. Native document rendering, links, anchors,
source-of-truth and whitespace checks own document acceptance. Existing YAML
syntax and typed domain consumers own configuration acceptance. No standalone
markdownlint or yamllint gate is installed. JSONC/YAML declaration capture does
not constitute execution of those upstream linters or an HTML security filter.
The functional native consumer accepts a valid linked document and reports both
a missing target and a missing anchor. YAML parsing accepts the real hook
configuration and rejects malformed input. Existing formatter, lint-command,
source archive and documentation tests supply their complete consumer evidence.

## Verification

Run the focused source-policy and consumer checks in the locked G2 environment
with tests/unit/test_repository_text_policies.py. Frozen dependencies and Node's
existing JSON5 reference consumer are required, with missing tools causing a
failure. Original source policies are immutable. Receipts are stored under
ftqc-reference-env/g2-text-policy-*20261123 beside the worktree.

This acceptance concerns complete declarations and their explicitly adapted
native owners. The hosted documentation site, full scientific environment locks,
upstream numerical discrepancies and unapproved API proposals keep their own
inventory entries and acceptance requirements.

## Batch result and retained failures

The eight focused policy checks pass. Strict typing, Black, Ruff, Integration
scope, repository language, architecture, multilevel IR and protected API
checks pass. The dependency-policy findings are the identical 720-entry baseline.
All 121 construction and coherent-execution integration checks pass.

The broader 164-check selection reports 162 passes and two existing hygiene
failures. Five historical source fixtures exceed the existing 1,800,000-byte
tracked-file limit. A second historical test treats the inline code references
T:21 and T:7 in RESOURCE_FLAME_GRAPH.md as links. The original test/checker,
all five fixture payloads and the document are byte-identical to the pre-batch
commit. Their hashes and full failure output are retained in the existing-failure
receipt. These remain G2 repository-quality gaps. The source-policy acceptance
covers the declarations and consumers described above, with these broader
failures recorded separately. No file-size allowance or failing test was relaxed.
