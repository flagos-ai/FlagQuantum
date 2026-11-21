# Development context bundle source audit

## Scope and ownership

Q-0091 and Q-0092 cover the complete context-packaging README and driver from
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`.
The unchanged source blobs are checked against the migration ledger.
`tools/development/context_bundle.py` owns source selection and process invocation.
Existing reference-page and notebook-export tools own rendered documentation.
Compiler and Simulation retain all component and numerical responsibilities.

The twenty source selections map in order to `SOURCE_GROUPS`. Each group lists
its split native owners. The flattened selection contains 116 unique files.
Addition, multiplication, comparison, Hadamard, Toffoli, allocation, join, QROM,
uniform/alias preparation, QFT, Hamming phasing, three block encodings,
qubitization, QPE, THC preparation, Hubbard SELECT and conjunction all have
explicit mappings. This acceptance closes the packaging source and its README.
Underlying algorithm acceptance continues under its existing ledger parents.

| Frozen source selection | Native group |
| --- | --- |
| `arithmetic/addition.py` | `integer-addition` |
| `arithmetic/multiplication.py` | `integer-and-fixed-point-multiplication` |
| `arithmetic/comparison.py` | `integer-comparison` |
| `basic_gates/hadamard.py` | `hadamard` |
| `basic_gates/toffoli.py` | `toffoli` |
| `bookkeeping/allocate.py` | `workspace-allocation` |
| `bookkeeping/join.py` | `register-join` |
| `data_loading/qrom.py` | `qrom` |
| `state_preparation/prepare_uniform_superposition.py` | `uniform-preparation` |
| `state_preparation/state_preparation_alias_sampling.py` | `alias-preparation` |
| `qft/qft_text_book.py` | `textbook-qft` |
| `rotations/hamming_weight_phasing.py` | `hamming-weight-phasing` |
| `block_encoding/unitary.py` | `unitary-block-encoding` |
| `block_encoding/linear_combination.py` | `linear-combination-block-encoding` |
| `block_encoding/product.py` | `product-block-encoding` |
| `qubitization/qubitization_walk_operator.py` | `qubitization-walk` |
| `phase_estimation/text_book_qpe.py` | `textbook-phase-estimation` |
| `chemistry/thc/prepare.py` | `thc-preparation` |
| `chemistry/hubbard_model/qubitization/select_hubbard.py` | `hubbard-select` |
| `mcmt/and_bloq.py` | `conjunction` |

## Runnable path

Render reference pages with the existing reference-generation commands and
notebook Markdown with `tools.development.notebook_execution` before packing.
Use the existing generated-directory cleanup tool for a fresh generated-doc
build. Authored documentation remains governed by its own ownership policy.

Install the optional development tool with an exact version in an isolated
location, using Node 22 or later. The acceptance environment uses Node 26.8.2
and Repomix 1.18.1 installed with lifecycle scripts disabled. For example,
`npm install --prefix /tmp/fq-context-tools --ignore-scripts --no-audit --no-fund --save-exact repomix@1.18.1`.
Then run `python -m tools.development.context_bundle --repomix /tmp/fq-context-tools/node_modules/.bin/repomix --output /tmp/flagquantum-context.md`.
A relative destination is resolved by the child from the checkout root.
The default is `tools/development/context/flagquantum-context-DATE.md`.

## Complete comparison and actual execution

`tests/unit/test_development_context_bundle.py` executes the untouched original
script and checks its twenty paths, date, arguments, Git discovery and failures.
An isolated Git checkout runs both original and native selections through the
actual installed packer. A local npx shim forwards the original invocation to
the exact installed executable. Every selected file and its contents match
under the explicit twenty-owner fixture mapping. Ignored reference Markdown,
nested notebook Markdown and stale Markdown are included. Other extensions
are excluded. Repeated packing preserves the same file-content selection.

The complete native checkout was packed to an external temporary destination.
All 116 selected native files were verified in the output, alongside existing
docs Markdown. The run collected 1,423 total files and produced 14,069,168 bytes
before this audit was added. The external JSON receipt records its SHA-256,
selected files, exact dependency versions and lock hash. Output size and token
metrics are classical packaging measurements. The tool emits no quantum gates,
allocates no quantum workspace and performs no measurements. Existing accepted
component evidence is reused.

## Source behavior and explicit adaptations

The source invokes unpinned `npx repomix` during import and has no child timeout.
The native module uses a main guard, an explicit installed executable, existing
process/root helpers, input existence checks and a 1,200-second packing bound.
It returns the packer's status to the CLI. Timeout and executable errors
propagate. Its selected-file existence check prevents silent owner omissions.

Both commands forward `--no-gitignore`, comma-separated includes and all
`docs/**/*.md`. Old Markdown is retained by this selection rule. The source
README's suggested broad ignored-file deletion maps to the existing narrowly
owned generated-documentation cleanup workflow. Packing itself deletes nothing.

No output style flag is added. Repomix 1.18.1 defaults to XML-style text despite
the `.md` suffix. The text has a prose preamble and raw file bodies, so it is
not a well-formed XML document. Tests compare packer file sections directly.
Native split ownership changes path labels and repository summaries. Byte-level
whole-document equality across differently named repositories is outside this
mapping contract. Repository/global Repomix configuration and its default
security filtering continue to apply. A filter may omit flagged files and the
packer reports those omissions. An atomic whole-output write is not promised.

## Dependency and maintenance review

The tool is an optional external development subprocess. No framework package
imports Repomix and no production dependency is added. Replacement is through
the existing executable argument and the same selection/output contract.
Removal requires deleting this optional command and its documentation.

The isolated npm lock contains 172 versioned packages, all with integrity hashes.
Its SHA-256 is
`1ce9add4416f1875bb9fb360c521ec38840b3e663f3aa1ca8c06d81e9687ab61`.
Repomix is MIT licensed. Transitive licenses are MIT (147), ISC (11),
BSD-2-Clause (4), BSD-3-Clause (3), BlueOak-1.0.0 (5), Unlicense (1) and
LGPL-2.1+ (1, jschardet 3.1.4). These remain external, unmodified tool packages
with their bundled notices and license files. Any redistribution must preserve
those terms, including jschardet's LGPL obligations. The deprecated
`@secretlint/secretlint-rule-preset-recommend` 13.1.0 installation warning is
recorded in the external receipt. Version updates require repeating the actual
selection and file-content conformance checks. The original source has no
historical dependency lock, so 1.18.1 is an explicit acceptance-environment choice.

## Acceptance and remaining boundaries

All six focused source/actual-tool scenarios pass. The concentrated batch
passes 725 cases with zero skips (1,058 dependency warnings). Strict typing
passes both changed Python files; Black, Ruff, Integration scope, architecture,
IR, API baseline and diff checks pass. The concentrated run included the first
four packing cases, with the final six-case focused run verifying the added
import guard, full input blob checks and actual module CLI.
The existing full-lock parents remain open beyond this isolated optional tool.
The full default-suite timeout, Algorithms NumPy ownership and order-dependent
copyright-checker behavior remain separate open evidence. Pending Proposals
116, 117 and 118 retain their approval requirements.
