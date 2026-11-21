# Stage-two documentation cleanup source closure

## Complete source and ownership

Q-0046 covers every operation and the entry point of
dev_tools/clean-stage-2-doc-outputs.py at immutable Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. The original Git blob is verified
against the frozen inventory. The native owner is
tools/development/documentation_cleanup.py. It owns generated-output removal
and reuses the accepted Git-root discovery owner for command validation.

The unchanged source assumes that every Markdown and notebook file below docs
is generated. Its own build guide describes branch-specific output tracking.
Native docs also contains authored Markdown, so that source entry point cannot
be applied directly to the native docs root. The native command requires an
explicit generated-only directory. Within the checkout it accepts only
docs/components and docs/reference/flagquantum. Other generated-only staging
directories must be outside the checkout. Resolving paths before validating
them also rejects a permitted directory name symlinked into authored docs.

```bash
python -m tools.development.notebook_publication
python -m tools.development.documentation_cleanup docs/components
```

The cleanup removes published notebooks from the owned directory and retains
index.rst, as the original retains unrelated RST files. The publication command
can regenerate the removed outputs from accepted authored assets. Reference
outputs in docs/reference/flagquantum can be cleaned through the same command.
Cleaning a complete custom staged documentation tree uses its explicit external
directory. The API clean_generated_outputs takes a caller-owned generated-only
directory; the CLI applies checkout ownership validation before invoking it.
It does not discover or delete authored reference pages or notebook source assets.

## Comparison and adaptations

The unchanged original main runs only against isolated temporary documentation
trees. The native function runs against equivalent trees. Removal order, printed
paths, remaining files and repeated/missing-directory behavior compare exactly
for nested checkpoints, Markdown, notebooks and both absent and present original
reference/autotoc.rst. HTML, images and unrelated RST remain present. A checkpoint
path that is a file raises the original AssertionError before later removals.
Filesystem errors propagate and earlier successful deletions remain visible.

Native renderers use reference/autotoc.rst.inc, so cleanup also removes this
native TOC spelling from an explicit staged tree. The legacy source spelling is
retained for source-format generated trees. No generated-output manifest,
duplicated catalog or new runtime dependency is introduced. The ownership check
belongs to the CLI; it is an explicit adaptation of the source's assumed docs
branch layout.

The real accepted native library is published into a temporary checkout and all
120 generated notebooks are removed through the actual cleanup module command.
An adjacent authored guide remains byte-identical. Missing command arguments,
the checkout root, its ancestor, docs root, authored docs and symlinked authored
paths are refused. Both the native reference subtree and an external generated
staging tree are exercised through the CLI.

This is classical developer tooling. Accepted notebook quantum semantics and
the measured primitive versus model-estimate distinction are reused. Cleanup
does not execute scientific notebooks or introduce gate-count claims. The
previous full default-suite timeout, Algorithms NumPy dependency ownership and
preceding-execution copyright interaction remain open.

## Batch verification

The concentrated regression passed 311 tests with 915 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and whitespace checks passed. The batch retained
the complete existing real Jupyter execution and complex-state checks. These
focused results do not resolve the separately recorded broad default-suite debt.
