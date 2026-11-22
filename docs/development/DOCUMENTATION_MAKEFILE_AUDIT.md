# Complete documentation Makefile and build guide

## Source and native ownership

Q-0122 covers the complete `docs/Makefile` from Qualtran v0.7.0 at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`4e92d9f146ba0ce21dd9f0b7302212d423a454d7`. The native `docs/Makefile` is
byte-identical, retaining Google's Apache-2.0 notice. Sphinx and sphinx-autobuild
remain optional development tools at their existing command boundary.

Q-0051 covers the complete `dev_tools/how-to-build-the-docs.md` at the same
revision. Its original blob is checked against the frozen inventory by the
source-file test. The native owner is
[the documentation build workflow](DOCUMENTATION_BUILD_WORKFLOW.md).

| Complete source responsibility | Existing native owner and actual evidence |
| --- | --- |
| Stage one metadata and notebook authoring | ComponentDocumentation, NotebookSpec, notebook_autogen and explicit notebook_catalog |
| Commit clean authored notebook sources | Existing notebook-cleaning and publication-check commands |
| Stage two notebook execution and export | notebook_execution, actual independent kernel and Markdown/notebook export evidence |
| Stage two docstring/API extraction | reference_templates and reference_docs, actual renderer and observable static setter refusal |
| Generated-only cleanup | documentation_cleanup, actual generated catalog cleanup and authored-path refusal |
| Stage three local HTML and hosting boundary | Native site config, actual full site and complete catalog publication, Makefile targets and existing hosting scope |
| Production branch reconciliation, ignore review and generated output commits | Maintainer-owned publication checkout, staged artifact review and existing release workflow |
| Local clean/rebuild preview | Explicit generated owner cleanup and actual Sphinx make-mode rebuild/clean |

The guide separates authored native docs from generated-only output. Its
publication-checkout policy adapts the original fixed docs-branch instructions.
The existing Git/release workflow retains authority. This source guide does
not establish deployed hosting or a promoted release.

## Complete Makefile behavior

All original variables and rules remain unchanged. The default/help target
invokes Sphinx make mode. The catch-all passes the selected target through the
same make-mode owner. Four live aliases invoke the watcher; auto-dev adds -a.
Both builders receive SPHINXOPTS followed by O. The original live defaults
include browser opening and automatic parallelism.

Original preflight requires both default executable names before any target,
even when the build command is overridden. Tests preserve each missing-program
refusal and ensure no child tool executes. An override command's exit seven
produces make's exit two and the original child-error diagnostic.

Tests copy the complete unchanged source/native Makefiles into a normal docs
working directory and compare every target's exact tool and argument sequence.
Separate actual executions use the accepted optional Python tools through tiny
executable launchers. Both original and native Makefiles execute real help,
strict HTML build, edited-source rebuild, clean and unknown-builder failure.
Source/output paths containing spaces remain supported by the original quoting.
The prior actual native live HTTP rebuild receipt is reused for the watcher
consumer, alongside this Makefile's exact dispatch evidence.

## Scope and retained boundaries

The direct Makefile launches the installed optional tools. It does not import
framework runtime classes, execute notebook kernels, change package metadata or
add a production dependency. The accepted environment's complete source input,
installed constraints and licenses remain recorded by the existing dependency
and site audits.

Full-site content warnings, complete API index and inherited-setter refusal,
original scientific notebook failures and hosted deployment remain independent
parent scopes. The guide reports these explicitly. Its source/native command
responsibilities close without a warning-free publication claim.

Neither file emits quantum gates, owns state or work registers, or adds a
measurement branch or resource formula. Accepted domain execution evidence
remains with its algorithm owners. Primitive measurements and source model
estimates retain their independent accounting there.

The combined dependency-closed batch passes 67 concentrated cases with zero
skips and four retained dependency/fork warnings, plus 29 integration cases.
Strict typing covers both new test files. Black, Ruff, Integration scope,
architecture, internal IR, protected API baseline, local documentation links
and whitespace checks pass. Prior full-site and live-rebuild evidence is reused.
