# Notebook library publication source closure

## Complete source and native responsibility

Q-0036 covers every constant and function in
/dev_tools/autogenerate-bloqs-notebooks-v2.py at the immutable Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389 revision. The original source and all
12 concept/root source assets are Git-blob verified through the inventory. There
are 11 concept entries and one root entry. Their explicit native destinations
are recorded in tests/fixtures/notebook_publication_source_mapping.json.

The native owner is tools/development/notebook_publication.py. The accepted
notebook_catalog owns the 110 registered topic paths and their grouping. Existing
notebook_autogen owns component metadata rendering and authored-cell merging.
The new owner composes the ordered render calls, publication, Sphinx contents and
unlisted-document reporting. Existing WriteIfDifferent owns output writes.

## Adaptation of authored and generated documents

The original publishes live BloqDocSpec instances into notebooks in its Python
package. Native equivalents are already accepted authored notebooks under
examples/notebooks. They contain domain-specific examples, declarations,
append-based execution and graphical representations across separate owners.
Publication retains their exact text and writes a generated copy under
/docs/components. It does not overwrite the source notebook files by default.
A caller supplying the existing domain renderer runs it over the 110 registered
topics before publication. The accepted NotebookSpec renderer is exercised in
this callback, preserving authored cells while updating generated title and
import cells. This is an explicit authoring/storage adaptation, not a claim that
all 237 upstream BloqDocSpec objects have become one native runtime superclass.

The source's metadata-identity missing-document report becomes a report of
unregistered native notebook assets, the publication units in the native catalog.
It excludes all registered topics, concept entries and root entries, folds
duplicate discoveries and reports the remaining paths in deterministic order.
Unlike source set iteration, warning order is stable. Registration remains
explicit: discovering a new asset never silently adds it to the publication.
Source notebook-generation metadata discovery remains available separately in
component_discovery, for workflows using ComponentDocumentation records.

## Ordering, contents and failures

Original render_notebooks and native render_notebooks invoke the supplied owner
in identical 110-topic order. The source TOC output compares byte-for-byte after
only the 122 explicitly audited path references and native branding are mapped.
The 11 section captions and original depths are unchanged; the root-components
caption reflects the native domain name. Empty sections and explicit depths
match the source helper. The native output uses a separate destination and creates
its parent directories through the publication owner.

One concept and the root entry also appear in the topic catalog. All 122
references are retained in the TOC, while duplicate output paths are written once.
The real native library therefore emits 120 different notebooks and one index.
Every notebook output compares byte-for-byte to its accepted native source.
Repeated publication preserves unchanged output timestamps through the existing
file writer. Source renderer errors propagate before publication starts. A later
missing source leaves previously written output visible, matching the source's
sequential partial-publication behavior; it does not fabricate a completed TOC.
A destination equal to the source is refused to protect authored assets.

## User workflow and verification

```bash
python -m tools.development.notebook_publication
```

The real module entry runs in a temporary Git checkout with the complete catalog.
The all-native library is separately published from this actual checkout into a
temporary destination. Tests also run the accepted NotebookSpec renderer through
the callback and verify preservation of an authored narrative and regeneration
of its metadata cells. No actual repository documentation is overwritten by tests.

This is classical publication tooling. Source notebook Git identity, native
imports and individual quantum semantics are reused from the catalog and the
accepted 110-topic/12-reference notebook parents. Publication does not execute
all scientific notebooks and makes no new gate-count or numerical-performance
claim. Existing measured primitive and model-estimate separation is retained.
Q-0024's independent shell clean-tree wrapper remains open. Full default-suite
NumPy dependency and preceding-execution copyright failures remain open.

The concentrated batch regression passed 298 tests with 916 visible warnings and
zero skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
team scope, architecture, multi-level IR, public API and whitespace checks passed.
The previously recorded full default-suite timeout and isolated dependency failure
remain unresolved and are not represented as a passing whole-repository result.


## Committed library index and actual site publication

Q-0133 covers the complete committed `docs/bloqs/index.rst` at the same frozen
revision. Its Git blob is `db2da97b72b546f02a246d2bb3ef1df39e43d175`.
The actual unchanged upstream generator produces this exact committed index.
The existing source-to-native notebook mapping then yields the exact native
index, including every section, entry, repeated path, caption and depth. The
only presentation mappings remain the already audited identity and asset paths.
No separate catalog or registration authority is introduced.

The real native publisher copies all 120 distinct authored notebooks unchanged
and generates the complete component library. The actual Sphinx CLI renders
that complete output with the accepted native configuration and execution off.
All 120 notebook HTML files and the library index are verified, including
arithmetic and finite-field entries. Content warnings retain ordinary Sphinx
policy. This acceptance establishes complete catalog publication, while
warning-free scientific documentation remains open. The local workflow is
recorded in `docs/README.md`.

This index owns navigation and publication structure. Domain execution,
complex-amplitude, workspace and separate primitive/model resource evidence
remain with the existing algorithm owners. The root homepage Q-0135,
fundamentals Q-0132 and API include Q-0136 remain separate. In particular,
the complete API generator still retains the previously recorded native
setter-signature refusal. No additional quantum primitive or model is defined.
