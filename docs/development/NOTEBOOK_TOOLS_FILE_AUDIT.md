# Notebook tool source closure

## Source and ownership

Q-0059, Q-0063 and Q-0064 cover complete untouched Qualtran v0.7.0
`clean_notebooks.py`, `jupyter_autogen.py` and its original test file under
`dev_tools/qualtran_dev_tools` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The tests verify all three original
Git blobs against the immutable source manifest. Google copyright and the
Apache-2.0 attribution remain in the native modules.

`tools/development/notebook_cleaning.py` owns tracked notebook discovery,
output/metadata cleanup, diffs and file replacement. It reuses the accepted
bounded subprocess owner. `notebook_autogen.py` owns notebook path selection,
factory-source extraction, cell assembly and merge-by-ID. It reuses accepted
Testing metadata, docstring parsing and WriteIfDifferent. Existing NotebookNode
represents cells directly, replacing the source's additional abstract cell base
and two wrapper classes. No new notebook execution engine is introduced.

Native imports start with `import flagquantum as fq`. Each owning domain
supplies its actual signature/call-graph code. Native components have no common
object graph method, so this dependency stays explicit. A generated native
notebook constructs WordSwap, uses `circuit.append(op)`, draws the actual circuit
and queries the same swap's stored call graph through existing Compiler/Drawer
owners. Nine observed CX instructions and nine declared CNOT calls are asserted
separately. A renderer replacement can change its code callback without changing
the notebook merger or component metadata.

## Complete source comparison

The unchanged three original tests execute through native metadata/cell-view
adapters. Source extraction compares every frozen example factory. Complete
frozen documentation specifications compare the exact ordered cell content,
cell type and persisted cq.autogen identifiers. Source-specific graphical code
is supplied only as a reference callback for this comparison. The native
execution fixture independently runs its FlagQuantum code cells and asserts
that generated Python imports no Qualtran code.

Path defaults and explicit stems, module titles, notebook initialization,
loading, generated-cell replacement, retained authored/superfluous cells,
stable existing notebook IDs and equal-content no-write behavior compare against
the original. Duplicate existing generated identifiers retain the original
ValueError before writing. Explicit overwrite initialization removes the old
file and returns an empty notebook. The original return-expression omission in
factory extraction is preserved. Factories must assign their named object
before the terminal return. Default library factories with direct returns can
therefore produce incomplete instance cells, exactly as the original does.

The old cq.autogen and class-documentation identifiers remain stored metadata
so existing generated notebooks can be updated. They introduce no original
package imports or public package namespace. The source's unused display
constant and redundant cell wrapper hierarchy have no native runtime consumer.

## Original cleaning defects and native corrections

The original uses `diff` with check=True. The normal difference exit status 1
raises before the dirty-notebook write path, leaving its temporary file behind.
The regression runs the untouched source on a dirty notebook, captures this
failure, and compares its fully preprocessed temporary notebook with the native
cleaned output. The source also leaves temporary files behind for a clean input
when do_clean=True. Tests reclaim source temporaries outside its implementation.

Native cleaning accepts normal diff statuses 0/1 and raises on actual diff
failure. It writes a same-directory temporary file, replaces only when dirty and
requested, and always removes its temporary path. Both write failure and diff
failure preserve the original notebook and reclaim temporary files. Dry runs
preserve input bytes. Tracked discovery, multi-notebook dirty counts and repeat
cleanup are exercised through a real temporary Git checkout. Untracked notebooks
are excluded. Git and diff are explicitly bounded to 15 seconds.

## Dependencies and inherited scientific evidence

nbformat and nbconvert are existing optional notebook-tool dependencies with
BSD-3-Clause licenses. Sphinx/attrs retain their accepted documentation-tool
boundary. No production dependency or package manifest changes. The libraries'
untyped notebook I/O is isolated at the external module boundary, and native
code verifies NotebookNode objects before accepting their results. Missing
optional packages remain explicit import failures. No casts or type suppressions
are added. Notebook execution remains the responsibility of the existing runner.

This batch generates documentation and source cells. It adds no quantum gate
or arithmetic decomposition. Accepted complex-amplitude, workspace and resource
comparisons for WordSwap remain authoritative. Notebook model calls and measured
primitive instructions remain separate. Original factory parser limitations and
retention of retired generated cells are documented source boundaries.

## Acceptance evidence

The frozen corpus contains 393 example factories and 234 documentation
specifications. All are traversed in the differential checks. The concentrated
regression passes 128 cases with eight existing pydot deprecation warnings and
zero skips. It includes the three unchanged original tests through native views,
existing docstrings/discovery, component metadata, notebook protocol examples,
circuit/graph rendering and stored-program call accounting. Strict typing
passes three files. Black, Ruff, Integration scope, architecture, internal IR,
protected API and diff checks pass.

A separate base-environment collection probe observes all twelve new tests as
deselected with no import errors when the reference marker is excluded. This is
an optional-dependency collection guard, separate from functional acceptance.
Sphinx and nbconvert load only when their actual documentation/cleanup operation
is requested. No type stub package was installed. Strict checks use the
validated existing library boundary. Evidence logs and the frozen corpus list
are stored under `ftqc-reference-env/g2-notebook-tools-*-20261121.*` outside the
repository.
