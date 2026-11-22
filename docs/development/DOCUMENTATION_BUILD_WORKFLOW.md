<!-- Adapted from Qualtran v0.7.0 dev_tools/how-to-build-the-docs.md, Google LLC, Apache-2.0. -->

# Documentation build workflow

## Author component notebooks

Class docstrings, lazy example factories and authored notebooks are the content
sources. Follow [component documentation](COMPONENT_DOCUMENTATION.md) when
editing component metadata. The existing `notebook_autogen` owner can render a
NotebookSpec through a domain-owned callback. The explicit `notebook_catalog`
selects the published library. Its default publication preserves the accepted
notebooks under `examples/notebooks` byte for byte.

Review and commit notebook sources with clean outputs through the existing
notebook-cleaning workflow. Generated publication copies retain their separate
output ownership. The domain examples and tests establish which scientific
notebooks execute successfully. The component guide retains two original
non-coprime factory counterexamples in its acceptance notes.

## Generate publication content

Use a full development environment with the optional notebook and reference
tools. From the repository root, publish the component catalog with the existing
command.

```console
python -m tools.development.notebook_publication
```

This writes `docs/components` and its ordered Sphinx index. It copies the 120
accepted distinct notebooks and adds their concept/root navigation. It does
not execute those notebooks. When refreshing computed outputs, use the separate
execution owner with its original parser and failure propagation.

```console
python -m tools.development.notebook_execution --no-only-out-of-date
```

The execution owner selects tracked notebook sources in the current checkout.
Inspect that scope before running a whole-library refresh. It exports notebooks
by default. Add `--output-md` when Markdown exports are needed. It aggregates
notebook failures and exits unsuccessfully when execution fails. The source
chemistry resource notebook's legend error and the broader scientific boundaries
remain documented in their domain audits.

Generate API pages with the existing loaded-namespace renderer.

```console
python -m tools.development.reference_templates
```

This writes `docs/reference`. The separate static renderer is
`python -m tools.development.reference_docs`. Its inherited-setter signature
refusal remains observable. Use the renderer whose accepted scope matches the
publication. A completed command establishes its own rendering scope.
Full-site publication retains recorded content warnings and API index gaps.

### Refresh generated outputs

Clean only an explicit generated-only owner directory before regenerating it.
The native docs tree also contains authored guides.

```console
python -m tools.development.documentation_cleanup docs/components
python -m tools.development.notebook_publication
```

The same cleanup command accepts `docs/reference/flagquantum`, or an external
generated-only staging directory. It rejects the checkout's authored docs root.
It retains RST navigation and unrelated files as described in
[the cleanup audit](DOCUMENTATION_CLEANUP_FILE_AUDIT.md).
Review generated output changes separately from authored content.

## Build and preview HTML

Install the optional renderer and watcher in the selected development
environment using the versions in [the documentation entry](../README.md).
Both `sphinx-build` and `sphinx-autobuild` must be discoverable on PATH for
Makefile targets, including help and clean. The original preflight checks both
names even when SPHINXBUILD is overridden.

```console
make -C docs html
```

The default output is `docs/_build/html`. Rendering uses stored notebook
outputs with execution disabled by the site configuration. Stage-three HTML
rendering consumes the already generated documentation. Its configuration
avoids importing the full FlagQuantum library. Kernel execution and API
extraction belong to the earlier generation stage.

`make -C docs` shows available targets. To remove the Sphinx build directory,
run `make -C docs clean`. Sphinx decides the behavior of other make-mode
targets. Unknown builders and child command failures propagate to make.

For an edit-and-rebuild preview, use an existing live target.

```console
make -C docs live
```

The aliases `auto`, `autobuild`, `live` and `livehtml` invoke the same watcher.
They open a browser and request automatic parallelism by default. `auto-dev`
also passes `-a` for rebuilding theme CSS and templates. SPHINXOPTS and O reach
both build and watcher commands. Override SPHINXAUTOOPTS to select a host/port
or omit the browser-opening option.

```console
make -C docs live SPHINXAUTOOPTS='--host 127.0.0.1 --port 8000'
make -C docs html SPHINXOPTS='-W --keep-going'
```

The full authored site currently retains content warnings. Strict builds of
small accepted fixtures pass. A strict whole-site build may fail on those
recorded warnings. Keep its diagnostics with the publication evidence.

## Review and publish artifacts

The original workflow commits stage-two output on a dedicated docs branch,
merges its main branch, verifies branch-specific ignore rules, regenerates,
reviews the diff, commits and pushes. FlagQuantum keeps its existing branch
and release ownership. Generated artifact review, source reconciliation,
reference/notebook generation and final HTML build are the corresponding
responsibilities here. An optional publication checkout must have the intended
base revision and an explicit generated-output tracking policy before a
maintainer commits those outputs.

Record the verified source revision, command exits, output inventory and
remaining warnings with the publication artifact. Review the staged diff before
publishing through the maintainer's existing release workflow. Previewing a site
uses local build output. Remote hosting and release promotion have independent
credentials and acceptance gates. This guide's acceptance covers the complete
three-stage responsibility mapping and the documented native commands.

The native `docs/.gitignore` excludes only the generated Sphinx, component and
API owner paths. Authored Markdown remains visible to Git. A maintainer's
publication checkout can explicitly stage selected generated outputs with
scoped force-add after reviewing its intended output policy. Source and native
Git behavior is recorded in
[the output ownership audit](DOCUMENTATION_OUTPUT_OWNERSHIP_AUDIT.md).
