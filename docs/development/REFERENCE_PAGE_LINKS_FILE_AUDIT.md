# Reference page and link source closure

## Frozen source and ownership

Q-0067, Q-0070, Q-0072 and Q-0077 cover the complete aliases.py,
_linking_writer.py, _page.py and _render_context.py under the original
make_reference_docs directory. Their unmodified Git blobs are verified at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389.

Tools owns reference_pages.py, reference_links.py and reference_context.py.
Page records preserve the source class/module distinction, ordered members,
optional initialization and path/assertion behavior. Standard-library dataclasses
supply the existing mutable record behavior. Link formatting owns regex matching,
reference targets and ordered alias descriptions. RenderContext passes the same
mapping objects into each writer. Existing WriteIfDifferent owns destination
writes and preserves equal-content file timestamps. These optional tooling
objects stay outside canonical IR and production execution.

## Complete function comparisons

Source and native tests compare initialized and uninitialized page properties,
root/nested/empty paths, independent member lists, every member category, alias
counts and insertion ordering. Full writer comparisons cover normal references,
missing aliases/targets, anchored members, Unicode slugging, unmatched syntax,
repeated linkification, literal writes and repeated target emission. Three output
roots compare complete body and reference-target bytes against the original.
The native generated page links circuit.append(op) documentation through the
same writer and reuses the accepted equal-content output owner.

The original code retains set iteration order for targets. This behavior is
preserved, so ordering can differ between Python hash seeds. Source page records
and internal serializers are outside protected Stable Core APIs. This classical
formatting batch reuses previous circuit complex-state, workspace and primitive
versus model evidence. It introduces no numerical execution or resource model.

## Original cache defect and native correction

The original global link cache uses output-root and alias text as its identity.
It omits the page and alias mappings. A source test creates two writers with the
same root and alias but different page mappings. The second incorrectly resolves
to the first document's path. Native writers resolve the current supplied mapping
on each call. A later mapping update is also observed immediately. This removes
cross-document global state. The frozen source remains untouched and its failing
behavior remains asserted separately.

## Optional dependencies and replacement boundary

The frozen docs.env.txt blob is verified. The isolated reference environment
uses its exact versions, Griffe 1.15.0, mdit-py-plugins 0.5.0,
markdown-it-py 3.0.0, mdurl 0.1.2 and colorama 0.4.6. Distribution metadata and
license files identify ISC for Griffe, MIT for the Markdown packages and BSD for
colorama. Installation uses the existing package resolver with these explicit
versions and no package cache. These are optional developer-documentation
requirements. Runtime manifests and runtime imports remain outside this change.

Griffe supplies static Python objects at the documentation boundary, avoiding a
second Python source parser. The anchors plugin supplies the source's exact
slug algorithm. Page rendering consumes owned page records and the narrow write
protocol. The native context accepts StringIO and the existing WriteIfDifferent
without consumer changes. Source providers and native providers pass the same
writer cases. Replacing either dependency is local to this Tools boundary and
requires those source differential checks. The exit path is removal of the
optional generator and its isolated dependencies if documentation ownership
moves to another established renderer. No dependency is promoted into Core.

## Acceptance and remaining source files

Run tests/unit/test_reference_page_links.py with the existing notebook,
docstring, example metadata, discovery and call-graph regression. Strict mypy,
Black, Ruff, Integration ownership, architecture, IR, API and diff checks cover
the four changed Python files.

Signature/docstring rendering, module/class/TOC renderers and the package walker
remain open under Q-0068, Q-0069, Q-0071, Q-0074, Q-0075 and Q-0076.
Package export files Q-0065, Q-0066 and Q-0073 stay open until their consumers
close. Q-0080's explicit notebook catalog and Q-0086's older TensorFlow-docs
pipeline remain separate complete-source work. Pending Proposals 116, 117 and
118 remain unimplemented.
