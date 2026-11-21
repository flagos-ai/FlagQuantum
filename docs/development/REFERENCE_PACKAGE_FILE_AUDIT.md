# Static reference package and generation source closure

## Frozen source and ownership

Q-0071 covers the complete original make_reference_docs/_make.py at Qualtran
v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Q-0065 exports its generator,
while Q-0066 and Q-0073 are license-only component/page package markers. All four
unmodified Git blobs are verified. The original export AST contains exactly
make_reference_docs and the two package-marker ASTs contain no executable code.

Native Tools separates static package traversal in reference_package.py from
native section policy and output orchestration in reference_docs.py. Accepted
page, context, link, signature, docstring, module/class and TOC owners supply the
complete dependency closure. make_reference_docs remains directly importable
from its owning module. The two empty source package markers need no duplicate
native wrapper packages. No original package namespace or Core export is added.
The existing optional Griffe/Markdown requirements and license boundary remain
as documented in REFERENCE_PAGE_LINKS_FILE_AUDIT.md.

Native major classes use existing FlagQuantum Circuit, ProgramRecord,
ComponentProgramBuilder, Signature, Register and RegisterType paths. Native
section captions follow their owning domains. Explicit additional links account
for the public lazy Circuit entry. Ecosystem and experiment-storage exclusions
supply the native minimal policy. The source's qualification policy is applied
explicitly during frozen-package comparisons. This keeps reference-only names
out of the native default configuration.

## Complete frozen-package evidence

Both walkers load the untouched frozen Qualtran package with its original major,
skip, minimal and alias-exception policies. They produce the same 41 pages,
353 aliases and 271 links. Page types, canonical object identities, member
categories, ordered page/member records, section order, aliases, preferred paths,
anchors and additional links compare. The original longest-path preference and
its set-order tie handling are preserved. Empty preferred-alias inputs retain
IndexError. Private/foreign alias filters, explicit exceptions, missing aliases,
root-package assertions, independent walker state, duplicate alias assertions,
minimal/skip toggles and orphan warnings compare directly.

The complete configured source and native generation paths each render all
41 frozen pages plus the TOC successfully. Both produce the same set of 42
files. Every byte outside source unordered submember regions compares directly.
Those regions use sets of (name, Griffe object) pairs whose identity-dependent
hashes change between separate parses. The comparison canonicalizes only each
`#### Members` region's named entries and blank separators, preserving and
checking each description and all other content. It does not modify either
reference source or generated files. The earlier renderer tests retain exact
byte comparisons when supplied the same concrete objects. Native output retains
the source ordering behavior, so cross-parse byte determinism is not claimed.

An independent package fixture runs both real loaders through properties,
cached properties, attrs.frozen constructor synthesis, aliases, modules and
minor/major classes. Its full page bytes compare, with native section captions
and explicit link identity verified separately. The original only synthesizes
constructors for attrs.frozen. Mutable/ordinary classes retain their existing
behavior. Native page records and renderers explicitly accept forwarded Griffe
Alias objects, matching source runtime behavior beyond its narrower annotations.

## Lifecycle and source defects

The original module removes property and functools.cached_property from global
Griffe visitor tables at import. Native ReferenceExtension binds the equivalent
filter to each current visitor through the existing Griffe extension hook.
Ordinary and native visitors run concurrently with independent property results,
and global tables retain their original content. The original AttrsExtension
class-member callback and synthetic constructor parameters are retained within
that same optional-parser normalization owner.

The original DFS marks modules seen only after descending. A real module alias
cycle reproduces RecursionError. Native tracks currently active canonical paths,
omits the repeated cycle edge and removes active state in finally, including on
an injected collision. The frozen acyclic package records remain identical.
The native default FlagQuantum walk completes with 1,520 pages, 7,254 aliases
and 6,232 links in this checkout. All six configured major classes are present.

The original generator opens the TOC before creating its destination, so a new
output root raises FileNotFoundError. Native creates the destination before
calling the existing renderers. Both behaviors are asserted. Existing
WriteIfDifferent exceptional-buffer semantics and direct TOC partial-output
semantics remain unchanged and documented.

## Actual native boundary and remaining work

The full native walk succeeds, but the default whole-package render reaches
Circuit.n_qubits.setter and raises the frozen signature renderer's ValueError
after 17 output files in the measured run. The original signature renderer
receives that same actual Griffe method and produces the exact same refusal.
The native orchestration is tested to propagate it. Setter/deleter support is
therefore an explicit remaining native documentation capability. This batch
does not claim successful full-library publication. It preserves the original
signature/decorator boundary while completing its source workflow.

The batch is classical documentation tooling. Accepted real circuit.append(op),
complex-state, workspace, measurement and separately observed primitive/model
resource evidence is reused. No new quantum algorithm, numerical backend or
resource estimator is added.

Run test_reference_package.py with the previous page-renderer, signature,
docstring, links, notebook, SDK helper, discovery, metadata and graph regression.
Strict mypy covers all eleven changed Python files. Black, Ruff, Integration
ownership, architecture, IR, protected API and diff checks complete the batch.

Q-0080's explicit notebook catalog and Q-0086's older TensorFlow-docs pipeline
remain separate complete-source work. Pending Proposals 116, 117 and 118 remain
unimplemented.

## Final verification

Concentrated regression passes 219 tests with zero skips and 889 visible upstream
parser/fork warnings. Strict typing, Black, Ruff, Integration ownership,
architecture, IR, protected API and diff checks pass. Base-environment collection
initially reproduced five missing-Griffe import errors despite marker exclusion.
The five reference test modules now defer optional imports until execution and
use postponed annotations. The same base collection succeeds with 14 selected
tests and 68 deselected tests. No import-or-skip path was introduced, and the
complete locked reference regression still executes every selected case.
