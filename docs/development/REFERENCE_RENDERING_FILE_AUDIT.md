# Reference signature and docstring rendering source closure

## Source and ownership

Q-0068 and Q-0069 cover the complete render_docstring.py and
render_signature.py under make_reference_docs/_components at frozen Qualtran
v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Tests verify both unmodified
Git blobs. Google copyright and Apache-2.0 attribution remain in the ports.

Tools owns reference_docstrings.py and reference_signatures.py. Both consume
optional Griffe objects and the accepted Writable boundary. Reference parsing
reuses the existing docstrings.parse_references implementation. Reference page
output reuses LinkingWriter and WriteIfDifferent. These changes add no new
public exports or execution semantics. Their dependency closure consists of the
already accepted page/link/context and documentation-parser modules.

## Complete signature behavior

The complete parameter formatter compares all five parameter kinds crossed with
absent/present annotations and defaults. Invalid missing parameter kind preserves
ValueError. Function and generic method outputs compare exact bytes across zero,
one and multiple parameters, absent/present returns and self/cls/unbound forms.
Property methods retain the required return annotation. Empty/misnamed method
receivers preserve IndexError and AssertionError.

Decorator dispatch compares regular, abstract, property, cached_property,
classmethod, staticmethod and abstract classmethod forms. Overloaded functions
retain the original empty output. Unknown decorators, multiple decorators and
fully-qualified functools.cached_property preserve exact original rejection
messages. The original simplistic CamelCase conversion is retained.

The original bespoke object-name table uses BloqBuilder and CompositeBloq.
Native examples instead use the owned ComponentProgramBuilder and ProgramRecord
names with builder and program variables. Both source-specific spellings and
native counterparts are explicitly asserted. All other comparable signature
bytes match the original. A native append signature composes with the accepted
reference writer and equal-content destination owner.

## Complete docstring behavior

All eight original section writers compare complete output and warnings. Cases
cover typed/untyped parameters and attributes, singleton unnamed and multiple
named/unnamed returns and yields, typed raises, notes, references, see-also,
example narrative/code and unknown sections/admonitions. Four nesting levels
include the original negative-level behavior. Empty section lists, summary-only,
multiple paragraphs and remaining parsed sections compare. The complete frozen
340-class documentation corpus compares summary text, rendered remaining
sections and warnings. Reference parsing continues through the accepted parser.

Invalid examples and missing exception annotations preserve ValueError. Empty
docstrings and a supplied non-text first parsed section preserve source errors.
Griffe treats a summaryless Args example as plain text in the locked environment,
and both implementations preserve that actual parsed outcome. The original
four-backtick example closing fence and unknown-section string formatting remain
unchanged.

Native section dispatch uses Griffe's concrete section classes for precise
static typing. Its supported inputs are canonical Griffe section instances.
Manually overriding a section kind to contradict its concrete class remains
outside this comparison. A text kind supplied through a different concrete
class is explicitly rejected at the parser boundary. This avoids carrying the
source's type-ignore dispatch into native tooling.

## Acceptance and remaining scope

Strict mypy, Black, Ruff, Integration ownership, architecture, IR, protected API
and diff checks cover the four Python files. Concentrated regression includes
the previous reference-page, notebook, SDK notebook-resource, discovery,
metadata, parser and call-graph workflows. Actual notebook execution remains
covered by its accepted complex-amplitude and kernel/spawn evidence. This batch
formats classical documentation and reuses that evidence. It introduces no
quantum primitive implementation or additional resource model.

The isolated optional dependencies and licenses follow
REFERENCE_PAGE_LINKS_FILE_AUDIT.md. Q-0071 package walking and Q-0074 through
Q-0076 page/TOC rendering remain open, along with their package export files.
Q-0080 explicit notebook catalog and Q-0086 TensorFlow-docs pipeline remain
separate full-source work. Pending Proposals 116, 117 and 118 remain
unimplemented.
