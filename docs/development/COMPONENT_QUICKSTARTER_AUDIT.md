# Component quickstarter source audit

## Scope and implementation

Q-0039 is the complete `dev_tools/bloq-quickstarter.html` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`, with Git blob
`fed8cda3ee0c55e415342c1750a6ff843acb7552`.
The native page is `tools/development/component_quickstarter.html`.
It generates a developer scaffold using existing native contracts. It adds no
runtime class hierarchy, framework export or second representation of a circuit.

| Source feature | Native owner or adaptation |
| --- | --- |
| Component class and description inputs | Frozen Python dataclass and escaped class documentation |
| Register add form and text/change events | Local DOM fields and immediate regeneration |
| Three register sides and widths | Existing Register, RegisterType, RegisterSide and Signature |
| Register descriptions and ordered signature | Preserved in generated class documentation and metadata |
| Composite construction placeholder | build_registers receives the existing builder and named input bindings |
| Composite dispatch | program delegates to ComponentProgramBuilder and its checked finalize |
| Resource declaration placeholder | Existing SymbolAllocator and RawCallees contract |
| Short display name | Generated short_name method |
| Native circuit entry | bind_register_program and append_to support ordinary circuit.append(op) |
| Syntax highlighting | Same optional highlight.js 11.9.0 URLs and integrity attributes |

Input words are supplied in signature order, with MSB-first physical wire tuples.
Clean and borrowed pools retain their existing meanings. Input release requires
explicit caller ownership through input_ownership. Created outputs are exposed
by bind().outputs. Thru inputs are returned, Left inputs consumed and Right
outputs created by the authored body. No placeholder pretends to implement
quantum behavior or a resource model.

Open the page in a browser, fill the component and register descriptions, and
copy the Python scaffold into the responsible module. Implement build_registers
and declared_calls before execution. The existing
`examples/algorithms/register_gates.py` shows a complete named-register operation
and its append path. The template uses that existing pattern.

## Full-source behavior and original defects

All three source JavaScript functions are executed unchanged by the Node DOM
fixture. It models the DOM operations used by both pages, including registered
input/change/click events. Tests compare field order, all three sides, width
expressions, documentation, placeholder behavior and display names. Tests also
compile and import the entire generated original Python module.

The original template calls Register with the removed `bitsize` keyword. Its
own generated signature raises TypeError under the frozen source environment.
Tests preserve that exact refusal and compare the original generated signature
syntax with the native executable metadata. The reference is unchanged.

The original composite placeholder receives Right-only registers as input
arguments. The native builder passes only Left/Thru bindings, matching the
existing native ownership contract. The original bloq_counts placeholder has
an obsolete return annotation; the native scaffold uses the existing declared
call contract. Neither placeholder supplies a model until its author implements it.

The source inserts raw identifiers and triple-quoted description text into
Python. Native generation rejects invalid/reserved identifiers and empty widths,
keeps names unique independently per input/output side, and escapes descriptions.
Width expressions remain authored Python expressions, including symbolic names
that must be defined by the author. The browser displays generated source and
does not execute those expressions. If the optional highlighter is unavailable,
the native page still shows plain Python text. The original highlighter-only
rendering has no such fallback.

## Executed component evidence

The generated class is imported and completed using existing builder methods.
A two-bit X body is bound and appended to a three-wire state with complex
amplitudes and an entangled spectator. Both the full unitary and every resulting
amplitude are checked. Applying the operation twice restores the exact input
within the declared numerical tolerance.

A separate completed Left/Thru/Right body consumes an explicitly owned clean
input word and allocates its Right result. The lifetime events, returned wires,
full state and zero work storage are checked. Missing input ownership retains
the existing binder refusal. No measurement branches occur in either example.

The X example emits two observed X primitives. Its independently supplied model
also declares two Clifford calls. The lifetime/view example emits zero gates.
These are tests of completed native authored examples; the source quickstarter
itself supplies no executable decomposition or numerical resource estimate.

## Validation and remaining acceptance

The latest focused run passes 13 cases. The concentrated register-builder,
binding, append, graph and notebook batch passes 305 cases with zero skips
and 36 existing warnings. That batch preceded the final two additional input
validation/event cases, both covered in the focused run. Strict typing,
Black, Ruff, Integration scope, architecture, IR and API checks pass.

The original Node fixture is a behavior test double, not a browser renderer.
The earlier in-app browser rejected the file URL under its HTTP/HTTPS-only
policy; that route was not retried. The actual browser evidence below now
closes the remaining renderer review for Q-0039.

The retained highlight.js assets are optional and loaded from the same pinned
CDN URLs with the original SRI and crossorigin attributes. Those assets were not
fetched or rendered in the Node fixture. This batch does not claim a new supply
chain audit of the inherited assets.

## Complete real-browser acceptance, November 26

Q-0039 is accepted. An isolated actual Chrome 158.0.8127.88 renderer reviewed
the unchanged native and original HTML over a read-only loopback HTTP server.
The server served only those two files. This uses the allowed HTTP route;
the rejected file URL was not bypassed. The in-app browser service timed out
twice without returning state, so this acceptance uses actual local headless
Chrome through its debugging protocol instead. It does not claim an in-app
browser success.

Desktop 1280x900 and mobile 390x844 screenshots were inspected. Inputs and
buttons remain visible, mobile register fields form one column, and the
document has no horizontal overflow. Long code lines scroll inside their code
container. Actual keyup, change, add-register and regenerate events create
Thru/Left/Right declarations, keep Right ports out of input parameters, reject
invalid identifiers and empty widths, then restore the valid scaffold.

The unchanged pinned CDN script and stylesheet loaded with the page's original
integrity attributes. Actual highlighting created 74 syntax spans in the
initial scaffold. A cache-disabled reload with the CDN blocked has no hljs
global and still renders the complete plain Python scaffold with no generator
error. Both highlighted and plain-code screenshots were inspected.

The browser-generated three-port Python module imports under the existing
native environment. Its ordered widths and sides match the form, quoted text
remains intact, and execution still refuses the unimplemented body. All 13
focused source/native tests pass again. Previous complete source, ownership,
complex-amplitude, append and resource evidence is reused because the HTML
is byte-identical; no new numerical or quantum behavior was introduced.

[The receipt](evidence/component_quickstarter/20261126/receipt.json) records the
HTML/source hashes, checks and screenshot hashes. Review images:
[desktop](evidence/component_quickstarter/20261126/desktop.png),
[mobile](evidence/component_quickstarter/20261126/mobile.png),
[highlighted code](evidence/component_quickstarter/20261126/highlighted-code.png),
and [plain code](evidence/component_quickstarter/20261126/plain-code.png).
The isolated browser profiles and HTTP server were removed or terminated.
Hosted services and other browser engines are outside this renderer evidence.
The original generated Register(bitsize=...) failure remains recorded above.
