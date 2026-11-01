# Native Qmod source frontend

The package and parser handoff stay private. The reviewed public workflow is
`synth.synthesize(Path("source.qmod"))` or explicit Qmod text through the same
entry. `source.py` owns bounded regular-file UTF-8 acquisition and converts
native diagnostics to existing public InteropConversionError reports. It does
no synthesis, planning, execution, SDK import or source evaluation. Literal
strings are always text. Version context is checked before file I/O; descriptors
are closed on every path. Ordinary file errors retain their OSError subclasses.

Ecosystem owns lexical parsing, native syntax, symbol resolution and translation
into the existing Compiler construction profile. This is private development
evidence for one pinned source snapshot, not a public adapter/registry entry or
Classiq SDK/cloud integration. The public Python entry remains
`from flagquantum import synth`; its complete workflow is documented in
[the synth README](../../synth/README.md).

## Ownership and shortest contributor path

`syntax.py` has bounded lexing and a small precedence parser. Its immutable
syntax records never leave the adapter. `adapter.py` converts to existing
HybridProgram/Value/Operation/SourceLocation records and delegates semantic
validation to Compiler. QmodImport is a private handoff containing the program,
expression definitions, source name, declared version context and inferred widths;
the existing InteropImportResult expects already lowered CoreIR and cannot carry
these construction definitions. This adds no public schema or generic IR.

The adapter does not synthesize, plan, execute, read files, evaluate source text,
generate Python callables or import optional SDKs. Compiler owns arithmetic,
initialization/lifetime checks, resource selection and target compilation;
Runtime owns execution. Never add kernels, per-file emitters or a second
semantic validator here.

```bash
python tests/fixtures/classiq_qmod/verify_sources.py
python -m pytest tests/team/ecosystem/test_qmod_parser.py tests/team/ecosystem/test_qmod_source.py tests/integration/test_qmod_parser_execution.py -q
```

The first command verifies seven source/license digests. The tests retain the
three original declarations, lower through Compiler and execute the original
one-qubit X plus separately labelled small synthetic cases. Larger original
files are not simulated by this test grid. See the
[delivery and compatibility counts](../../../docs/roadmap/CS07_QMOD_PARSER_DELIVERY.md).

## Initially admitted grammar

One `qfunc main` and optional bounded helper definitions have only output scalar parameters. Scalars are qbit, bare qnum
or `qnum<positive literal width>`. All are unsigned integral storage in this
profile. Statements are fresh assignment, allocate(var), allocate(literal,var),
X/H/S/Z on qbit, hadamard_transform on one scalar, and the bounded phase
statement below. Right-hand expressions
use declared outputs, nonnegative decimal literals, parentheses, +, * and a
literal **2. Standard arithmetic precedence applies. Line and block comments
and CRLF are accepted. The entire source must be consumed.

Omitted qnum widths resolve at initialization. Literal allocation sizes must
agree with explicit declarations. Fresh assignment uses the complete unsigned
expression domain; a wider declared target is preserved with zero extension.
Quantum variables are not folded from the basis values used to initialize them.
Pure literal subexpressions can be folded. Duplicate output names, unknown
symbols, uninitialized reads and repeated initialization are rejected, with
native source locations. Single-qubit calls on qnum, numeric indexing, arrays,
slices, bind, locals, classical parameters, Input/Const,
free/drop, control/invert/within, signed/fractional numbers, in-place assignment
and other functions/operators are outside this subset and must fail explicitly.

Limits: 262,144 UTF-8 source bytes, 4,096 tokens, depth 32, 78 decimal digits per
literal, 256-bit scalar domains and at most 4,096 expanded effect operations.
The existing Compiler independently enforces total model/node/live-width budgets. No truncation or silent best-effort conversion is allowed.

`parse_qmod(..., source_version=...)` requires explicit context matching retained
Classiq commit `2bfd835e73b6882fcc515a725098b04766c238e7`. That argument is a
caller declaration, not proof of source authorship or a hash check. The original
fixture manifest separately proves byte integrity. Current documentation may
change; do not silently accept another source version or expand the grammar.

## Ordered sign phases

`phase(expression, pi);` applies (-1)**expression without a result flag. The
expression uses the arithmetic subset above, references initialized quantum
outputs and has an inferred result width of exactly one bit. `pi` must be the
unshadowed symbolic coefficient; no evaluation or SDK is involved. Source names
shadowing `phase` or its `pi` coefficient are refused at the intrinsic call.
`phase(x*y, pi)` can use a direct CZ marker; `phase(x*x, pi)` preserves aliases.
Quantum-dependent constants preserve their overall sign, including `x*0+1`.
Pure literal phases, multibit results, comparisons and arbitrary angles remain
unsupported. Width inference uses declared domains, even after `x=0`.

This statement emits the existing qreg.phase callee and borrows live values;
Compiler handles exact cleanup before later mutation or arithmetic. It does
not consume or initialize the borrowed outputs. Use the default synthesis mode:
Qmod `output_mode="phase"` remains a refused Python-only expression mode.
Default logical CPU execution is admitted; public explicit basis/line targets
are refused by the existing exact-phase target guard. Memory is checked at the
existing planning boundary, before execution, rather than by the source parser.

Run `python -m examples.synthesis_qmod_phase` for a gate-free, named-result
workflow. Focused tests are `tests/team/ecosystem/test_qmod_phase.py` and
`tests/integration/test_synthesis_qmod_phase.py`. See the
[delivery record](../../../docs/roadmap/SYNTH_QMOD_PHASE_DELIVERY.md).
These are synthetic probes, not additional admitted originals.

## Fresh-output quantum helpers

A source can define at most 32 qfuncs with exactly one main. Helpers accept only
fresh output scalar variables, and every output must be initialized by return.
They may call each other in an acyclic graph, including forward references;
repeated calls must use different fresh storage. Argument kinds and resolved
widths match exactly. A bare qnum actual can infer its width from a verified
helper output; an already specified width cannot be coerced. There are no
captured external outputs, mutable borrows, Input/Const roles or local storage.
Builtin functions and the pi coefficient symbol cannot be redefined.

Each definition is independently translated and verified by Compiler, including
unused definitions. The adapter then substitutes output references in verified
register effects and threads the existing effect token. Arithmetic callees use
definition-qualified names and are reused across repeated calls. Final main is
verified again. The adapter does not clone quantum states, substitute source
text, execute Python, add a call executor or duplicate Compiler's lifetime
rules. Expanded effects point to the call site; arithmetic callees retain their
definition locations. Existing token, node, width and 4096-effect budgets apply.

Run `python -m examples.synthesis_qmod_helpers` for nested preparation and
`x*x+y` with named output reading. See
[helper delivery](../../../docs/roadmap/SYNTH_QMOD_HELPERS_DELIVERY.md) and
`tests/team/ecosystem/test_qmod_helpers.py` plus
`tests/integration/test_synthesis_qmod_helpers.py` for the contributor path.
This is fresh Output composition, not general function-role or scope admission.
