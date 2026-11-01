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

## Admitted grammar

One `qfunc main` has Output scalars; helpers may add fixed-width Const scalars.
Scalars are qbit, bare qnum for Output,
or `qnum<positive literal width>`. All are unsigned integral storage in this
profile. Statements are fresh assignment, allocate(var), allocate(literal,var),
X/H/S/Z on qbit, hadamard_transform on one scalar, and the bounded phase
statement below. Right-hand expressions
use declared quantum parameters, nonnegative decimal literals, parentheses, +, * and a
literal **2. Standard arithmetic precedence applies. Line and block comments
and CRLF are accepted. The entire source must be consumed.

Omitted qnum widths resolve at initialization. Literal allocation sizes must
agree with explicit declarations. Fresh assignment uses the complete unsigned
expression domain; a wider declared target is preserved with zero extension.
Quantum variables are not folded from the basis values used to initialize them.
Pure literal subexpressions can be folded. Duplicate output names, unknown
symbols, uninitialized reads and repeated initialization are rejected, with
native source locations. Single-qubit calls on qnum, numeric indexing, arrays,
slices, bind, locals, classical parameters, consuming Input,
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

## Const and fresh-output quantum helpers

A source can define at most 32 qfuncs with exactly one main. Helpers accept
`const q:qbit`, `const x:qnum<N>` and fresh Output scalars. Const qnum requires
an explicit unsigned literal width; Output can retain bare-width inference.
Every Output must be initialized by return. Const actuals must already have a
known width and be initialized, even when the callee does not use them.
They may call each other in an acyclic graph, including forward references;
repeated calls may reuse Const actuals but need different fresh Output storage.
Const actuals may alias each other; Outputs must be disjoint from all arguments.
Argument kinds and resolved
widths match exactly. A bare qnum actual can infer its width from a verified
helper output; an already specified width cannot be coerced. There are no
captured external variables, mutable borrows, consuming Input or local storage.
Builtin functions and the pi coefficient symbol cannot be redefined.

The adapter preserves source positional matching, then emits the Compiler ABI
with Const actuals before Outputs regardless of source parameter order. It
returns owned qreg.call records and all definitions, including unused helpers,
only after Compiler verification. Compiler owns initialization, fresh roles,
aliases, complete returns, graph/expansion budgets, substitution and cleanup.
The former Ecosystem Output-only substituter and alias checker are removed;
no second role checker or call executor is introduced. Arithmetic callees keep
definition-qualified names. Existing source/token/width limits and Compiler's
4096 source/final-node/expansion-visit budgets apply, including empty calls.

Const preserves basis values, while phase and entanglement may change. It is
not classical const or an unchanged reduced state. It may be arithmetic/phase
input or a nested Const argument, but cannot be a gate target, initialized,
computed into or passed as Output. Even S/Z and cancelling X/X through gate
calls are conservatively refused; use the admitted one-bit/pi phase statement.
Callers may mutate their values after return. Main stays Output-only with known
source preparation. `input`, `input const`, `output const`, unqualified mutable
parameters and arbitrary unitary effects remain unsupported.

Run `python -m examples.synthesis_qmod_helpers` for nested preparation and
`x*x+y` with named output reading. See
[helper delivery](../../../docs/roadmap/SYNTH_QMOD_HELPERS_DELIVERY.md) and
`tests/team/ecosystem/test_qmod_helpers.py` plus
`tests/integration/test_synthesis_qmod_helpers.py` for the contributor path.
Run `python -m examples.synthesis_qmod_const` for the public Const computation
workflow. See [Const delivery](../../../docs/roadmap/SYNTH_QMOD_CONST_DELIVERY.md),
`tests/team/ecosystem/test_qmod_const.py` and
`tests/integration/test_synthesis_qmod_const.py`. This admits closed Const/Output
composition, not general function-role or scope semantics. Source error classes
and coordinates remain; role/alias/expansion diagnostics now use Compiler codes.
