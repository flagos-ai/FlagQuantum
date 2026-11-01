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
use declared quantum parameters, nonnegative decimal literals, parentheses, +, *,
positive literal powers **1 through **8 and unsigned binary ==, !=, <, <=, >, >=. Comparisons have lower
precedence than arithmetic and produce one-bit 0/1 results. Line and block comments
and CRLF are accepted. The entire source must be consumed.

Omitted qnum widths resolve at initialization. Literal allocation sizes must
agree with explicit declarations. Fresh assignment uses the complete unsigned
expression domain; a wider declared target is preserved with zero extension.
Quantum variables are not folded from the basis values used to initialize them.
Pure literal subexpressions can be folded. Duplicate output names, unknown
symbols, uninitialized reads and repeated initialization are rejected, with
native source locations. Single-qubit calls on qnum, numeric indexing, arrays,
slices, bind, unscoped local initialization, classical parameters, consuming Input,
free/drop, control/invert, signed/fractional numbers, in-place assignment
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
Pure literal phases, multibit results and arbitrary angles remain
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

## Bounded compute/apply/cleanup scopes

Run `python -m examples.synthesis_qmod_within` for the shortest source workflow.
Function-level `flag:qbit;`, `tmp:qnum<N>;` or `tmp:qnum;` declarations may be
prepared by fresh quantum-expression assignments in a lexical
`within { ... } apply { ... }` block. Earlier prepared locals can feed later
computations. Apply admits existing one-bit `phase(expr,pi)` and
Const-only helper calls. Helpers can themselves contain supported scopes.
Block words are contextual: existing helpers and variables named within or
apply remain valid, including calls to within(...) inside an apply block.

The adapter emits existing qreg.local/compute/phase/call records, followed by
qreg.uncompute/release in reverse preparation order. Compiler proves dependency
preservation, uses each selected computation's exact inverse and reuses proven
zero storage. Locals never become named results. After apply, a source local is
uninitialized again; a later block can reuse the name with fresh owned identity.
Bare qnum infers its width once and retains it across blocks. Unused declarations
allocate no storage and do not appear in the output-only inferred-width map.

Local reads or initialization outside scopes, pure literal preparation, allocate
or gates in preparation, helper preparation calls, Output preparation, arbitrary
gates/writes in apply, Output helper arguments, nesting in preparation and declarations
inside blocks are refused. Empty apply is valid; empty within is refused. One-bit comparison phase arguments use the existing Compiler predicate path.
Manual uncompute/release/free/drop and consuming Input remain outside admission.
Generated local and cleanup operations count toward existing expansion budgets.
No simulator, SDK, allocator, executor or duplicate lifetime checker is added.
See the [contract](../../../docs/api-changes/FQ-SYNTH-QMOD-WITHIN-20261101.md)
and [measured delivery](../../../docs/roadmap/SYNTH_QMOD_WITHIN_DELIVERY.md).

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
captured external variables, mutable borrows or consuming Input. Helper-owned
local storage is limited to the within/apply profile below.
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

## Unsigned comparisons

Comparisons reuse Compiler qint.equal/qint.less with operand reversal and a
one-bit equality-to-zero complement. Operands are emitted in source order;
unequal widths are zero extended by Compiler. Literal-only comparisons fold
exactly, while quantum results retain their full declared domains. Const helper
arguments may alias. Parenthesized comparisons compose with arithmetic and with
explicitly parenthesized comparisons; unparenthesized chains, logical operators,
signed values and statement-level control remain refused. Python's admitted
comparison-chain semantics do not silently apply to native source.

The official [phase statement](https://docs.classiq.io/qmod-reference/language-reference/statements/phase/)
documents polynomial expressions. FlagQuantum additionally admits one-bit
unsigned comparison predicates in phase(expr,pi), including comparisons nested
in admitted arithmetic. This is a FlagQuantum-native extension to the import
profile, not a claim that Classiq accepts this spelling. Existing Compiler
predicate construction provides the sign and cleanup without exporting a flag.
The pinned upstream original-source cohort does not increase with this extension.
Run python -m examples.synthesis_native_predicate_phase for a complete workflow.

Run `python -m examples.synthesis_qmod_comparisons`. Tests are
`tests/team/ecosystem/test_qmod_comparisons.py` and
`tests/integration/test_synthesis_qmod_comparisons.py`.
The additive `comparison_sources.json` fixture sidecar records the unchanged
upstream comparator_integer_example.qmod and shared license. Its full three-bit
input and inferred one-bit flag pass both public file and text execution with
complete-state and zero-work checks: baseline three originals plus this one
give four admitted originals. The original three-file manifest and seven-digest
verifier remain frozen; the additive provenance test checks the new cohort.
The inspected two-variable comparator requires signed allocation and remains
unsupported. See the [comparison delivery](../../../docs/roadmap/SYNTH_QMOD_COMPARISONS_DELIVERY.md)
for measured costs and boundaries; this does not imply all-corpus support.

## Nested apply scopes

Apply may contain another supported within/apply block, to lexical depth 32.
Preparation still admits only fresh quantum-expression local assignments.
Each inner scope cleans only its newly materialized locals and restores the
outer context; parent locals may feed inner computations and stay live until
parent exit. Reinitializing a live parent is refused by Compiler. Inner locals
cannot escape; siblings may reuse released names with fixed widths. Declarations
remain function-level and scope words remain contextual identifiers.

Run `python -m examples.synthesis_qmod_nested`. The example computes x+y into
energy, then phases energy<4 inside an inner scope, returning only x/y.
See the [contract](../../../docs/api-changes/FQ-SYNTH-NESTED-APPLY-20261101.md).

## Bounded positive literal powers

The exponent must be a literal leaf in 1..8; zero, negative, fractional,
classical/quantum variables, computed exponents and larger literals are refused.
Parenthesized bases may themselves contain powers. Right associativity is
preserved: x**2**2 has a nonliteral exponent and is refused, while (x**2)**2
is admitted. Full domain inference retains the existing 256-bit guard.

Power one reuses its operand. Higher powers expand left to right to at most
seven existing qint.multiply operations with exact intermediate widths. Compiler
selects multiplication implementations and scratch schedules, rather than a new
power kernel. Literal-only powers fold; quantum-dependent constants retain their
quantum dependencies. Existing phase, Const, local cleanup and resource limits
still apply. This is bounded support, not optimal addition-chain selection.
Run python -m examples.synthesis_qmod_powers for the complete cube workflow.
