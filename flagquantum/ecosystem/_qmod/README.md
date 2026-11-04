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
explicitly parenthesized comparisons; unparenthesized chains, keyword and/or/not,
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

## Integer allocation and bitwise expressions

Use `&`, `|`, `^` and unary `~` on unsigned integer expressions, including
one-bit comparison flags. AND uses the narrower operand width. OR and XOR
zero-extend to the wider width. Inversion complements exactly the operand's
inferred width. The operators are eager and coherent, so `0&x` and `1|x`
still require an initialized `x`.

`allocate(width, SIGNED, 0, target)` declares a two's-complement integer.
`UNSIGNED` selects an unsigned integer, and `True` and `False` are accepted
as the equivalent signedness literals. Signed allocation and signed expression
outputs are scoped to `main`. Bitwise operators sign-extend signed operands
and zero-extend unsigned operands where needed. Register reads expose the
raw basis encoding. Signed arithmetic is limited to subtraction and negation. Signed helper ports
and fractional allocations remain outside this subset.

`target ^= mask` toggles the selected bits of an initialized writable integer.
A negative literal mask requires a signed target. A distinct initialized
integer may also supply the mask when its width fits the target. The source
is preserved and the update reuses X or CX gates without workspace.

Precedence (highest first) is **, ~, *, +, &, ^, |, comparisons. Parenthesize
comparison operands, for example phase((x<2)|(x==3),pi). Unary depth counts
against existing expression limits. The one-bit path keeps its existing
arithmetic lowering. Multi-bit expressions use bitwise reversible gates and
the existing compute and cleanup scheduler.

Run the original unsigned AND program from the repository root:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/bitwise_and_2vars_example.qmod")
program = synth.synthesize(source)
print(dict(program.probabilities(fq.run(program), registers=("a", "b", "res"))))
```

## Literal-angle PHASE and RZ originals

Uppercase `PHASE(k, q)` and `RZ(k, q)` admit a nonnegative integer or contiguous
fixed-decimal literal k<=2**53 in radians and an initialized scalar qbit. Exact
decimal bounds are checked before float64 conversion; source literals are at
most 78 characters. Both are reserved intrinsics. Negative, exponent-form,
symbolic, computed and quantum angles remain refused, as do numeric/array and
Const gate targets. Decimal data expressions are limited to the extrema and subtraction profiles below.
Lowercase
`phase(expression, pi)` remains the separate predicate operation. Within/apply
actions and role rules are unchanged.

Compiler owns the existing register gate validation and emits the existing
Core phase instruction with its named theta parameter. Runtime and numerical
kernels are reused; no matrix or backend-specific implementation is introduced.
Run `python -m examples.synthesis_qmod_phase_gate` and
`tests/integration/test_synthesis_qmod_phase_gate.py`. Compare amplitudes, since
probabilities alone do not expose phase errors. See the
[delivery](../../../docs/roadmap/SYNTH_CONSTANT_PHASE_GATE_DELIVERY.md).

The native high-level counterpart is an ordinary Python float-return phase
function, not a Qmod string or Circuit gate wrapper. Run
`python -m examples.synthesis_native_and_qmod_rz`; see
[the native-first delivery](../../../docs/roadmap/SYNTH_LITERAL_RZ_DELIVERY.md).

## Scalar pair gate originals

`CX(control, target)`, `SWAP(first, second)`, and
`CRX(angle, control, target)` accept distinct initialized scalar qbits.
CRX uses the same bounded literal angle syntax as PHASE and RZ. CX and CRX
preserve the control's basis value. SWAP changes both operands. Writable
operands must have the Output role. These calls remain outside the
within/apply subset. Compiler validates the two-register effect and lowers
to the existing Core gates.

Run `python -m examples.synthesis_qmod_pair_gates` to compare the pinned
originals with a native qfunc and a Circuit. All three originals start in
and preserve the zero pair. Coherent regression cases test control order,
SWAP movement, and CRX phase against independent amplitudes.
## Sparse Pauli evolution

`exponentiate(operator, t, q)` accepts an inline `SparsePauliOp` with
`SparsePauliTerm` and `IndexedPauli` records. Its `num_qubits` must match the
allocated `qbit[]` target. Each term's indices must be distinct and in bounds.
Index zero is the low bit of the output register. Empty support is an identity
term and contributes global phase. Time and coefficients can be signed literals
or bound classical `real` values.

Following the [Classiq guide](https://docs.classiq.io/explore/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide#exponentiate),
`exponentiate` runs one first-order Suzuki–Trotter step. FlagQuantum applies the
terms in literal order with the existing Pauli gate lowering. Inline sparse
operators also work with `suzuki_trotter` and its existing order and step limits.

Run the pinned original from the repository root:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/hamiltonian_simulation_guide.qmod")
program = synth.synthesize(source, classical_inputs={"t": 0.37})
print(fq.run(program).statevector())
```

A top-level `name: SparsePauliOp = SparsePauliOp { ... };` declaration
binds an immutable operator with literal coefficients. Pass its name as the
first argument to `suzuki_trotter` or `exponentiate`. The pinned Trotter guide
uses this form with ten first-order steps:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/hamiltonian_simulation_guide_trotter.qmod")
program = synth.synthesize(source)
print(fq.run(program).statevector())
```

### qDRIFT

`qdrift(operator, t, count, q)` takes an inline or named `SparsePauliOp`
and a literal sample count in 1..1024. The existing effect budget also applies.
Each sample is drawn with replacement using probability `abs(h) / lambda`,
where `lambda = sum(abs(h))`. Its Pauli rotation uses signed evolution angle
`sign(h) * t * lambda / count`. Zero total weight or zero time gives identity.

FlagQuantum seeds a local Python RNG with 0 per call during synthesis to make
the sampled Circuit reproducible. Running it applies that fixed sequence.
This seed and draw sequence are FlagQuantum conventions.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/hamiltonian_simulation_guide_qdrift.qmod")
program = synth.synthesize(source)
print(fq.run(program).statevector())
```

### Power-of-two remainder

`x % n` accepts an unsigned integer expression and a positive power-of-two
literal divisor. It keeps the low `log2(n)` bits, with a one-bit zero for
`n == 1`. Its precedence matches multiplication. The remainder reuses the
unsigned AND lowering with a low-bit mask. Bounded Boolean synthesis can
combine it with surrounding arithmetic, bitwise operations and equality.

Run `tests/fixtures/classiq_qmod/modulo_example.qmod` through
`synth.synthesize(Path(...))`. Its result register contains 3 for inputs 4
and 7 with divisor 4.

### Real amplitude preparation

`prepare_amplitudes([a0, a1, ...], bound, out)` initializes a fresh numeric or
qubit output in computational basis order. The normalized real literal vector
must have a power-of-two length from 2 to 256. Its length determines the output
width. Negative entries preserve their signs, including their relative phases.
A declared width must match the vector.

`bound` is a nonnegative L₂ error ceiling. Both zero and positive ceilings
use the exact real RY/CX preparation to floating-point precision. A positive
ceiling does not currently select a smaller approximate circuit.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/example_prepare_amplitudes.qmod")
program = synth.synthesize(source)
print(fq.run(program).statevector())
```

### Probability preparation

`prepare_state([p0, p1, ...], bound, out)` accepts a power-of-two array of
nonnegative probabilities that sum to one, from 2 to 256 entries. Entries
may be decimal or integer literals, or fractions such as `1/2` with a positive
integer denominator. Fraction syntax is local to the probability array.
The output must be fresh and its width must match the array length.

The prepared amplitudes are the nonnegative square roots of the probabilities.
`bound` limits L₂ error in the probability vector. The shared real-amplitude
lowering uses exact preparation to floating-point precision for any admitted
bound. The pinned `example_prepare_state.qmod` and `add_bell_states.qmod`
fixtures show probability literals and fraction literals, respectively.

### Bell preparation

`prepare_bell_state(1, out)` prepares the normalized state
`(|00> - |11>) / sqrt(2)`. `prepare_bell_state(2, out)` prepares
`(|01> + |10>) / sqrt(2)`. Both use a fresh two-qubit array and reuse
the GHZ preparation with a final Z or X gate, respectively.

### Array-controlled X

`control (controls) { X(target); }` accepts a whole initialized qubit array and
a distinct initialized writable scalar target. The target flips only on the
all-ones branch. For larger arrays, synthesis groups the controls and reuses
the existing clean Toffoli ladder across groups. It clears the group flags and
scratch before the next operation. The workspace counts against the ordinary
synthesis budget.

### Type II cosine and sine transforms

`qct_type2(x)` and `qst_type2(x)` apply orthonormal Type II transforms in the
frequency-row convention used by the Classiq executable tutorial.
`qct_qst_type2(x, q)` applies the cosine block for selector zero and the
negative sine block for selector one, preserving their relative phase.
The data register must be initialized and writable, with one to four qubits.
The combined form requires a distinct initialized scalar selector.

The bounded lowering uses registered gates and requires no workspace. Its
gate count grows with the transform matrix dimension.

### Fixed real arrays and encoding

Entry parameters declared as `real[N]` accept fixed-length sequences of
finite real values through `classical_inputs`. Rotation angles can read
elements at literal indices, as in `RY(data[0], q)`.

`encode_in_angle(data, out)` prepares one output qubit per element using
`RY(pi * data[i])`. `encode_on_bloch(data, out)` uses one qubit per pair,
applying RX and then RZ with angles scaled by pi. An odd final element uses
RX with an angle scaled by 2 pi. Both operations initialize a fresh qubit
array, with element zero on its least significant qubit.

### Unsigned extrema

Arithmetic expressions can call the unshadowed two-operand intrinsics
`min(a,b)` and `max(a,b)` with unsigned integer literals, initialized
registers, or other admitted expressions. They compare and select coherently,
preserving the inputs and cleaning temporary qubits. Nested integer extrema
compose with arithmetic and comparisons.

An extremum operand can also be a nonnegative, exactly binary-representable
decimal literal such as `3.5` or `1.25`. The importer aligns the binary point
exactly across operands. A fractional result requires a terminal,
inferred-width `main` output of type `qnum`. Its named probabilities use raw
unsigned basis values, with the number of fractional bits recorded in
`program.report["qmod_fractional_bits"]`. Fractional arithmetic beyond extrema and subtraction, and signed extrema,
are outside this profile.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/maximum_float_example.qmod")
program = synth.synthesize(source)
result = fq.run(program)
scale = 1 << program.report["qmod_fractional_bits"]["res"]
for (a, raw), probability in program.probabilities(
    result, registers=("a", "res")
).items():
    assert raw / scale == max(3.5, a)
    print(a, raw / scale, probability)
```

### Subtraction and negation

Fresh numeric assignments admit `a - b` and `-a`. The importer derives the
result domain from the operand widths and signedness. It preserves enough
bits for negative results and for negating the most negative input. The same
subtraction accepts exactly representable binary decimals and aligns the
operands to a common binary point. Fractional results require a terminal
inferred-width main output.

Named probabilities expose the raw basis encoding. For a register in
`program.report.get("qmod_signed_registers", ())`, decode the raw value as
two's complement before dividing by `2**fractional_bits`.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/subtraction_2vars_example.qmod")
program = synth.synthesize(source)
result = fq.run(program)
width = len(program.registers["res"])
for (a, b, raw), probability in program.probabilities(
    result, registers=("a", "b", "res")
).items():
    difference = raw - (1 << width) if raw >> (width - 1) else raw
    assert difference == a - b
    print(a, b, difference, probability)
```

### Quantum Fourier transform

`qft(target)` applies the exact forward quantum Fourier transform in place
to an initialized writable qubit array. It uses the same gate sequence as
the FlagQuantum Circuit primitive, including bit reversal. On the array in
register order, the first wire is the most significant bit. Qmod index
`target[0]` addresses the least significant bit. The transform does not
allocate work qubits.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(Path("tests/fixtures/classiq_qmod/qft.qmod"))
state = fq.run(program).statevector()
assert state.numel() == 16
print(program.report["resources"])
```

### Single Pauli exponentiation

`single_pauli_exponent(paulis, coefficient, target)` applies
`exp(-i * coefficient * P)` in place using the Pauli evolution convention.
Use a literal array of `Pauli::I`, `Pauli::X`, `Pauli::Y`, and `Pauli::Z`
with one entry per qubit, indexed in Qmod array order. The coefficient can
be a supported numeric literal, bound real, or literal-index read from a bound
real array. Identity words preserve the global phase.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/variational_quantum_imaginary_time_evolution.qmod")
program = synth.synthesize(
    source, classical_inputs={"angles": [0.13, -0.29, 0.41, -0.57, 0.68, -0.83]}
)
state = fq.run(program).statevector()
assert state.numel() == 16
```
