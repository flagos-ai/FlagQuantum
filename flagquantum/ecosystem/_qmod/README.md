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

Limits: 262,144 UTF-8 source bytes, 4,096 tokens, expression depth 40, lexical scope depth 32, 78 decimal digits per
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
raw basis encoding. Signed arithmetic is limited to subtraction and negation. Equality and
inequality compare integer values after sign- or zero-extension. Ordered
comparisons with signed operands remain outside the profile. Signed helper ports
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
Const gate targets. Decimal data expressions are limited to the extrema, subtraction and multiplication profiles below.
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
allocated `qbit[]` or `qnum` target. Each term's indices must be distinct and in bounds.
Index zero is the low bit of the output register. Empty support is an identity
term and contributes global phase. Time and coefficients can be signed literals
or bound classical `real` values. Sparse coefficient literals also accept
an optional `+ 0j` suffix, as in `(-20) + 0j`.

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
and a literal sample count in 1..2048. The existing effect budget also applies.
Each sample is drawn with replacement using probability `abs(h) / lambda`,
where `lambda = math.fsum(abs(h))`. Its Pauli rotation uses signed evolution angle
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
`program.report["qmod_fractional_bits"]`. Fractional arithmetic beyond extrema, subtraction and multiplication, and signed extrema,
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

### Unsigned fixed-point multiplication

A fresh assignment can multiply unsigned values and binary-exact decimal
operands. The compiler multiplies the raw integers and adds their fractional
bit counts. A fractional result requires a terminal inferred-width `main`
output. For example, `3.5 * a` multiplies by the raw integer 7 and reports one
fractional bit. Divide the named probability key's raw result by two to read
the numeric value.

### Combined Type I cosine and sine transform

`qct_qst_type1(x)` applies an in-place transform on one initialized writable
register of one to four qubits. For a register of width n, the low
`2**(n-1)+1` basis indices form the normalized cosine block. The remaining
indices form the normalized sine block multiplied by `i`. The transform
reuses the bounded gate decomposition and requires no auxiliary qubits.

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(Path("tests/fixtures/classiq_qmod/qct_qst_type1.qmod"))
print(fq.run(program).statevector())
```

### Named state preparation constants

`prepare_state` and `prepare_amplitudes` accept a top-level `real[]` literal
constant as the vector and a top-level `real` literal constant as the bound.
These are immutable source values. Named vectors use the same width and
normalization checks as inline vectors.

### Phase flips controlled by qubit arrays

`control(x) { phase(pi); }` flips the phase of the all-ones basis state of an
initialized qubit array. `control(x[i]) { phase(pi); }` flips the phase when the
indexed qubit is one, using the existing Qmod index order. A multi-qubit phase
flip reuses the bounded array-control lowering and its clean workspace budget.

### In-place exponential state preparation

`prepare_exponential_state(rate, q)` applies one `RY` per qubit of an
initialized array. On zero input, the output probabilities are proportional
to `exp(-rate * k)` for basis integer `k`. An existing state receives the
same rotations without resetting the register. A signed real literal or a
bound finite real rate within the existing ±2**53 input range is accepted.

### Linear Y rotations

`linear_pauli_rotations(bases, slopes, offsets, x, q)` accepts three literal
arrays whose lengths match the initialized target array. This profile accepts
the Y axis as either `2` or `Pauli::Y`. The rotation angle on each target
qubit is `slope * x + offset` with unsigned `x`. The disjoint control array
is preserved. The lowering uses an offset RY and bit-weighted CRY gates.

### Inline unitary matrices

`unitary(matrix, x)` accepts an inline complex matrix of size 2 by 2 or
4 by 4 on a matching initialized qubit array. Matrix entries may contain
signed real and imaginary literals, such as `1j` or `((-0) - 1j)`. The
compiler checks unitarity, lowers to registered gates, and preserves global
phase in the supported phase-preserving target profiles.

### Flexible phase estimation

`qpe_flexible(lambda(power) { ... }, phase)` accepts one first-order,
single-step `suzuki_trotter` call in the callback. The time expression can
multiply numeric literals, `pi` and the static power parameter. The phase
output uses `qnum<n, UNSIGNED, n>` with one to eight bits. Allocate it before
the call and read the raw result divided by `2**n`. The controlled evolution
retains the phase of identity Pauli terms, followed by an inverse QFT.

### Hadamard test and hidden target storage

Writable scalar and array helper ports support a bounded controlled QFT and
a single-Hadamard `within` with one bound helper call in `apply`. A fixed-width
local qubit array must be allocated before use and dropped before return.
`drop` removes it from named readout while retaining its effect on the visible
marginal. Its physical wires remain in the executable state and cannot be
reused as clean work qubits.

### Bounded modular exponentiation

`modular_exp(n, a, x, power)` multiplies an initialized residue `x` by
`a**power` modulo `n`. The base and modulus are positive integer literals
with `n >= 2` and `gcd(a, n) == 1`. The disjoint unsigned target and
exponent registers support up to four and eight qubits, respectively.
The permutation fixes target encodings at or above the modulus. The
`allocate(ceiling(log(n, 2)), x)` width form accepts a literal modulus and
uses integer bit length.

### In-place Bloch encoding and inversion

`inplace_encode_on_bloch(data, target)` applies the Bloch rotations for
complete pairs of bound real data to an initialized qubit array or a matching
one-bit `qnum`. A function-scope `invert` reverses
the rotations and negates their angles. Both forms preserve the existing
register ownership and require no temporary qubits.

### Static single-qubit inversion

Function-scope `invert` reverses admitted static single-qubit gates and
in-place Bloch rotations, using the registered gate inverses. The body must
contain only gate operations. A one-element output array declared as
`qbit[1]` can be passed directly to single-qubit gates. `I`, `SX`, and
`RZ(pi, target)` are supported.

### Classical array helpers and static qubit views

A Qmod helper can receive a bound classical array as `real[]` or
`real[N]`. Pass a named array, a half-open static slice, or a list of
finite literals and bound elements. Helpers can use static array lengths
and loop indices to address elements in a bounded gate-only `repeat`.
The entry point still requires explicit lengths for classical arrays.

A writable scalar helper port can bind to a qubit array element. A
writable array port can bind to a whole qubit array or an ordered list
such as `{state[2], state[0]}`. Each element refers to its original wire.
Writable ports must be disjoint, including within a list. Gate-only
helper calls preserve that mapping across nested calls. Operations
requiring whole registers, such as scoped QFT, still require whole
registers. Fixed-width qubit output arrays and the `CZ` gate are also
supported in this profile.

`RZZ(angle, view)` accepts any two distinct locations in a static qubit
array, ordered view or half-open slice such as `q[i:i + 2]`. A function-scope
`if` can select a gate-only block with a static integer comparison, such
as `q.len > 2`. Bound helpers can scale a real array element by the
unshadowed constant `pi` for RX, RY and RZ.

Bound helpers can invoke the existing bounded `suzuki_trotter` operation
on initialized whole registers and `unitary` on initialized qubit arrays
or ordered views. For an inline matrix, the first view element is the
least-significant matrix bit, as for a whole Qmod array. A static
`repeat` may call a bound helper. Helper calls share a cumulative
expansion budget of 4096, including calls that emit no operations.

### Coherent reversible helper composition

An initialized unsigned integer register can bind to a writable helper
array without copying its storage. A static repeat can control an X
operation with a nonempty slice of that array. The controls and target
must be distinct wire locations even when they share a register.

Scalar comparisons `coin == 0`, `coin == 1` and their inequality forms
can coherently control a helper whose expansion consists of X and indexed
multi-controlled X operations. A `control ... else ...` applies the two
supported branches coherently. `invert` reverses checked gate/helper
expansions, taking registered single-qubit adjoints and retaining the
self-inverse indexed controlled-X operations. Neither composition admits
allocation or changes of register ownership. Scratch space uses the
existing clean multi-controlled-X scheduler and is included in budgets.

### Modular increments and storage bindings

A writable unsigned `qnum` supports `a += 1` and `a += -1` modulo its
declared or bound width. Both operations reuse indexed controlled-X gates
and can appear in reversible helper compositions. `apply_to_all(H, a)`
and `apply_to_all(Z, a)` act on all physical locations of a register/view.

Function-scope bindings can move an allocated owner into declared local
views, as in `x -> {low, high}`, and restore it with
`{low, high} -> x`. The first part receives the least-significant
locations. All split widths must be explicit and cover the source
exactly. A single unsized local array can infer the full source width.
Sources become unavailable until restored. A merge must reproduce the
original allocated owner's wire order. Restore outputs and close local
views before return. Destination owners cannot be overwritten. Bindings
reuse storage and preserve initialization checks through the shared
register verifier.

Nested coherent controls can select reversible helper compositions using
array elements or one-bit unsigned numeric selectors. Their physical
controls must remain disjoint from every target in the expanded helper.

### Bounded real phase programs and QAOA composition

Specialized helpers can apply `phase(expression, gamma)` with a bound
finite real coefficient. Expressions support indexed quantum bits,
unsigned numeric values, finite classical constants and bound array
elements, addition, subtraction, products, literal powers and division
by a nonzero classical real. Constants retain their global phase. The
adapter expands a real polynomial with the Boolean identity b*b=b,
bounded to 24 distinct quantum bits, degree eight per monomial, 256 terms and 20000 intermediate
monomial products. Indexed phase operations reuse registered phase-gate
lowering and the phase-preserving target checks.

A helper can map a single-qubit gate body with
`apply_to_all(lambda(q) { RX(beta, q); }, qba)`. Static loops support
bounded integer products and `floor(array.len / positive_integer)`.
A `qfunc(real, qbit[])` parameter accepts a named helper with matching
ports, specialized at ordinary call sites. Named callbacks retain the
existing recursion, alias, width and expansion checks.


Full-size 24-bit telecommunications originals have an explicit validation
entry point. Run each case in a fresh process to collect synthesis,
execution and reference timings together with peak process RSS.

```bash
PYTHONPATH=. python tools/validate_qmod_sparse_phase.py \
  --case network_traffic_optimization --scale 0.007 \
  --check-phase --output network-validation.json
```

The tool preserves the original width and layer count, compares every
complex128 amplitude with an independently constructed Circuit, and
checks the classical cost phase over the full input domain in chunks.
The 24-bit state buffer occupies 256 MiB. Routine tests check planning,
memory refusal and separate input/degree/expansion budgets.

### Flat quantum records

The Qmod frontend accepts flat qstruct definitions with scalar qbit,
one-dimensional qbit[N], two-dimensional qbit[C][R], fixed-width unsigned qnum<W> and qnum<W>[N]
fields. Explicit numeric modifiers must be False, 0. Fields must be unique, nonempty and fit
the 256-bit structural budget. Record allocation, record size,
whole-record Hadamard and bounded single-qubit maps reuse the existing
register storage. A record can be passed to a matching writable helper
or produced by a matching Output helper.

Declaration order fixes each field's consecutive logical locations,
with the first field occupying the least-significant part. Field array
indices retain the existing least-significant-bit convention. The owned
register declaration carries field kind, width and record type, and the
Compiler independently checks complete coverage and output name
uniqueness.

Named output registers are the disjoint fields, such as `v.f` and
`v.interaction`; numeric arrays expose each integer as `v.w[0]`, `v.w[1]`,
and so on. The first element occupies the least-significant field segment.
Array len counts elements; record size counts bits. Each numeric word keeps
its declared width and is decoded as an unsigned integer. Read bit fields with
`program.probabilities(result, registers=("v.f", "v.interaction"))`.
`program.report["qmod_records"]` retains the declared schema.
A bounded repeat may contain an admitted single-qubit map callback.
Numeric words and indexed numeric array elements can be read by bounded real
phase arithmetic. Whole fields and indexed numeric words support Hadamard;
direct scalar gates still require qbit fields or indexed qbit elements.
Nested records, signed/fractional numeric fields, numeric array arithmetic
without an element index, multidimensional numeric arrays and bit arrays above
rank two remain outside this profile.

For a bit field `grid:qbit[C][R]`, the outer dimension is R rows and each row
contains C bits. `grid.len` is R and `grid[i].len` is C. Rows occupy consecutive
low-to-high logical segments and columns retain the least-significant-bit
convention within each row. Both indices must be bounded static integers or
admitted loop indices. Scalar gates and phase arithmetic use `v.grid[i][j]`;
row views such as `v.grid[i]` support Hadamard and writable array helpers.
Whole matrices and rows are not scalar phase operands. The owned declaration
retains `("grid", "qbit[][]", C, R)` and the Compiler independently checks it.
Readouts expose disjoint rows as `v.grid[0]`, `v.grid[1]`, and so on, decoded
as unsigned bit patterns; no overlapping whole-matrix readout is introduced.

Run an unchanged original portfolio program without designing gates or decoding
physical bit strings:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

source = Path("tests/fixtures/classiq_qmod/portfolio_optimization.qmod")
program = synth.synthesize(
    source,
    classical_inputs={"params": [0.003, -0.006, 0.009, -0.012, 0.015, -0.018]},
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
distribution = program.probabilities(result, registers=("v.w[0]", "v.w[1]", "v.w[2]"))
print(sum(distribution.values()))  # Approximately 1; integer values range from 0 to 7.
print(program.report["resources"])  # 12 qubits, 271 operations, depth 51.
```

The original electric-grid program uses `x:qbit[4][3]`. Run all four QAOA
layers and read the first row without physical wire numbers:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    Path("tests/fixtures/classiq_qmod/electric_grid_optimization.qmod"),
    classical_inputs={"params": [0.003 * (i + 1) * (-1 if i % 2 else 1) for i in range(8)]},
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("v.x[0]",))))
print(program.report["resources"])  # 18 qubits, 373 operations, depth 54.
print(program.report["workspace_qubits"])  # 0.
```

Full original record examples can be verified explicitly with
`PYTHONPATH=. python tools/validate_qmod_records.py --case ising_model --scale 0.003 --check-phase --output ising-validation.json`.
The tool uses Python's arithmetic AST parser and an independent closed
classical evaluator, exact rational finite differences and directly
constructed Circuits. It does not import the Qmod parser or phase
normalizer for its reference costs.

### Bounded predicates controlling real phases

A single real phase operation may be guarded by a predicate over unsigned
integer words, record fields or scalar bits. Supported comparisons are
`==`, `!=`, `<`, `<=`, `>` and `>=`; Boolean comparisons/bits compose with
`&`, `|`, `^` and `~`. Predicate arithmetic uses exact integers with bounded
coefficients, including subtraction, multiplication and literal powers in 1..8.
Classical thresholds must be integral. Fractional predicate arithmetic and
division are rejected. Intermediate integer arithmetic has no implicit word-width
wraparound. At most eight distinct predicate input bits are admitted, including
reads removed by cancellation.

The predicate and phase may read the same storage. This read sharing applies to
the diagonal phase operation; non-diagonal controlled operations retain their
existing alias rules. The real-expression path combines the Boolean indicator
with the existing phase polynomial and reuses checked diagonal lowering. Its
union of input bits, degree, terms and products retain the existing phase budgets.
A constant cost is multiplied by the predicate, preserving relative phase
between selected and unselected amplitudes. Reads remain initialization-checked
when the cost or predicate cancels. Nested control and multi-operation control
bodies remain outside this real-phase slice.

```qmod
control (2*v.a + 3*v.b <= 12) {
    phase(-(3*v.a + 5*v.b), gamma);
}
```

Classical real arrays support a static half-open slice followed by one element
index, such as `params[0:3][i]`, in bound real helper arguments and gate angles.
The element index may be an admitted loop index. Both the slice and its element
are checked against their own bounds; empty slices and negative indices are
rejected. This introduces no quantum-dependent host indexing.

Run the unchanged original Knapsack program, including helpers and three layers:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    Path("tests/fixtures/classiq_qmod/qaoa_knapsack.qmod"),
    classical_inputs={"params": [0.003, -0.006, 0.009, -0.012, 0.015, -0.018]},
    max_qubits=5,
    max_workspace=0,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("v.a", "v.b"))))
print(program.report["resources"])  # 5 qubits, 856 operations, depth 721.
```

`python tools/validate_qmod_knapsack.py --case qaoa_knapsack --scale 0.003 --output knapsack-validation.json`
checks the complete original against a direct Circuit with dense 32-state
diagonal phases and RX mixers. Its reference does not use the predicate or
polynomial compiler. The other two retained cases are `qaoa` and
`algo_design_QCE_tutorial_part_II`.

### Checked qperm calls and fixed Output arrays

The parser retains a distinct `qperm` marker. Before conversion to owned records,
the source boundary checks its complete admitted call graph, including ordinary
qfunc helpers, both control branches, repeat/map bodies, inverse and within/apply
scopes. The admitted subset preserves computational basis labels up to diagonal
phases: existing X, Z, S, I, RZ, CX, CZ and SWAP operations, arithmetic assignment
and XOR, in-place unit increments/decrements, checked binding and fresh allocation.
Width, freshness, aliasing and ownership still use the existing Compiler checks.
H, generic rotations and unproved calls are rejected, including indirect calls
and unused qperm definitions. Algebraic cancellation of unsupported operations
is not assumed. Dynamic callbacks and unimplemented port roles remain outside
this conservative qperm subset.

A fresh helper Output may be a fixed one-dimensional `qbit[N]` array. Its array
kind and width remain intact through the call, including N=1. Caller and callee
must match exactly. Duplicate writes, overlapping outputs, already initialized
actuals and escaped output storage are rejected. This does not introduce
unsized Output arrays in helpers, Input ports or general Const array/record ports.
The checked static phase-oracle profile below is the bounded Const-array exception.

`multi_suzuki_trotter([SparsePauliOp {...}], times, order, repetitions, target)`
accepts one inline sparse operator with order 1/2 and 1..256 repetitions, or
2..16 paired operators/times with order 1 and exactly one repetition. `times` can
be a bound real array, a checked slice or a finite classical-expression array.
The operator and time lengths must match. All forms reuse existing Pauli
lowering, widths, global phases and expansion limits; no term is discarded.

Run either unchanged molecular example with one nonzero time:

```python
from pathlib import Path

import flagquantum as fq
from flagquantum import synth

program = synth.synthesize(
    Path("tests/fixtures/classiq_qmod/molecular_energy_curve.qmod"),
    classical_inputs={"params": [0.37]},
    max_qubits=1,
    max_workspace=0,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("state",))))
print(result.statevector())
print(program.report["resources"])  # 1 qubit, 1 operation, depth 1.
```

The original `discrete_quantum_walk.qmod` needs no classical binding. Load it
with synthesize and read registers `("x", "coin")`; it returns `(1, 0)` and
`(1023, 1)` with probability one half each. Its actual circuit uses 16 qubits,
150 operations and depth 142, including five clean work qubits.

`python tools/validate_qmod_permutations.py --case molecular_energy_curve --parameter -0.81 --output molecular-validation.json`
compares the full state against an independent X followed by RY Circuit.
Use `--case discrete_quantum_walk` without `--parameter` for the original walk
and its full-state/clean-workspace reference. Molecular energy curve and molecule
eigensolver are two original paths containing the same source blob; coverage is
reported by both paths and distinct blobs.

## Compact large Pauli data

Source acquisition is bounded at 512 KiB and 65536 tokens. The existing
expression, scope, definition, helper-expansion and 4096 owned-node caps stay
independent. One ordered Pauli factor translates to one owned
`qreg.pauli_rotation` effect; Compiler verifies and maps the physical gates.
The previous adapter-level expanded Pauli template is removed. Final qubit,
workspace, instruction/depth and execution-memory constraints still apply.

The unchanged 551-term Trotter and 2000-sample qDRIFT originals retain 12 target
bits and have full-state independent Circuit acceptance. Source-term order,
seed-zero draws, identity/global phase and numeric bit order are preserved.
The finite absolute-weight normalization uses math.fsum. Sampling stays at the
adapter boundary; no random state or external Qmod type enters Compiler.
See docs/api-changes/FQ-SYNTH-UP06-PAULI-20261105.md and the independent tool
`tools/validate_qmod_large_pauli.py`. This remains local corpus integration work.

## Bounded exact integers and static composition

Qmod main and ordinary bound helpers accept int and int[N] parameters. Helpers
may use inferred-length int[] arrays. Values must be exact non-Boolean Python
integers of at most 256 magnitude bits; arrays have 1..256 elements and exact
main-port lengths. Main values use the existing classical_inputs mapping.
No real-to-integer rounding or quantum-state read supplies a static value.

power(count) repeats a checked static body 0..256 times without exposing an
index. repeat retains its index. A loop may contain integer-comparison if
statements with gate-only bodies, ordinary checked helper calls and admitted
Pauli/unitary/phase operations. Allocation, consuming ownership, nested loops
and quantum-dependent static conditions remain outside this profile. Existing
scope, helper-expansion, owned-node and final resource caps are unchanged.

phase(expression) uses coefficient 1.0, with constant/global and bounded real
quantum phases sharing the existing lowering. Integer-to-real angle conversion
is allowed only within the exact real range. Quantum arithmetic never becomes
host classical data. Pauli targets can be static ordered views in a single
writable owner, with distinct locations and checked local width/bit order.
Mixed-owner Pauli targets and unsupported multi-operator formulas fail closed.

The three unchanged Fermi, BB84 and credit-card originals have independent full
state-vector acceptance. Run tools/validate_qmod_integer_composition.py or
python -m examples.synthesis_integer_composition. The example also shows the
existing native qfunc equivalent; Python qfunc parameter syntax is not extended
by this Qmod adapter slice. Arbitrary classical execution and cscope/VQE/save
remain separate unsupported dependencies. Top-level exact integer literal
constants are admitted by the checked phase-oracle slice below.

## Checked phase oracles and uniform Grover composition

One whole Const qbit array may be borrowed by a phase-only function, optionally
with exact scalar integer arguments specialized from literals or top-level int
literal constants. Only phase effects, one-predicate phase controls and calls
forwarding the same borrowed register are admitted. Writes, allocation, Output,
ownership transfer, quantum captures, slices and additional array alias ports
are refused. Definitions and callback structure are checked even at power(0).

grover_operator accepts a named checked phase function or one-argument static
lambda, the built-in hadamard_transform and one whole writable array. The caller
prepares the initial state explicitly. Each round applies the oracle followed
by 2|s><s|-I; its global minus sign is retained. This agrees with the Classiq
[Grover definition](https://docs.classiq.io/sdk-reference/qmod/functions/open_library/grover).
The adapter expands these constructs into existing owned phase and gate effects;
Compiler still owns validation, workspace and physical costs. No truth-table
oracle constructor or foreign SDK executes during synthesis. General callback
ports, custom preparations and predicate-output phase_oracle calls are separate
dependencies, not enabled by this profile.

Run python -m examples.synthesis_grover for executable Qmod/native qfunc/Circuit
comparisons. Native qfunc composes existing phase functions and Hadamard calls;
this slice adds no Python Grover export. The unchanged 3_sat_grover.qmod and
grover_max_cut.qmod fixtures use r=0/1/2 with independent dense Circuit and
analytic full-vector references. Their bit order, absolute phase, clean scratch,
source/license digests and real budget refusals are checked. Use
tools/validate_qmod_grover.py --case 3_sat_grover --iterations 2 --output /tmp/sat.json
and repeat for grover_max_cut. See the contract
docs/api-changes/FQ-SYNTH-UP06-GROVER-20261105.md.

## Checked predicate phase oracles

The Qmod integration profile admits a qperm with one Const whole array or flat
fixed unsigned record, one distinct writable single-bit marker and optional
exact scalar integer arguments. Its body must be exactly marker ^= predicate;
the expression must prove a one-bit result and cannot read the marker. All
definitions are checked, including unused definitions. Nominal record identity,
scalar field widths and lexical bindings are retained; field names are not
renamed when a same-named callback formal shadows its own actual register.

phase_oracle accepts a named checked predicate or a two-argument forwarding
lambda. The phase form marks satisfying states by -1. A direct predicate call
XORs any initialized marker, including nonzero and coherent states, using the
existing conditional phase effect between marker Hadamards. There is no new
owned opcode or physical gate template. Capture, input/marker aliases, extra
mutable ports, Output escape and malformed predicates fail closed.

Mathematical negative differences are preserved through comparisons. Unsigned
bitwise arithmetic and positive power-of-two literal modulo are evaluated only
within the existing eight-input-bit predicate bound; monomial, degree, product,
source and final instruction caps are not raised. Larger graph-coloring and
whitebox predicates remain separate resource/semantic blockers, not implied
coverage. grover_search composes uniform preparation with bounded checked Grover
rounds and retains their absolute global phase. General callback ports, custom
preparation and nested/array-valued borrowed records remain outside this slice.

Three unchanged originals are accepted separately: grover_operator.qmod,
oracles_workshop.qmod and grover.qmod (r=0/1/2). The workshop subtraction does not
wrap; the graph original marks >=4, unlike the previous <=4 Max-Cut original.
Run tools/validate_qmod_predicate_oracle.py --case grover --iterations 2
--output /tmp/predicate.json for independent full-state Circuit comparison.
python -m examples.synthesis_predicate_oracle compares Qmod, existing native
qfunc phase intent and an independent Circuit. It does not add a native Python
phase_oracle/Grover export or a Python QStruct API. The private contract is
docs/api-changes/FQ-SYNTH-UP06-PREDICATE-ORACLE-20261105.md.

## Bounded decision predicates

The integration profile retains the eight-bit local phase evaluator. A budget
refusal can use a separate twelve-input decision profile with at most 262144
integer evaluator steps, 4096 reduced decision nodes and 128 disjoint accepting
paths. Eight-bit leaves and shared equal branches produce signed cubes. Groups
of at most six literals reuse existing Boolean compute candidates. Each cube's
flags are uncomputed and released before the next; Compiler clean-tail scheduling
and clean MCX lowering own physical reuse. No new canonical IR or public export
is introduced. Existing degree, indexed-phase width, owned-node and final costs
remain independently bounded. Static eleven/twelve-bit Grover reflections reuse
this path; smaller reflections preserve their previous resource template.

The two unchanged graph-coloring and whitebox originals retain one and ten
rounds. Run tools/validate_qmod_decision_predicates.py with --case
whitebox_fuzzing (or grover_graph_coloring) and --output /tmp/decision.json.
References use an independent gate-level Grover Circuit and analytic amplitudes;
production still executes every original round. The largest validation-only
reference uses its own clean conjunction workspace and no dense operator matrix.
This is local CPU correctness evidence, not a performance or corpus-wide claim.


## Closed quantum callback composition

The private Qmod adapter additionally admits borrowed `qfunc(qbit[])` and
`qfunc(qnum,qbit)` parameters. Closed named functions and lambdas specialize to
existing first-order statements before oracle normalization. Quantum captures,
Output/ownership escape, aliases, wrong signatures, nested callback values and
recursive wrappers refuse. Expansion remains bounded by 4096 statements and
sixteen levels. Existing real callbacks keep their previous path. Only borrowed
Hadamard preparation is admitted in this wrapper within/apply profile; it is
self-inverse and emits preparation, apply, reverse preparation. Callback arrays
and general captured lambdas are not supported.

The unchanged hidden_shift.qmod and deutsch_jozsa.qmod originals have independent
public Circuit and analytic full-state checks, including absolute phase, numeric
bit order and clean workspace. Their numeric predicates reuse the local
eight-input budget, including exact integer min/max; dynamic array widths are
proven by literal allocation. Run tools/validate_qmod_quantum_callbacks.py with
--case hidden_shift (or deutsch_jozsa) and --output /tmp/callback.json. Run
python -m examples.synthesis_quantum_callbacks for the original Qmod, existing
native qfunc phase-composition and Circuit comparison. This adds no higher-order
Python qfunc annotations or native phase_oracle export. Evidence is local CPU
correctness, not a performance or full-corpus claim.


## Checked exact amplification and retained work

The private Qmod exact_amplitude_amplification profile accepts one initialized
zero target of one through seven unsigned bits, a closed sign oracle, a closed
product-RY preparation and a positive marked-subspace amplitude. It independently
checks that amplitude against the preparation and predicate mass to absolute
1e-12. It reduces the Grover angle using a retained hidden ancilla and admits at
most 32 rounds. The ancilla and any simultaneous Compiler scratch count toward
max_workspace; its allocation is counted once in total physical width. No early
release or reset is asserted. Existing hidden semantic data keeps its previous
accounting. Owned workspace markers must be exactly True.

inplace_prepare_state accepts normalized product probability vectors with
amplitude L2 factorization defect at most 1e-12. Existing exponential RY angles
are reusable in this checked composition. Nonproduct/complex preparations,
quantum captures, consuming/Output callbacks, non-sign oracles, nonzero or
unproved-zero input, unsupported scopes and over-budget requests refuse.
Unsigned literal += offsets in -256..256 reduce modulo register width and
compose the existing +/-1 kernel; they are not a new optimized adder template.
Zero offsets still check initialization. All physical costs remain budgeted.

The unchanged exact_amplitude_amplification.qmod has independent Circuit and
analytic full-state acceptance. The unchanged partial_exponential_state_preparation.qmod
has a supplied amplitude 0.996625424765961 inconsistent with the approximately
0.9979545784437375 mass of its exp(+x/2), x>=21 preparation. It deliberately
refuses; a clearly labelled corrected diagnostic copy tests that full path,
including x += -18, without increasing original coverage. No original parameter
is silently changed. Run tools/validate_qmod_exact_amplification.py with --case
exact_amplitude_amplification (or partial_exponential_state_preparation) and
--output /tmp/amplification.json. The second choice validates the corrected copy
and the original refusal. Run python -m examples.synthesis_exact_amplification
for original Qmod, existing native R/phase composition and Circuit comparison.
Native rounds and its auxiliary output are explicit; this adds no native exact
amplification export or native for-loop support. Evidence is local CPU correctness.


## Scalar-controlled Pauli evolution and classical angles

The experimental Qmod boundary supports one scalar unsigned bit (including an
indexed qbit array element) controlling pure Pauli evolution, with literal
zero/one polarity and an optional else branch. Static helpers reuse the same
specialization and existing owned Pauli effects. Identity terms retain their
relative phase. Nested controls, allocation or mixed effects inside these
branches, overlapping control/target storage and signed numeric controls refuse.
Positive Const controls borrow storage; negative/else controls require writable
storage because the existing implementation uses paired X operations. Signed
argument metadata is preserved through bound helper calls.

Finite classical angle expressions support +, -, *, /, pi and bound real inputs.
Matrix, probability and state-preparation literal domains retain their previous
scanner and checks. No quantum-dependent angle is admitted by this profile.
Run python -m examples.synthesis_controlled_evolution for the unchanged BPDE
Qmod, native phase-composition qfunc and independent Circuit full-state comparison
at two nonzero delta values. Native steps are explicit; this adds no native
Hamiltonian-array export or Python for-loop support. The original's dropped
Hamiltonian target is semantic data, not a promised zero auxiliary register.
The separately recorded partial-exponential amplitude mismatch remains refused;
no original corpus source or baseline is rewritten. Evidence is local CPU only.


## Checked swap test and source-data refusal

The experimental Qmod adapter composes swap_test from existing Hadamard and
indexed controlled-X effects. Equal nonempty initialized writable input views
and one fresh scalar qbit Output must have disjoint locations. Existing owned
allocation and Const checks remain authoritative, including scalar Output
forwarding through static helpers. No public native swap_test export is added.

The pinned swap_test.qmod has amplitude squared norms 1.1072 and 1.1076 with
zero preparation error bound, violating the official unit-norm contract. It
remains refused; no automatic normalization or weakened verifier is introduced.
A separately archived, normalized diagnostic copy checks the complete path
against independent Circuit permutation and symmetric/antisymmetric algebra.
It contributes zero original execution coverage. Run tools/validate_qmod_swap_test.py
--output /tmp/swap-test.json. Run python -m examples.synthesis_swap_test for valid
one-bit Qmod, existing native phase/CX qfunc composition and Circuit comparison.
Dropped input states are retained semantic data, not clean workspace. This is
local CPU correctness evidence only. Existing partial-exponential source-amplitude
refusal and historical baselines remain unchanged.


## Lexical binding-only within/apply

The experimental Qmod adapter supports a within preparation consisting only of
register -> bindings, followed by bounded static gates and specialized gate
helpers in apply. It restores incoming view, width and suspended-name metadata
on scope exit; temporary aliases cannot escape. This is a lexical view scope,
not allocation, numerical inversion or a proof that arbitrary quantum data is
zero. Allocation, initialization, release and other ownership-changing apply
effects refuse after specialization. Existing arithmetic and Hadamard within
paths retain their contracts. Static loops in other quantum scopes are unchanged.

Whole registers and scalar/static indexed locations can concatenate into one
helper array actual, preserving low-to-high source order and existing alias
checks. Bound helpers may declare an unallocated unsized qbit[] alias. All
source initialization, Const mutation, width, loop shadow and resource budgets
remain checked. Run tools/validate_qmod_binding_scopes.py --case 0 --output
/tmp/autoencoder.json (also case 1) for the unchanged quantum_autoencoder.qmod
with two nonzero parameter bindings and independent Circuit full-state comparison.
Run python -m examples.synthesis_autoencoder for the original Qmod, existing
scalar native qfunc composition and Circuit comparison. No native array/loop
export or numerical kernel is added. Dropped coded data retains physical state;
it is not zero workspace. Existing source-semantic refusals and baselines remain.


## Checked static amplitude-amplification iterations

The Qmod boundary admits amplitude_amplification(reps,oracle,space_transform,target)
with named single-array callbacks and bounded static repetitions. It composes
existing preparation and Q=-A*S0*A^-1*S_chi without changing the original global
minus sign. Oracle validation covers indexed Z and the checked pi sign-phase
profile, including zero-round and unused definitions. Closed +,-,* index arithmetic
and array length preserve actual bounds; fractional phases, captures, writes and
Output escape refuse. Existing pure phase/pair effects invert in reverse order.
No amplitude parameter or state-normalization repair is inferred.

Untyped array/slice arguments may borrow a flat record layout only at its exact
width; named records remain nominal and signed/fractional casts refuse. A local
flat record retains semantic storage like an existing local array. Binding-only
apply accepts the same checked pure unitary effects; ownership changes refuse.
X apply_to_all and phase-flip views reuse indexed lowering; whole-register routes
remain unchanged. Static loops and physical/effect/helper budgets remain bounded.

Run tools/validate_qmod_iterations.py with --case quantum_counting (also
quantum_counting_iqae or qmci_pi_estimation), --repetitions 0/1/2 and --output
/tmp/iterations.json for unchanged originals against independent Circuit and
analytic full vectors. Run python -m examples.synthesis_amplification_iterations
for native scalar phase/CX qfunc, Qmod and Circuit comparison. Native rounds are
explicit; no native amplification, array or mutable helper export is added.
Dropped problem data remains semantic, not clean workspace. Actual scratch is
checked separately. Source-semantic refusals and historical baselines are retained;
this is local CPU correctness evidence only.
