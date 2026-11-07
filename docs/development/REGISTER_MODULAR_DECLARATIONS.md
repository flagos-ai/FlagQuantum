# Controlled-add and modular resource declarations

The locked Qualtran v0.7.0 commit is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler models describe original
logical resource declarations independently of emitted primitives. Production
code does not import the reference SDK.

## Shared CAdd model (G2-A42)

`arithmetic_resource_model.source_controlled_add_cost(m, n, control_value=...,
legacy_shims=...)` composes the existing predicate and Add models in the source
order: m And computations, one Add and m And-adjoint cleanups. Both control
polarities and both QEC profiles use the same underlying authorities. Unsigned
and Montgomery resource queries at 1 <= m <= n match the original source in
both directions. Symbolic dimensions and signed unequal-width mathematical
semantics are not added by this query.

The root has 1+m+n logical ports. At the last predicate call all earlier loaded
bits are still live. During Add, the original control and input word remain
bystanders. `CostCall` records those lifetimes and yields peak 2m+2n. The model
does not allocate wires or execute a numerical circuit.

The source one-bit Add declaration has -3 Clifford gates; consequently modern
one-bit CAdd has -2. This diagnostic is preserved, not presented as a physical
gate count. Existing emitted one-bit arithmetic remains unchanged. Modern
source QEC counts can omit negative-control X gates while the emitted stream
retains them; the selected profile is explicit.

Run `python -m examples.algorithms.register_controlled_addition` for the existing
append workflow with declared categories/peak and actual primitives printed
separately. It verifies coherent active/inactive branches, absolute phase and
zero work for both polarities. Tests compare all seven source categories and
widths for uint/Montgomery, directions and profiles, replace the shared child
models, and retain the original gate/complex-measurement/cleanup regressions.

## Modular placeholder declarations (G2-MOD21)

`modular_declarations.py` ports the four direct call declarations in the original
`arithmetic/_shims.py`. It uses existing flat register layouts, typed resource
callees and `SymbolicResourceFormula`, without a new model registry or SDK type.
G2-MOD21 covers concrete nonnegative widths. G2-MOD22 adds symbolic signatures
and aggregate formulas below. Python-integer negative widths remain invalid;
symbolic expressions describe source declarations without physical binding.

| Native declaration | Original source | Direct calls |
| --- | --- | --- |
| `multi_controlled_x` | MultiCToffoli | n-2 Toffoli |
| `controlled_subtraction` | CSub | CAdd plus three BitwiseNot |
| `less_than` | Lt | n Toffoli |
| `controlled_half` | CHalf | n TwoBitCSwap |

`modular_arithmetic_registers` preserves the source names and flat widths.
`modular_arithmetic_calls` returns native direct keys and literal multiplicities,
including symbolic widths. Arithmetic children retain their Montgomery encodings.
`source_modular_arithmetic_resource_formula` consumes the existing typed QEC
query and original signature/callee width fallback. CSub uses the shared CAdd
composition in the chosen profile; it does not duplicate predicate/Add math.
Lt's `signed` flag changes neither its original signature nor its declaration.

Zero call weights remain visible. The original MultiCToffoli with zero/one
controls declares -2/-1 Toffoli; those diagnostics remain declaration data and
are not passed off as executable repetition counts. The original width fallback
also includes the three-bit atomic callee at zero/negative multiplicity. Native
executable `CostCall` validation is unchanged. Zero-width CSub has failing source
QEC and width queries and is rejected by the aggregate formula; its register
metadata and successful direct source declaration can still be inspected.

All four original placeholders lack gate and classical decompositions. There
is therefore no source gate stream, complex-amplitude channel or measurement
history to compare for them. The native output is a formula, not an appendable
operation. The actual shared CAdd emitter continues to carry its independent
gate/state/clean-work evidence. The runnable example
`python -m examples.algorithms.modular_resource_declarations` prints declared categories,
peak and T equivalents; it makes no physical-gate or quantum-execution claim.
Tests compare original signatures, direct callees/multiplicities, every QEC
category, both profiles/directions, the signed flag and source failure boundaries.

## Symbolic CAdd declarations (G2-A43)

`controlled_add_resource_model.source_controlled_add_resource_formula` consumes
shared Add and predicate models. `source_add_resource_formula` now accepts the
same explicit QEC profile, including original And-adjoint cleanup counts. The
default queries and executable arithmetic remain unchanged.

Two Python integer widths dispatch to the existing composed CAdd model. Mixed,
symbolic and Sympy-integer dimensions retain the original undecomposable
signature/callee maximum. They do not gain fictitious loaded/carry storage. For
m=n=3 the original symbolic fallback is 7 logical qubits, while the concrete
execution model peaks at 12. Substitution into a symbolic formula and rebuilding
with Python integers are separate original paths. Sympy integers are not cast.

The source's symbolic metadata can also accept reversed literal widths even
though the concrete Add constructor rejects them; resource declarations retain
that distinction without claiming physical execution. One-bit negative Clifford
counts remain the original diagnostic. Tests compare uint/Montgomery, controls,
profiles and both directions, replace all shared child categories, and verify
both integer dispatch and the original symbolic width fallback.


## Symbolic modular signatures and aggregate formulas (G2-MOD22)

`modular_arithmetic_registers` and the aggregate formula now retain symbolic
widths, including Sympy integer literals. Concrete and symbolic aggregation
now both consume the shared native arithmetic recipes. Symbolic CSub consumes the shared CAdd declaration
and queries BitwiseNot through the existing typed `OnEachCall(X, n)` rule.
The original signature/callee width fallback is preserved. Other declarations
query their original Toffoli/Fredkin edges with the existing QEC counter.

The example accepts `--symbolic` and `--legacy-profile`. Tests compare every
source gate category and peak width in both directions/profiles, positive,
nonnegative and unconstrained integer symbols, expressions, literal zero/one/
three and Python integers. Lt's signed flag still leaves its original unsigned
signature and costs unchanged. Literal-zero CSub keeps its original query
failure, and concrete/symbolic peaks are not conflated after substitution.

The raw `modular_arithmetic_calls` interface now returns the same native
identity type for concrete and symbolic arithmetic children. It delegates to
`arithmetic_callees`, eliminating its duplicate source multiplicity formulas and
folded ComponentCost keys. The cost profile changes query results, not key
identity. No symbolic literal is cast. All four source shims remain declarations,
not executable implementations.

## Native graph consumers (G2-RES38)

Symbolic CAdd and all four modular aggregate formulas now query the same native
source edges. Existing Add/predicate/CAdd model implementations are injected as
static child costs through the established QEC query interface. This preserves
replaceability, including all seven category fields, without duplicating edge
multiplication. Concrete loaded/carry lifetimes and symbolic maximum-width
fallbacks remain separate from additive gate aggregation.

Tests replace native rows without changing these formula consumers, and separately
replace their existing child models. The n=3 symbolic peak of 7 versus concrete
peak of 12 remains unchanged. These are declaration/model tests; existing concrete
Add/CAdd execution tests independently verify complex states, feedback branches
and clean work.
