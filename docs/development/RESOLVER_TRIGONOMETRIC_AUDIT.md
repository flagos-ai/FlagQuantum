# Complete arcsine and arctangent constructor declarations

Both complete arithmetic/trigonometric/arcsin.py and arctan.py, including their
unchanged original tests, remain locked to Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Their entire definition, numerical,
foreign, presentation and resource disposition is already assigned in
TRIGONOMETRIC_FILE_AUDIT.md. The format owner legacy_trigonometric_factories.py
reuses ArcsineDeclaration and ArctangentDeclaration directly for all signatures,
validation and defaults. It owns constructor argument checks and names only.

ArcSin retains MSB-first unsigned fixed-point x and occupied result words,
num_iters and degree defaulting to four. Equality of fractional and total width
is allowed by the original despite its stricter error text. ArcTan retains the
original opaque select/target and bit sign layout, including its one-bit port
specialization. All ports are THRU. Concrete and symbolic signatures round-trip
through the original wire and equal their existing native declaration owners.

Both frozen roots lack primitive bodies. They therefore resolve to bodyless
Program declarations with explicit physical-lowering refusal. Symbolic ArcTan
raises the original DecomposeTypeError through its Cirq-style decomposition
route, while concrete roots report DecomposeNotImplementedError. The existing
normal circuit.append(op) lookup, polynomial and arithmetic constructions retain
their separately certified semantics and precision domains.

ArcTan's self-adjoint occupied-target XOR, its original missing generic
with_registers implementation and register-dimension refusals remain explicit.
ArcSin's original numerical scaling and raw callback overflow/error behavior
retain their accepted Simulation witnesses. Full complex-state, small raw
histories and clean-work certificates for independently constructed physical
owners remain at their previous scopes. No source root primitive sequence is
available for a whole-root gate-by-gate claim.

Independent native and original model regressions retain the ArcSin polynomial
and Newton iteration Toffoli formula. ArcTan creates a fresh symbolic c for the
multiplicity of PlusEqualProduct, whose declared estimate is eight times the
square of target width. This unresolved multiplicity remains symbolic.
Actual lookup/polynomial gates are counted separately from these estimates.
Bodyless declarations are never counted as zero-gate physical execution.

Two complete constructor declaration identifiers raise local catalogue
coverage from 86/303 to 88/303 after publication. Whole-file physical parity,
foreign protocols and all previously registered generic construction boundaries
remain open at their original parent items. Q-0876 remains open. Receipts use
ftqc-reference-env/g2-trigonometric-constructor-*.

## Local batch acceptance

The concentrated native regression passes 194 cases, including seven constructor
cases, complete declaration and source-file audits, original callback/foreign
witnesses and certified independent physical owners. Local integration passes
145 cases. The two unchanged original source files pass six cases, with three
original ArcSin no-decomposition/count/type skips and four slow ArcTan cases
deselected. Existing complete original intake remains reusable. Nine frozen
blobs and the real Kickmix 0.1 Simulator are verified. Both Python files pass
strict typing, Black and Ruff. Compiler scope, language, architecture, IR and
protected API checks pass. The dependency checker retains its identical 720
baseline findings. Publication is verified separately against the exact Git SHA.

## Empty-register constructor closure

The original ArcSin explicitly constructs both named THRU registers, including
at width zero. The native declaration now preserves those registers instead of
using the convenience signature builder that omits empty words. Actual serialized
source libraries for fractional widths 0, -1 and -2 resolve and verify with both
ports intact. Positive-width and symbolic constructor signatures retain the same
owners. ArcTan instead uses the original convenience builder; its empty select
or target ports remain omitted, while its sign bit remains present.

The declaration regression compares the original resource formula for all three
empty ArcSin profiles and three iteration/degree pairs. Negative estimates remain
raw source-model values, not measured primitive counts. No physical body is added:
default constructor resolution still refuses physical binding for both roots.
Existing independently implemented lookup and approximation evidence is unchanged.
In particular, the explicit native ArcTan lookup still requires positive input
and target widths even though original empty-profile operator oracles exist; that
separate execution gap remains recorded. Whole-file Q-0224 and Q-0226 remain open
for the previously documented numerical, decomposition and execution boundaries.
Receipts use `ftqc-reference-env/g2-trigonometric-empty-*`.
