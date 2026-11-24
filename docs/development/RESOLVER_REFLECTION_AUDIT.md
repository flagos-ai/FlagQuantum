# Identity and prepared-reflection constructor disposition

Reference: complete `reflections/prepare_identity.py` and
`reflections/reflection_using_prepare.py`, plus their unchanged test files,
at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. All four Git
blobs match the locked inventory. Fixtures require the actual Kickmix 0.1
Simulator and Qualtran 0.7.0.

## Shared ownership

Typed identity selection composition now lives in Compiler preparation. The
existing Algorithms identity_prepare_program delegates to that owner after its
existing binding checks. Typed prepared-reflection composition also lives in
Compiler preparation. The existing zero_state_reflection_program retains its
signature, validation and IdentityPrepare requirement, then delegates. Compiler
does not import Algorithms, instantiate upstream classes or own numerical work.

The common reflection graph calls exact unprepare, marks all-zero selection,
then calls prepare. Selection words are MSB-first. Private junk arrays are
allocated and released row-major. In the source Cirq-to-Bloq graph the phase
target allocation occurs at its first use after unprepare; retaining the earlier
qalloc request order would change the comparable logical graph. The final
open-control X keeps the existing EARLIEST position around a relative phase.

Compiler formats owns the two original constructor contracts. Identity recipes
retain explicit typed selection registers. Other PREPARE children require an
explicit owner that supplies selection names and a complete exact inverse
helper closure. The enclosing resolver owns the already resolved forward
helpers; the new graph keeps that reference external instead of duplicating or
guessing its implementation. Unknown inverse owners refuse.

## Verification scope

All three original examples are covered. The original alias-PREPARE reflection
is compiled using the existing alias graph and exact inverse through an explicit
provider; its complete recursive calls and operands match the reference. This
does not add an automatic alias constructor or repair its source simulation
xfails. Alias metadata factory ownership remains a separate direct dependency.
The original zero-reflection example matches at serialized depths zero and one,
compiles with source imports blocked and binds to the existing exact primitive
stream. JSON and prior append-consumer checks remain in regression.

Small zero reflections compare full complex operators, all applicable erasure
histories and zero private work. A separate nontrivial PREPARE with array junk
and an explicit exact inverse tests private typed allocation and complete
complex output cleanup. This positive test does not establish clean junk for
every arbitrary PREPARE. The caller owns that semantic property. Existing
Algorithms consumers and complete source gate/feedback tests run unchanged.

## Preserved boundaries

The source reflection adjoint returns self for nonreal phases. That source
behavior is retained separately from mathematical inversion by conjugation.
Uncontrolled nontrivial scalar phase retains Proposal 119's existing refusal.
Shaped, one-sided, zero-width and symbolic identity declarations do not invent
source executable bodies. Generic PREPARE inverse and junk classification need
an explicit owner; unsupported classical/dtype admission stays explicit.

Actual T/Tdag, rotations, measurements and physical leases are reported separately
from source And/Clifford/rotation and logical peak estimates. Source equal-key
PREPARE model folding and original simulation/count xfails remain source-owned.
Generic constructor counts and parent acceptance are not incremented for this
concrete/external-owner disposition. No protected API or schema is changed.

## Acceptance receipt

Concentrated regression passes 286 cases without skips. Local integration
passes 145 cases. The final constructor selection passes 20 cases after the
optional NumPy collection guard was added. Strict typing covers six files;
Black, Ruff, Compiler and Algorithms scope, language, architecture, IR and API
checks pass. Dependency findings are exactly the existing 720 lines after that
guard correction, with no new dependency violation accepted.

The complete unchanged two original test files yield 18 passes, one original
skip and three original xfails; their notebook separately passes with local
Jupyter permission. The initial notebook sandbox denial is retained. No source
test body or reference implementation was edited.

Three-bit uncontrolled zero reflection emits eight T-family primitives, 12 CX,
two measurements and six peak wires versus source two And, 17 Clifford,
two measurements and six logical peak wires. The original six-bit controlled
example emits 20 T-family, 30 CX, five measurements and 12 peak wires versus
source five And, 31 Clifford, five measurements and 12 logical peak wires.
The original alias reflection emits 242 T-family, 630 CX, four Rz and 50
measurements with 41 peak wires versus source six CSwap, 50 And, 461 Clifford,
four rotations, 50 measurements and 41 logical peak wires. No private lease
remains live; alias numerical junk restoration is not asserted by that lifetime
inventory and its original simulation xfails remain explicit.

Detailed receipts use `g2-reflection-constructor-*-20261124` in the external
reference ledger. Earlier allocation-order and metadata-key diagnostics were
corrected before the complete passing regression; their failures were not
waived by changing operand matching or modifying the reference.
