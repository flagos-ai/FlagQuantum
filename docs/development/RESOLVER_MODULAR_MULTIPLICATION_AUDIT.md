# Complete modular multiplication constructor disposition

## Source, interfaces and ownership

The complete mod_arithmetic/mod_multiplication.py and original test file remain
unchanged at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. The existing
MODULAR_MULTIPLICATION_FILE_AUDIT.md assigns all 45 definitions and methods,
all six examples and notebook assets. Its complete original 1,959-case intake
and accepted channel, append and resource evidence are reused.

The format owner legacy_modular_multiplication_factories.py handles only named
constructor arguments and signatures. ModDbl delegates modular_doubling_component;
CModMulK delegates modular_multiplication_component. SingleWindowModMul delegates
single_window_component. The private _DirtyOutOfPlaceMontgomeryModMulImpl delegates
windowed_multiplication_component, and DirtyOutOfPlaceMontgomeryModMul delegates
montgomery_multiplication_component. Shared arithmetic, exact controlled swap,
QROAM and measurement-assisted erasure retain their existing owners.

Word registers are MSB first. The single-window x array is low-order first, while
target storage is MSB first. ModDbl returns a changed logical x binding and
releases the old high bit after cleanup. CModMulK retains its positive control
and owns a temporary result until inverse-scale clearing and exact swapping.
Montgomery core ports are through-going. The wrapper allocates RIGHT product,
history and reduction ports, or consumes the matching LEFT ports when uncompute
is set. Symbolic inputs retain bodyless declarations, including symbolic array
shapes admitted by approved Proposal 120. Execution uses circuit.append(op).

## Constructor and codec boundaries

ModDbl and CModMulK resolve through the original wire for unsigned and Montgomery
inputs, including all four word examples. The wire's dtype argument and declared
register both omit optional Montgomery modulus metadata, so these constructors
retain the restored width-only descriptor. Exact bound Program equality connects
both constructors to the certified existing owners. Full unchanged source
primitive operands, measurement slots and feedback are compared again.

The three window constructors reconstruct modulus-bearing signatures from mod,
but their original stored register descriptors lose that modulus. Direct native
constructor resolution retains the original source signature and equals the
existing graph exactly. Resolution of the original wire fails the declared
signature comparison. The generic resolver and dtype schema keep their existing
contracts. The original private implementation also lacks a built-in resolver
entry, so its native explicit mapping is tested separately. Window identifiers
remain partial until the codec mismatch has an approved, tested disposition.

The wrapper wire serializes boolean uncompute as integer zero or one. Both are
accepted and normalized to the existing boolean owner argument. Other direction
values refuse. Source permissive truth values remain a boundary. The private
core's non-divisible window declarations exceed the accepted executable owner
domain. Generic symbolic execution, invalid/exterior residues, source carry
wraparound and final reduction-add versus callback-subtract counterexamples
retain their existing recorded scopes.

## Phase, cleanup and independent resources

All previously accepted complex-state, spectator and all-small-history doubling,
controlled multiplication and lookup certificates are reused. Exact program
and bound-output equality proves the constructor introduces no execution change.
Window/Montgomery larger trajectories retain their three selected histories,
with no claim of exhaustive product-history coverage. Work is clean on the
recorded promised domains. The model-only source lookup adjoint still has no
physical primitive stream for a whole-window gate-by-gate comparison.

Actual doubling T-family/measurement counts equal its source model. CModMulK
requires seven T gates per exact swap bit to reconcile its primitive inventory.
Window erasure contributes independently measured gates and events where the
source supplies a model-only leaf. The source model and emitted inventory remain
separate, including the earlier whole-file audit's measured substitution tables.

Two complete identifiers raise local accepted catalogue coverage from 75/303
to 77/303 after publication. SingleWindowModMul and
DirtyOutOfPlaceMontgomeryModMul remain partial for the original wire mismatch.
Q-0576 and Q-0876 remain open for the listed whole-file and generic admission
boundaries. Receipts use ftqc-reference-env/g2-modular-multiplication-constructor-*.

## Local batch acceptance

The concentrated native suite passes 389 cases, including all 14 constructor
cases. Local integration passes 145 cases. The unchanged original source test
file passes 1,410 cases, with six original strict-type skips, two expected
failures and 549 slow/notebook deselections. Earlier complete 1,959-case intake
remains reusable at its recorded scope. Seven frozen blobs and real Kickmix 0.1
Simulator are verified. Both Python files pass strict typing, Black and Ruff.
Compiler scope, language, architecture, IR and protected API checks pass. The
dependency checker retains the identical 720 baseline findings. Publication
remains pending the separately recorded Git HTTPS transport failure.
