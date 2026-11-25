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
signature comparison in the historical constructor batch. The typed-view
continuation below closes that native binding gap without changing the codec.
The generic resolver and dtype schema keep their existing contracts. The
original private implementation also lacks a built-in resolver entry, so its
native explicit mapping is tested separately.

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

At that historical publication, two complete identifiers raised local catalogue
coverage from 75/303 to 77/303. SingleWindowModMul and
DirtyOutOfPlaceMontgomeryModMul still had the original wire mismatch, closed by
the continuation below. Other recorded constructor domains remain separate.
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

## Publication verification

The accepted module is published in the seven-commit range ending at
05031474d26059a3c531269738aa7b16fa07a923 on PR #1084. The GitHub Git
database API preserved every local blob, tree and commit SHA and updated the
integration branch with force disabled. The verified publication supersedes
the historical local-only transport status above. The combined published
catalogue covers 85/303 constructor identifiers, with 218 remaining.
The per-module execution, symbolic, codec and model boundaries remain recorded
at their original scopes. Exact publication evidence is retained in
ftqc-reference-env/g2-git-object-publication-20261123.json.

## Source-wire Montgomery view interface

Approved Proposal 103 prerequisite 0.5 already admits optional Montgomery
modulus metadata, and existing Cast owns exact width-preserving nominal views.
The native window constructor now accepts only the source codec's demonstrated
omission: outer Montgomery ports have no parameters; the inner port has only
the constructor's modulus. Names, directions, shapes, quantum widths and
classical extents must otherwise match. Wrong explicit moduli, encodings,
widths, classical storage, shapes, directions, names and port counts still fail
before composition. This does not change Core admission, a schema, a public
entry point or any reference codec.

The outer Program retains the exact source-wire ports. Existing typed Casts
adapt its LEFT/THRU inputs, call the unchanged product owner and restore
RIGHT/THRU outputs. The views emit no gates, allocation, feedback or numerical
conversion. Source modulus arguments remain the child arithmetic authority.
Direct native construction with no supplied source signature retains its exact
previous Program. Symbolic source-wire declarations retain their actual outer
ports and remain bodyless and physically refused.

The internal through-going Montgomery core may release and replace caller
product storage. It still requires explicit input ownership transfer and
permission to return owned workspace. The public wrapper allocates these words
in the forward direction and consumes matching owned LEFT inputs in reverse.
Metadata views do not grant ownership or hide these requirements. Expanded
logical operation indices change when views are inserted; physical primitive
order, ownership wire/flag order and instruction offsets are compared separately.

## Window source-wire acceptance

Eight concrete direction/profile combinations retain exact original library
bytes and resolve at depths 0/1/2 to identical native Programs. Complete bound
CircuitIR equality with the previous direct owners proves unchanged primitive
operands, measurement feedback and slot identities, not merely matching counts.
Ordered output aliases and every lifetime's wire/dirty flag/instruction offset
also match. Direct native Programs with no outer signature remain exact.
Actual symbolic source libraries retain their wire signatures and physical
body refusal. Eight malformed-port profiles fail before typed composition.

Existing complete complex-state, source-callback/decomposition comparisons,
spectators, one-bit all-history and larger selected-history cleanup evidence is
reused through exact bound-circuit equality and rerun with the window owners.
No extra clean bit, primitive, measurement or classical result is introduced.
At n=4, w=2, p=5, a single window retains 231 CX, 100 T-family gates and 31
measurements; the source declares 23 And, 106 Cliffords and 23 measurements.
The Montgomery core and wrapper retain 533 CX, 228 T-family gates and 69
measurements; the source declares 53 And, 255 Cliffords and 53 measurements.
The existing model-only lookup erase substitution remains the explanation for
these separate accounting levels. Resource rows also distinguish allocated
work peak from returned owned outputs and original logical peak estimates.

The source's actual dtype codec still omits modulus. Its private-core built-in
reconstruction still raises ValueError; native explicit constructor mapping is
separate and succeeds. Frozen source and test blobs remain unchanged. Generic
truth-valued directions, symbolic physical execution, non-divisible private
windows and the previous arithmetic/lookup/source counterexamples remain open.
No whole-file acceptance or new global constructor count is inferred here.

The concentrated constructor/window/product/Cast regression passes 98 cases;
its final expanded constructor module passes all 26 cases in both Cirq lanes.
The independently selected window-owner Cirq 1.6 run passes 43 cases, and local
integration passes 145. Both changed Python files pass strict typing, Black and
Ruff; scope, language, links, architecture, IR and protected API checks pass.
The two frozen source/test blobs and real Kickmix 0.1 Simulator are reverified.
Receipts use `g2-window-codec-*-20261125`.
