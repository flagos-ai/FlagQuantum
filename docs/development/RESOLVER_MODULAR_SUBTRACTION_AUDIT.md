# Complete modular negation and subtraction constructor closure

## Source and interfaces

The complete mod_arithmetic/mod_subtraction.py and its original tests remain
unchanged at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. All four
constructors and all four original examples are assigned. Existing complete
whole-file evidence in MODULAR_SUBTRACTION_FILE_AUDIT.md is reused.

Compiler formats/legacy_modular_subtraction_factories.py owns constructor checks
only. The existing modular_negation_component, controlled_modular_negation_component,
modular_subtraction_component and controlled_modular_subtraction_component own
all execution decisions. Both uint and Montgomery inputs retain their restored
wire dtype. Controlled variants retain both scalar binary polarities. Ordered
ports are x, optional y and leading ctrl. All words are MSB first, through-going
and caller-owned. Symbolic dtype or modulus inputs retain bodyless declarations.
Finite execution continues through the normal circuit.append(op) path.

ModSub and CModSub reuse exact forward/inverse AddK loaders around their existing
modular addition child. The inverse loader is the already accepted exact adjoint,
including specialized erasure and cleanup. ModNeg retains its shared nonzero
flag and controlled fanout. CModNeg retains the paired measured conjunction and
zero tests. The constructor introduces no alternate arithmetic implementation.

## Dependency, phase and resource acceptance

The actual serialized four constructors bind to the identical Programs as their
previously accepted native owners. Every source gate, operand, measurement slot
and feedback matches the unchanged source traversal. Both control polarities and
uint/Montgomery views are exercised. All four symbolic source examples preserve
the original register declarations. A missing or invalid dtype, control value,
constructor field or signature fails before execution.

Existing full source-owner tests retain complete residue-column channels,
complex spectator amplitudes, every small raw measurement history and restored
private work. Exact bound-program equality connects this constructor lane to
those accepted semantics. The full owner regression is rerun within this batch.
Every actual T-family, measurement and reset inventory is scanned separately
from the unchanged QEC model. Actual T-family count is four times its And count
for these decompositions, and actual measurements match its measurement model.
Peak/lifetime and model formulas remain independently covered by the source
owner evidence.

## Source boundaries

The original modulus+1 loader fails when that constant cannot fit the operand.
The constructor retains that refusal in all four finite graphs. Previously
recorded exterior-residue dirty-flag witnesses, callback/decomposition domain
differences, strict source dtype issues and symbolic physical refusal remain
explicit. Montgomery dtype serialization loses its optional modulus, so finite
wire objects use their restored width-only descriptor. The separate ModAdd
constructor wire-signature mismatch remains recorded in its own audit and is
not repaired by these wrappers.

These four complete identifiers raise locally accepted catalogue coverage to
75/303, with 228 remaining after publication. Generic Adjoint/Controlled,
arbitrary-target Pauli, partial OnEach and ModAdd, pending Proposals 116-119 and
whole-repository/GPU/performance acceptance remain open. Q-0876 remains open.
Batch receipts use ftqc-reference-env/g2-modular-subtraction-constructor-*.

## Local batch acceptance

The concentrated native regression passes 1,379 cases, including the 21 focused
constructor cases. Local integration passes 145 cases. The complete unchanged
original source test file passes 303 cases, with two original strict-type skips,
two expected failures and 125 slow/notebook cases deselected. The existing
whole-file intake evidence retains those larger examples at its recorded scope.
Seven frozen source blobs and the real Kickmix 0.1 Simulator are verified.
Black, Ruff, strict typing for both changed Python files, Compiler scope,
repository language, architecture, IR and public API checks pass. The dependency
checker retains exactly the same 720 baseline findings. Publication is pending
the separately recorded Git HTTPS transport failure.

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
