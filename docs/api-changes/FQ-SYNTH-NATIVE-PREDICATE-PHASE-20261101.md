# Native one-bit predicate phases

The [documented Classiq phase grammar](https://docs.classiq.io/qmod-reference/language-reference/statements/phase/)
is polynomial-only. This proposal is a FlagQuantum-native predicate extension
to the import profile, not an additional upstream compatibility claim. Retained
Classiq sources and their evidence counts remain separate.

Following #994 (ea1d2da8d34ecd35ccf4b68a7222408d9f2b03af), admit
phase(expression,pi) for existing unsigned comparison expressions, including
comparisons nested in admitted arithmetic when the complete inferred result
domain is exactly one bit. Preserve all existing initialization, Const effects,
coefficient, scalar-width, target, source and resource constraints. Pure literal
phases, arbitrary angles, wider results, comparison chains and logical operators
remain outside the profile. Quantum-dependent fixed values retain their sign.

Ecosystem removes its polynomial-only rejection and the redundant comparison
tracking flag. Existing qreg.phase and Compiler own comparison kernels, selected
cleanup and resource accounting. No new API, IR, operation, executor, dependency
or lifetime mechanism. Explicit local scopes remain available for reusable values.

Acceptance: six unsigned relations over all sixteen two-bit pairs; both synthesis
objectives; coherent amplitudes, references, exact signs, interference and zero
hidden work; Const helpers and scopes; replacement of an explicit local producer;
budget/memory/target guards; existing invalid-source diagnostics. Preserve original
source/license hashes, four-original evidence, trials 0/2 and unreleased status.
