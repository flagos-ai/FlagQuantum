# G2 upstream-defect acceptance decision

Status: proposed; no checklist status or implementation contract changes.

The integration preserves Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The current planning rule keeps
upstream defects open even after complete source adaptation and unchanged
source-test replacement pass. This requires a decision between completion of
the faithful migration and improvement of the original algorithms. Neither is
silently substituted for the other.

## Concrete candidates for review

| Parent | Complete migration evidence | Original defect still reproduced | Decision requested |
|---|---|---|---|
| Q-0649, `qsp/fast_qsp.py` | [Complete definition/dependency audit](QSP_OPTIMIZATION_FILE_AUDIT.md), published `934a21028`; all 28 original cases pass with the native solver substituted. Objective operations, RNG consumption, optimizer arguments and returned iterates match. | Real-mode RNG 42/91, lengths 4/8: both original and native residuals are about 0.02339667/0.00545494. Original unsuccessful iterates are returned unchanged. | Accept faithful migration while retaining these accuracy defects as optimization work, or require a new solver policy before parent acceptance. |
| Q-0766, `linalg/polynomial/qsp_testing.py` | [Sampling and grid audit](QSP_POLYNOMIAL_CHECKS.md), published `c7583e59`/`3b0450ec`; all five source functions, lazy FFT, dtype/failure order, RNG, actual replacement and original tests have recorded evidence. | The four-point FFT check accepts coefficients `[0.53, 0.53*exp(i*pi/4)]` although their between-grid peak is 1.06. Both implementations return the same sampled result. | Accept the original sampled contract while retaining the absent continuous-circle certificate as optimization work, or require a new certification algorithm before parent acceptance. |
| Q-0924, `surface_code/qec_scheme_test.py` | [Complete source/test and consumer audit](SURFACE_CODE_SCHEME_FILE_AUDIT.md), published `f1968a200`; all four original tests pass; scalar formulas and four production consumers retain identical results. | At physical error `1e-4`, budget `1e-7`, both select distance 5 and compute `1.0000000000000002e-7`, exceeding the exact floating budget. | Accept the unchanged original-test responsibility while retaining conservative budget rounding as optimization work, or require a separately specified numerical change before parent acceptance. |

The three audits and their existing counterexample tests are reused. The
unchanged cases are not rerun merely to recount them. No claim of a rigorous
polynomial certificate, successful optimization or strict budget satisfaction
is proposed.

## Proposed disposition

If faithful migration is the acceptance target, close only reviewed parents
whose entire source surface, dependencies, original assets, actual consumers
and applicable source comparisons are already verified. Keep every original
defect visible with its reproducible input, user impact and owning native
module. Preserve the failure in regression coverage and the locked source.
Algorithmic improvements remain separate work and require their own contract
and validation where applicable.

This proposal does not accept missing native calls, unresolved constructors,
missing physical execution for an executable source path, absent test assets,
uncertified native-only behavior, or unapproved protected contracts. It does
not automatically close any other parent. In particular, the remaining Hubbard
generic-constructor and external-capture scope is not waived by this decision.

If stronger mathematical guarantees are required for G2, retain the existing
parent statuses and specify each improvement independently. Exact source
parity and a corrected algorithm must then be reported as distinct paths.
The frozen reference must never be patched to make either path appear equal.
