# G2 upstream-defect acceptance decision

Status: approved by the user on 2026-11-26, limited to Q-0649, Q-0766 and
Q-0924. No implementation contract or other parent acceptance is changed.

The integration preserves Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. These three parents previously
remained open solely for reproduced upstream defects after complete adaptation
and source-test replacement. The user explicitly accepted their faithful
migration, with the original defects retained as separate optimization work.

## Accepted complete source responsibilities

| Parent | Complete migration evidence | Original defect still reproduced | Approved disposition |
|---|---|---|---|
| Q-0649, `qsp/fast_qsp.py` | [Complete definition/dependency audit](QSP_OPTIMIZATION_FILE_AUDIT.md), published `934a21028`; all 28 original cases pass with the native solver substituted. Objective operations, RNG consumption, optimizer arguments and returned iterates match. | Real-mode RNG 42/91, lengths 4/8: both original and native residuals are about 0.02339667/0.00545494. Original unsuccessful iterates are returned unchanged. | Faithful migration accepted; accuracy and optimizer-success policy remain separate work. |
| Q-0766, `linalg/polynomial/qsp_testing.py` | [Sampling and grid audit](QSP_POLYNOMIAL_CHECKS.md), published `c7583e59`/`3b0450ec`; all five source functions, lazy FFT, dtype/failure order, RNG, actual replacement and original tests have recorded evidence. | The four-point FFT check accepts coefficients `[0.53, 0.53*exp(i*pi/4)]` although their between-grid peak is 1.06. Both implementations return the same sampled result. | Faithful sampled-contract migration accepted; a continuous-circle certificate remains separate work. |
| Q-0924, `surface_code/qec_scheme_test.py` | [Complete source/test and consumer audit](SURFACE_CODE_SCHEME_FILE_AUDIT.md), published `f1968a200`; all four original tests pass; scalar formulas and four production consumers retain identical results. | At physical error `1e-4`, budget `1e-7`, both select distance 5 and compute `1.0000000000000002e-7`, exceeding the exact floating budget. | Complete original-test responsibility accepted; conservative budget rounding remains separate work. |

The three audits and their existing counterexample tests are reused. The
unchanged cases are not rerun merely to recount them. No claim of a rigorous
polynomial certificate, successful optimization or strict budget satisfaction
is made.

## Approved disposition and limits

Close only the three reviewed parents above, whose complete source surface,
dependencies, original assets, actual consumers and applicable comparisons
are already verified. Keep every original
defect visible with its reproducible input, user impact and owning native
module. Preserve the failure in regression coverage and the locked source.
Algorithmic improvements remain separate work and require their own contract
and validation where applicable.

This decision does not accept missing native calls, unresolved constructors,
missing physical execution for an executable source path, absent test assets,
uncertified native-only behavior, or unapproved protected contracts. It does
not automatically close any other parent. In particular, the remaining Hubbard
generic-constructor and external-capture scope is not waived by this decision.

Any later stronger mathematical guarantee must be specified independently.
Exact source parity and a corrected algorithm must be reported as distinct paths.
The frozen reference must never be patched to make either path appear equal.

## Explicit extension approved on 2026-11-27

The user subsequently approved the full
[G2 faithful-migration closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md).
The three-parent decision above remains the historical evidence record; it is
no longer the only authorized disposition. Review each additional complete
source owner separately. Missing native implementation, valid-source calls,
execution, assets and failed mandatory checks remain open. Approval alone
closes none of the remaining parents.

Q-0477 is individually reviewed and accepted for complete faithful migration.
Its [full audit](QROAM_CLEAN_FILE_AUDIT.md#faithful-migration-acceptance-on-2026-11-27)
retains the source's model-only erasure, omitted selectors, inactive classical
control failure and independent cost discrepancies with owning follow-ups.
Native paper erasure remains separately specified and verified.
