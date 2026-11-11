# Classical recursive failure context

The existing Simulation-owned `ComponentClassicalState` now preserves the
recursive `NotImplementedError` boundary of Qualtran v0.7.0
[`ClassicalSimState._recurse`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py).
Previously a supplied recursive evaluator's error escaped without identifying
the enclosing component. Each recursive boundary now raises a contextual
`NotImplementedError` from the original exception, preserving every cause.

Compiler still supplies dependency order; Simulation consumes the existing Core
record, numerical actions and explicit recursion callback. This change neither
discovers Python decomposition methods nor imports Compiler into Simulation.
No protected API or encoding admission changes. Users keep writing
`circuit.append(op)`; the existing append and hierarchical-adjoint regression
exercises that physical path independently of numerical failure handling.

## Exact scope

- The value callback runs before the phase callback. Their direct exceptions
  are unchanged, including a direct `NotImplementedError`.
- Recursion is attempted only for `NotImplemented` values and an absent phase.
  Only a `NotImplementedError` from the supplied recursion callback is wrapped.
- Output binding and phase application remain outside this catch. Parent
  assignments are not committed when a recursive evaluator fails. Existing
  assignments, last operation and phase remain inspectable.
- The native missing-callback diagnostic stays explicit. Foreign
  `DecomposeTypeError` and `DecomposeNotImplementedError` translation remains an
  open interoperability boundary; those classes are not added to native code.
- Source pure recursion creates its own default random handler, whereas phased
  recursion inherits the parent's handler. Native callback owners still choose
  their nested policy explicitly. This batch makes no automatic-recursion or
  random-policy parity claim.

## Evidence

`tests/unit/test_component_classical_recursion.py` compares real pinned source
states against native states in pure and phased modes. Twenty-four cases cover
value, phase and recursion errors across four exception categories, including
exact cause identity and callback stopping order. Twelve nested graph cases
cover depth 2/3/5 and both input bits: the complete type/message cause chain
matches the source, and each failed parent retains only its original input.
The source classes are test-local Bloq implementations using the actual source
dispatch; the frozen reference files are unchanged.

Numerical exception context emits **zero primitives** and owns **zero quantum
work bits**. It has no successful measurement branch or new circuit decomposition
to count. Physical hierarchical add/adjoint and measurement consumers remain in
the concentrated regression; their measured primitives and source QEC estimates
remain separate in their existing evidence. Failure-chain depth is a diagnostic
count, never a gate or resource-model estimate.

Q-0882/Q-0883 remain open for source-wide dtype admission, foreign decomposition
errors and complete original-test replacement. This extends existing child
evidence without changing the fixed parent or child denominator.

## Local verification

- 36 new source comparisons; 2,934 concentrated regressions and 84 CPU/tensor-network checks passed.
- Strict typing for both changed Python files, Black, Ruff, team scope, architecture, multi-level IR and protected API checks passed.
- The state owner measured 97% branch coverage (127 statements, 52 branches; 3 uncovered refusal sites). Six reference Git blobs match the frozen inventory.
- Default smoke/unit run: `5 failed, 21783 passed, 300 skipped, 1037 warnings in 117.52s (0:01:57)`. Observed failures are members of the seven known baseline IDs; 2 baseline IDs were not observed before the configured max-failure stop. This is not a green full-suite result.
- Evidence: `ftqc-reference-env/g2-component-classical-recursion-{source,default-comparison}.json` and matching verification logs.
