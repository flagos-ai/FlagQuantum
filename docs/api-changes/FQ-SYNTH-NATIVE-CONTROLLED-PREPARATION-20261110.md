# Native controlled preparation and borrowed reversible composition

Status: observed refusal; implementation and independent acceptance pending.

## Requirement and shared boundary

The frozen complex-network quantum walk uses four conditional probability-state
preparations, conditional Grover reflections, register swaps and three rounds.
A native probe fails at controlled inplace preparation inside a borrowed helper.
The source adapter already represents the required pure effects. Native capture
must compose the same checked effects rather than importing the source adapter,
embedding source text or manufacturing a dense matrix for the complete program.

Reuse native probability validation and inplace preparation, mutable helper
binding, within/apply and inverse rules, predicate compute/uncompute and the
Compiler's existing controlled-effect expansion used by QPE. Preserve their
ownership and precision contracts. Admit direct controlled inplace preparation
and bounded static mutable helper callbacks only when capture proves pure
reversible effects on existing storage. The callback may have static classical
parameters and nested reversible regions; its actual storage must be disjoint
from every condition input. No allocation, measurement, reset, drop, Output
publication, escaping local storage, recursive expansion or host execution is
admitted. Existing AST, expansion, width and resource limits remain authoritative.

Conditional scalar global phases must become actual phases of the control
branch. Indexed negative controls must retain their exact polarity. Existing
inverse preparation conventions and logical bit order must remain unchanged.
Extend shared controlled-effect expansion only for explicitly checked effect
forms needed by these compositions; unsupported effects must refuse clearly.
Closed scratch may be reused only with a verified compute/action/uncompute
lifetime; it must never be controlled by naively repeating its allocation.

Python's `QNum.__eq__` annotation must remain compatible with `object.__eq__`
and therefore has Boolean type. Extend the `control` annotation to
`QNum | QArray[QBit] | bool` so ordinary captured equality predicates type-check.
This describes AST comparisons, not execution of host predicates or acceptance
of literal `True`/`False`. Literal host conditions continue to refuse. The live
array-control admission advances to1.1 with this authorization; retain the
earlier array-control proposal and all SHA-bound evidence unchanged. There are
no new exports or changes to parameter order, defaults or host-call behavior.

## Acceptance and delivery split

1. Compiler capture and shared expansion, with independent conditional-unitary
   and global-phase interference tests, nonzero complex inputs, inverse regions,
   negative selection, alias/Const/Output escape refusals and all budgets.
2. Native complex-network walk with original probability vectors and all three
   rounds. Independent Circuit builds conditional preparation and diffusion from
   the documented state-preparation unitary convention; compare all complex
   amplitudes including the dropped environment and all work. Do not equate a
   reduced output distribution with full-program acceptance. Verify rounds0..3
   and at least two nonzero phase perturbations independently.
3. Re-run existing QPE and controlled callback regressions, all216 originals and
   approved6 after final production changes; preserve baseline resources and
   record any explicit differences. Only then update catalog, provenance,
   five-category inventory, immutable evidence and delivery ledger.

Starting accepted coverage is192/216 quantum programs,24/213 Notebook workflows
and14/19 revision-nine merged scopes. No push, integration PR or full remote CI
before all216 native quantum programs pass independent acceptance. This contract
does not claim measured teleportation, flexible10bit QPE, general signed numeric
callbacks or complete Notebook workflows.
