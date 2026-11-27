# Experimental implementation ownership

The experimental namespace owns capability entry points and its artifact codec.
Three remaining function definitions move to existing domain modules:

- Controlled-Pauli operation-name decoding: Core Operators Pauli.
- Controlled-Pauli dense matrix construction: Simulation Pauli numerics.
- Fixed-shot matching with a caller-supplied graph: QEC Execution Experiments.

These are exact function extractions. Validation, matrix conventions, shot and
seed checks, decoder selection, sampling, confidence intervals and results stay
unchanged. No new product package, abstraction, workflow or stable export is
introduced. Drawer consumes the Core name decoder directly. The two existing
experimental modules expose their shipped functions without owning behavior or
retaining private helper globals. The root nine-namespace discovery contract and
all frozen API/contracts remain unchanged.

The controlled-Pauli plot-gallery example and documented direct QEC workflow
explain why those two existing entry modules remain. They are still explicitly
unstable with no compatibility guarantee, not two new stable promises. Their
exact exports and owners are itemized in the public-entry inventory (102 entries).

The five source/target modules are unchanged in PR1084's own base-to-head diff
through ce4e769d0ff9b188e9c16972719ba943a52f4bcd. Its verified changed-text set
has no direct reference to these two experimental modules. This implementation does not integrate the draft PR. Complete formal source
regressions and clean installed-artifact verification pass, as recorded in the
[acceptance](shared-implementation-ownership-acceptance.md). The pushed commit
still requires its own exact-head CI qualification.

The original controlled-Pauli helper module did not postpone annotations. Its
two relocated functions retain their original eager annotation values so raw
callable signatures as well as resolved hints remain unchanged. This preserves
introspection during a structural move without changing experimental stability.

The later complex-phase increment through
`177ed24aba6c27d72b6fcb4b79b60747d8feec9e` has six SHA-verified files
and no new candidate reference or source/target overlap. The actual import guard
checks 3,202 incoming Python trees for removed experimental implementation
globals while allowing the preserved public entry points.

Activation audit: the next nine-file mean-estimation/complex-phase increment
through `0742c440a100ce56eaec9934f9cf1487a84e0bc5` is SHA-verified.
It has no source/target intersection or new candidate reference; the actual
retired-import guard passes on 3,204 incoming Python trees.
