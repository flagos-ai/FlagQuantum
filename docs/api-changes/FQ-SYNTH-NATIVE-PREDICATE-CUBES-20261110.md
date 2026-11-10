# Bounded unsigned predicate cube candidate

Status: implemented and locally verified; QLBM native/Circuit acceptance complete.

## Observed requirement

The QLBM stream/reflection program uses at most eight quantum input bits per
predicate. Its native primitive composition captures, but the representative
fragment needs sixteen work bits in the default arithmetic candidate and fails
the default 512 MiB execution memory ceiling. A workspace-eight request refuses
both arithmetic candidates. The optional Boolean candidate rejects unsigned
comparisons in this bitwise-composed predicate and currently bounds active
inputs to six bits. Retain these diagnostics; do not allocate the infeasible
state or weaken the memory constraint.

## Semantic and implementation boundary

Extend the existing optional Boolean value candidate for one-bit unsigned
predicate results containing equality/ordering comparisons. Reuse existing
verified expression definitions, normalization/facts, ANF expansion, comparison
operations, bindings, candidate selection and clean multi-controlled X lowering.
No new public API, source interpreter, unchecked predicate or opaque algorithm
matrix is introduced. Existing multi-bit arithmetic input limits remain intact.

For at most eight active input bits, a deterministic bounded construction may
cover the exact satisfying assignments with disjoint signed Boolean cubes.
Each cube is a conjunction of fixed input bits; omitted bits are free. Merge
only cubes whose union is exactly one cube. The final cover must be an exact
partition of the truth set, so XORing all cube-controlled X gates implements the
predicate for every nonzero target and coherent input without cancellation or
phase assumptions. Reuse clean ladder scratch and restore temporary negative
control flips. This is an exponential bounded candidate, not an asymptotic
quantum advantage or optimal Boolean minimizer.

Existing ANF product, source-operation, width and mapped-gate limits remain
authoritative. Explicitly bound truth enumeration and cube expansion, retain
deterministic candidate order, and refuse when the exact candidate cannot be
verified or mapped within its budget. Arithmetic alternatives remain available.
Record actual workspace and gate costs before selection. A budget or mapping
refusal must not masquerade as proof that no possible circuit exists.

## Acceptance

- Exhaust every active input and both target values for representative
  comparisons, conjunctions, disjunctions, negative literals and multi-cube
  predicates, including ranges and boundary values.
- Independently verify exact/disjoint cover semantics, determinism, resource
  accounting, clean workspace and absolute complex coherent amplitudes.
- Retain the six-input limit for existing multi-bit arithmetic examples and
  all original AO resource baselines. Wider/oversized/signed/unknown predicates
  and insufficient budgets must refuse predictably.
- Re-run the QLBM fragment under the original memory ceiling and bounded
  workspace; acceptance requires an actual selected candidate and execution,
  not merely AST admission.
- Accept the complete twelve-data-bit, thirteen-step normalized approved QLBM
  program only after independent Circuit full-state checks, two additional
  nonzero models, bit-order/boundary/phase checks and every work amplitude.
  Retain unchanged source bytes and the original normalization refusal.
- Run all216 original files and approved revisions after final implementation.
  Preserve previous raw resource evidence; any changed measured resources need
  explicit per-case records rather than overwriting the original baseline.

## Validation corrections

The original QLBM main contains thirteen stream/reflection pairs, not ten.
An independent source comparison exposed the shortened initial prototype;
the native schedule now preserves all thirteen pairs. Source struct fields are
allocated in a different physical order, so acceptance aligns logical amplitudes
using each program's declared register mappings. Original bytes and prior raw
resource baselines remain unchanged. The final suite also checks the schedule
against the frozen source, preventing a shared native/reference constant from
hiding omitted steps.

## Measured acceptance

The final QLBM suite passes twelve tests, including approved-source/native/
independent full states, two signed coherent boundary models, all clean work,
named marginals, normalization, exact source schedule and resource refusals.
Predicate cube tests pass 263 cases; Compiler passes 1,594; related regressions
pass 309; comparison resource regressions pass 198. Fresh original216 replay
retains210 execution outcomes and six semantic refusals; all six approved
revisions execute. Two comparator resource changes are recorded separately from
unchanged raw baselines. Native QLBM uses19 physical/7work bits,3342operations
and depth2770 for all13steps, versus approved source15physical15704operations/
depth9109. This is local correctness evidence, not hardware parity or complete
Notebook certification.
