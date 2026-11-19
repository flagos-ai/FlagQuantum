# Source workflow and acceptance

The pinned source is `algorithms/search_and_optimization/grover/grover.ipynb`
at commit `b69c4893d8264d1e59ce0909df2bd88efed25e00`. Its eighteen code cells
are reviewed and hash-bound in `provenance.json`; source Qmod is not bundled.

| Original cells | Native behavior |
| --- | --- |
| 5 | Bounded adaptive classical loop, 1000 observations per attempt, strict individual-row threshold, duplicate repetition skip, final-attempt failure return. |
| 8, 11, 13 | Three-variable six-clause SAT truth table in logical variable order. |
| 15, 16, 18, 20 | Native coherent predicate, uniform reflection, local circuit/resource inspection, small SAT search and top three observed rows. |
| 23, 25, 27 | Four-variable thirteen-clause SAT, truth table, local circuit/resource inspection, search and top five observed rows. |
| 33, 35, 36 | Original five-vertex six-edge graph, cut-at-least-four predicate and native coherent search. |
| 37, 39, 41, 42 | Threshold 0.1, up to 32 observed rows, most observed partition, planar node colors and crossing-edge widths. |

The source loop computes `ceil(pi/4 * sqrt(2**i))` for `i=0..3`, hence powers
1, 2, 2, 3. Duplicate power two consumes iteration two without a quantum run.
A single marked row must strictly exceed 0.45 for SAT or 0.1 for MaxCut.
The total marked probability is not the stopping criterion. Failure returns the
last actual attempt with `succeeded=False`. Source `n_solutions=3/5` are display
limits: both SAT formulas actually have two solutions. The graph has ten marked
assignments at the retained inclusive cut boundary four.

Static repetition binding and explicit Torch sampling replace remote late
binding and remote sampling. Native base seeds 1100/1200/1300 plus loop index
make observations reproducible; the unseeded remote sample trajectory is not
claimed. Equal counts are ordered by packed integer. Integer bit i is logical
q[i]; tuples present q[0] first. Named register probabilities remove manual
wire decoding from the user path. Local text circuit drawings and resource
records replace the hosted viewer. Node colors use the most observed final
word, whether or not that word triggered the stop. Edge widths are 1.5 for
crossings and 0.5 otherwise.

Independent Circuit references construct a diagonal classical-predicate oracle
and the exact reflection `2|s><s|-I`, then compare full absolute complex states
for all three widths and powers zero through three. Check every adaptive
sample with the same explicit seed, exhausted loops and duplicate iterations,
strict equality thresholds, truth-table boundaries, logical bit order, zero
clean workspace and actual resource-budget refusal. The fresh Notebook must
run all three complete sections and display an actual inspected graph PNG.
All 216 frozen quantum counterparts and six approved revisions, default tier,
quality checks and CLI complete local acceptance. Classification:
`single_device_fast_path`; no hardware, distributed, release or quantum
advantage claim is made.

## Recorded local acceptance

On November 13, 2026, all 51 focused cases and 49 catalog/dependency cases
passed. Strict typing for Python and flattened Notebook code, Black, Ruff,
architecture, documentation links, repository language and compiler-team scope
checks passed. A fresh Jupyter kernel executed all six native code cells and
one independent evidence cell; all nineteen evidence scenarios passed. The
actual inline partition PNG was inspected and the CLI completed.

The three original searches executed four quantum programs and drew 4,000
actual observations. All stopped under their strict individual-row thresholds.
The additional threshold-one demonstration executed three programs and drew
3,000 observations; it correctly reported failure and loop indices 0, 1, 3.

| Problem | Repetitions | Physical qubits | Operations | Depth | Clean workspace |
| --- | --- | --- | --- | --- | --- |
| small_3sat | 1 | 3 | 22 | 16 | 0 |
| large_3sat | 1 | 4 | 88 | 75 | 0 |
| large_3sat | 2 | 4 | 172 | 149 | 0 |
| max_cut | 1 | 5 | 531 | 465 | 0 |

All twelve full absolute complex states (three widths, powers zero through
three), all source truth-table entries and every adaptive finite-shot histogram
matched independent Circuit references. Both source display limits and all
observed graph assignments are retained. Tests protect strict threshold equality,
duplicate-loop consumption, exhausted loops, tie ordering, logical bit order,
invalid bindings and actual qubit-budget refusal.

The complete 216-original plus six-approved replay was unchanged: 210 original
execution passes, six semantic refusals and six approved revision passes. The
complete default tier recorded 19,618 passed, 768 skipped, 25 failed and six
errors. Thirty unresolved nodes match the preceding baseline. Its one added
node was a watchdog child-cleanup test denied process-group signaling and `ps`
by the sandbox. Without modifying either watchdog source or test, the entire
watchdog test file passed 22/22 in an authorized environment. The raw full-run
failure is retained separately from this resolution; the full default tier is
not reported as green.

All 4,109 Python inputs stayed frozen through the terminal default run. Only
catalog count and validated-entry assertions change afterward and are retested
before sealing. This records local acceptance, not remote CI, hardware or
release certification.
