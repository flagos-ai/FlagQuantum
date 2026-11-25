# Literal THC preparation and source comparison boundaries

The locked reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete constructor graphs
from `chemistry/thc/prepare.py` now have native typed composition owners.
This closes the logical construction work, not physical acceptance of that
whole source file. The original reference is unchanged.

## Ownership and interfaces

| Owner | Responsibility |
| --- | --- |
| `registers/preparation/thc_uniform_source_component.py` | Literal 35-call uniform graph and explicit directional dependencies. |
| `registers/preparation/thc_preparation_component.py` | Literal 17-call PREPARE root and exact reversal using supplied child pairs. |
| `registers/preparation/thc_preparation_metadata.py` | Shared selection, target, retained-bank and through-work signatures. |
| `formats/legacy_thc_preparation_factories.py` | Constructor payload validation and scoped source graph resolution. |
| `algorithms/primitives/thc_chemistry_prepare.py` | Existing mathematical native operation; delegates metadata without changing its numerical emitter. |

Words are MSB-first. The lookup produces RIGHT selected targets and retained
dataset-major banks. The uniform graph owns its four-bit private word. The root
threads caller-owned `s`, `less_than` and `extra_ctrl`; these are not silently
classified as private clean work. The source contiguous-index classical map has
no executable decomposition: its supplied reversible XOR extension remains an
explicit native dependency. No backend selection or numerical execution moves
into Compiler. Executable native operations retain `circuit.append(op)`.

## Source defects and physical acceptance still open

* `GreaterThanConstant` has no executable source decomposition. Literal graph
  resolution retains a bodyless declaration and physical binding refuses it.
* The original uniform success predicates `nu < M` and `nu == M + 1` are
  disjoint. The final X also leaves a bit dirty before a clean Free. Matching
  the original calls does not prove a valid prepared state or clean work.
* Sentinel `M + 1` exceeds the source word for M equal to 1, 3 or 7.
* PREPARE positionally interchanges `succ` and `nu_eq_mp1` after its uniform
  call. Literal composition preserves that wiring, without adding gates.
* With `log_block_size=0`, the source root asks for absent `junk_target0_`.
  Native source resolution reports that failure instead of repairing the graph.
* The source QROAM inverse is model-only. A supplied paper-based native erase
  is a separate execution path, never an upstream gate decomposition.

The mathematical THC implementation already uses different preparation logic.
Its numerical, branch and work-cleanup evidence cannot certify the literal
source profile. Whole-file physical acceptance therefore remains open until
these boundaries have an explicitly accepted disposition and the required
end-to-end complex amplitudes, applicable histories and work checks are proved.

## Preparation comparison correction

The old generic test terminal classifier could stop at composite one-sided
PREPARE roots. `tests/unit/preparation_reference.py` traverses those roots and
selection lookup wrappers; bodyless source models and explicitly named native
extensions remain terminals. Earlier independent numerical evidence remains
useful, but the old traversal alone did not prove full hierarchy parity.

That stronger comparison exposed six workspace bookkeeping calls in the native
reserved QROM profile. Source constructor bindings explicitly choose
`reserve_workspace=False`. The existing default remains True for native
physical consumers. On the diagnostic table the canonical source root has 62
calls, while the reserved native root has 68. This distinction concerns stored
call graphs; bookkeeping calls are not physical primitive gates.

## Evidence

`test_thc_uniform_source_component.py` checks four dimension pairs, both
directions and producer depths zero/one, literal calls and operands, frozen
source blobs, and explicit physical refusal. The PREPARE source test checks
the original M8/N8 eight-bit table with block log two and a smaller eight-bit
table with block log one, both directions and producer depths zero/one. It
also retains the zero-block source counterexample. The strengthened sparse
PREPARE constructor tests use the same hierarchy traversal.

External receipts under the reference environment:

* `g2-thc-uniform-literal-source-20261125.log`: eight uniform cases passed.
* `g2-source-preparation-canonical-calls-20261125.log`: thirteen combined
  THC/sparse constructor and hierarchy cases passed.
* `g2-thc-source-original-tests-20261125.log`: unchanged source suite, thirteen
  passed, three source skips, one source xfail, notebook deselected.
* `g2-thc-source-sparse-resources-20261125.json`: six directional resource rows
  for the affected executable sparse factory; actual emitted primitives,
  retained outputs and private leases are separate from original QEC/peak models.
* `g2-thc-literal-models-20261125.json`: four original eight-bit uniform/PREPARE
  directional model rows, each with a verified physical refusal and no measured
  primitive count. Original forward PREPARE estimates 216 T, 23 Toffoli,
  12 CSwap, 121 And, 2452 Clifford, two rotations and 121 measurements;
  the source peak model is 125. These are estimates, not emitted instructions.

The affected batch regression passed 393 cases; local integration passed 145.
The final thirteen-case hierarchy run also passed after adding an assertion
that the preparation root cannot be a terminal. Strict typing of fifteen
touched Python files, formatting, team scope, repository language, architecture,
IR and public API checks passed. Dependency-policy output retains exactly the
720 existing findings; that check is not globally passing.

Literal THC physical primitive counts are unavailable because binding refuses
missing bodies. Source model estimates must not be presented as measured gates.
Large M450/N152 constructor intake is separate from small numerical fixtures;
no large dense or full measurement-history result is claimed.
