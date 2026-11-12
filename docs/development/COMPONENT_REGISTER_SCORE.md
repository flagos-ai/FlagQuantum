# Core component register scores

`drawer/component_score.py` projects an existing typed `ProgramRecord` into
`RegisterScore`; `register_score_data.py` supplies glyph records to the existing
JSON, offline HTML and Matplotlib renderers. Compiler owns validation and graph
selection. Drawer owns presentation only: no decomposition, tensor supply,
execution, backend selection or protected Core schema changes occur here.

## Shortest workflow

```python
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.register_score_mpl import draw_register_score

record = addition_component_program(5)
score = component_register_score(record)
data = register_score_data(score)
figure, _ = draw_register_score(data)
figure.savefig("addition.svg")
```

Default symbols use native display names. An optional `style(function, step)`
callback can set the step's gate and control values, while preserving its exact
input/output identities. An explicit `wire_symbol(step, register, side)` callback
to `register_score_data` can supply component-specific glyphs. Its connector call
has `register=None, side=None` and must return `Text` with string `text` and
integer `fontsize`. Callback exceptions propagate. These presentation callbacks
do not infer numerical providers from function names or change execution.

## Ordering and ownership

The selected entry is a finite, one-block typed quantum-call graph admitted by
Compiler's existing component traversal validation. Positive concrete word
widths are required; classical, symbolic and zero-width words, conditioned calls
and parameterized calls are refused. Empty array dimensions retain their existing
zero-port semantics.

Stored block declaration order seeds layout; the existing register-score owner
computes display dependency generations and reuses free lanes. Physical component
scheduling may reorder independent calls and is intentionally not reused as a
display ordering authority. That reordering can change horizontal endpoints even
when the physical computation is equivalent.

THRU outputs keep their input's display lifetime. RIGHT outputs create fresh
collision-free lifetimes before LEFT inputs release their lanes. Shaped registers
retain row-major element order and word widths. No SSA identity or register
metadata is modified. Final returns must cover every live display lifetime once.

Single-bit Add follows the pinned direct CNOT branch with no split/join wrapper.
Larger Add uses the existing factorized graph and shared physical sequence;
drawing does not alter its measured erasure or exact adjoint.

## Locked-source evidence and limits

`tests/unit/test_drawer_component_score.py` compares unchanged Qualtran v0.7.0
at `bcb17bb25360ed68ce3bc084645db3e929799389`: atomic X/CNOT/Toffoli, all
four original closed notebook examples, and expanded Add at widths 1, 2, 3, 5,
and 32. Explicit presentation callbacks reproduce source component glyphs.
Sequential/dependency coordinates, horizontal/vertical lines and glyph attributes
match. Native identity strings and open-boundary register/dtype text differ and
are normalized explicitly in those comparisons; closed examples need no boundary
text normalization. This is not a claim of byte-identical JSON or general
source-wide drawing coverage.

| Expanded graph | Columns / vertical lines | Horizontal lifetimes | Port records | Maximum lane |
| --- | ---: | ---: | ---: | ---: |
| Add(5) | 33 | 18 | 94 | 13 |
| Add(32) | 249 | 99 | 688 | 94 |

These are presentation counts, not primitive gates or cost-model estimates. The
unchanged physical/resource accounts remain in [ADDITION_COMPONENTS.md](ADDITION_COMPONENTS.md).
Automatic production tensor-provider composition and built-in contraction planner
speed parity remain open; successful drawing does not certify either.

Run the focused test above in the locked reference environment, then the existing
arithmetic, component and drawer regressions. Evidence prefix:
`g2-component-score-`, including JSON/HTML and an inspected Add(5) SVG/PNG.

Local validation: 23 new cases, 8430 concentrated regression cases, 1479
coverage-run cases and 84 CPU/TN integration cases pass. One existing register
Graphviz rendering test is skipped because Graphviz is unavailable; the new
JSON/HTML/Matplotlib SVG path passes. Strict typing of 13 files, Black, Ruff,
Compiler scope, architecture, IR and protected API checks pass. The projection
has 95% statement/branch coverage (63 statements, 24 branches); conditioned-call
and incomplete-live-return refusals are not exercised in this coverage run. The
Add graph has 100% coverage (53 statements, 12 branches). Five source blobs
match the fixed inventory. No default full-suite rerun was required for this
slice; its previous seven recorded baseline failures remain unresolved.
