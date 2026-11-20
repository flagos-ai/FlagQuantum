# Fixed and windowed point-addition source audit

The acceptance unit is the complete Qualtran v0.7.0 `ec_add_r.py`, frozen blob
`195654d5b4b269d13d07e96a755424755b0196e5`, with its direct lookup, arithmetic
and point-value dependencies. Its fourteen class/function definitions and two
BloqDocSpec constants have the following disposition.

| Complete source responsibility | FlagQuantum owner and evidence |
| --- | --- |
| ECAddR class, fields and signature | `elliptic_curve_fixed_point.py`; scalar control, unsigned coordinate words and symbolic widths |
| ECAddR.on_classical_vals | Same owner; unchanged control-zero identity and nonzero point-addition callback, existing small-domain original comparisons |
| ECAddR.wire_symbol | `drawer/elliptic_curve_symbols.py`; scalar control circle, coordinate labels and empty root text |
| `_ec_add_r`, `_ec_add_r_small`, `_ECC_ADD_R_BLOQ_DOC` | Existing fixed-point primitive and source factory intake, with original symbolic/concrete declarations |
| ECWindowAddR class, fields and signature | `elliptic_curve_window_addition.py` and shared fixed-point signature; MSB-first row-major control array and unsigned coordinate metadata |
| ECWindowAddR.qrom | Existing concrete point table; `elliptic_curve_lookup_resources.py` adds symbolic-word lookup declarations and preserves original fixed single-block assumption |
| ECWindowAddR.build_composite_bloq | Existing lookup, six-stage addition and measurement-assisted erase owners; retained banks, explicit scratch/feedback and actual concrete source wiring |
| ECWindowAddR.build_call_graph | Existing three-child source graph; separate forward/erase resource formulas now use existing Compiler resource owners |
| ECWindowAddR.on_classical_vals | Existing independently evaluated raw Montgomery callback, original fast/slow parameter domains |
| ECWindowAddR.wire_symbol | `drawer/elliptic_curve_symbols.py`; array control glyphs, both coordinate labels and source root text |
| `_ec_window_add_r_small`, `_EC_WINDOW_ADD_BLOQ_DOC` | Existing concrete 16-bit, four-control example and table; original full-source factory/model receipt |

## Resource-query interface and ownership

`windowed_point_lookup_resources(width, window, point)` returns the two original
lookup child names mapped to existing SymbolicResourceFormula records. The
coordinate data remain owned by the existing table builder. Concrete inputs
reuse data-dependent Compiler estimates for both directions. A symbolic word
with a concrete window uses three shape-only targets of that word width, table
extent 2**window, explicit selector width window and forward block exponent zero.
This copies the original qrom property's assumption that word width exceeds the
window. Both estimates are compared independently with the actual original
qrom property and its adjoint, across every model category and logical peak.

A minimal model query is executable in the existing reference environment.

```python
from sympy import Symbol
from flagquantum.algorithms.primitives.elliptic_curve_lookup_resources import (
    windowed_point_lookup_resources,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint

n = Symbol("n", positive=True, integer=True)
models = windowed_point_lookup_resources(n, 4, ECPoint(15, 13, 17))
assert set(models) == {"QROAMClean", "QROAMCleanAdjointWrapper"}
```

These are analytical lookup estimates. ECAdd retains its independent arithmetic
resource owner and open source acceptance. Physical instruction counts remain
in the existing concrete append/branch evidence. The query builds no circuit,
allocates no workspace and supplies no physical action for unknown widths.
Existing users continue to append the bound window-addition operation to a
DynamicCircuit with explicit MSB-first ports, workspace and classical slots.

## Actual source boundaries

A symbolic window makes the original qrom property explicitly pass
log_block_sizes=None into a tuple converter. Both concrete-word/symbolic-window
and symbolic-word/symbolic-window cases fail with TypeError before a table
exists. Native lookup queries reject those declarations with an explicit
ValueError. This is an exception adaptation. Default block optimization from a
different constructor is outside this source-property evidence.

ECAddR supplies a classical callback and signature but no physical decomposition
or QEC call graph. Both original scalar factories fail the corresponding
queries with DecomposeNotImplementedError. Original width-only peaks remain
2*n+1 for the symbolic factory and 11 for the five-bit factory. Actual foreign
capture of the concrete scalar forward, adjoint and additional-control objects
fails at the existing Ecosystem decomposition boundary. The source factories
are unchanged. The windowed factory supplies a three-child model and original
peak 257, with its complete model retained in the intake receipt. That original
aggregate is separate from the independently checked lookup-only estimates.

## Remaining acceptance

Q-0458 remains open. The symbolic-word model and both wire-symbol methods now
have owned implementations. Full ECAdd dependency acceptance, its source
arithmetic cleanup defects, complete inherited external-provider scope and
symbolic physical execution retain their existing dispositions. The original
erasure leaf and scalar fixed-point leaf have no physical decomposition.
Existing concrete gate/feedback, coherent-state, all tested branch and clean-work
proofs are reused with their explicit domain promises. Model costs are never
reported as measured primitives.

The source intake receipt is `g2-ecc-add-r-intake-20261121.json` in the shared
reference environment. It lists every source definition and actual factory
outcome. Lookup comparisons are in `g2-ecc-lookup-accepted-*-20261121.log`.
Initial symbolic-window expectations are retained as diagnostic failures, and
the accepted tests assert the actual original exceptions. Diagram callback and
whole-score evidence is recorded separately under `g2-ecc-symbols-*`.

## Accepted batch checks

The complete dependency regression passes 114 cases and the local integration
lane passes 29, with zero skips. Four changed Python files pass strict typing,
Black and Ruff. Algorithms and Compiler scopes, architecture, multi-level IR
and protected API checks pass. An additional Cirq 1.6.1 lane checks the resource
and glyph scenarios. `g2-ecc-add-r-final-*-20261121.log` records the batch gates.
`g2-ecc-add-r-model-evidence-20261121.json` holds twelve independently queried
source model rows and two separately counted existing physical streams. Runtime
verification uses Qualtran 0.7.0 and Kickmix 0.1 with its native Simulator.
The physical streams reuse the existing full complex-state and cleanup tests.
They contain no changes from this model/presentation batch.
