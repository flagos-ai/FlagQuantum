# Complete fixed-point error-analysis notebook

Q-0197 covers all eight original cells of
`qualtran/bloqs/arithmetic/error_analysis_for_fxp_arithmetic.ipynb` at Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Git blob
`57a9927840390329d583a7203c941ece4386ef47` is restored and checked before
execution. Reference execution uses the existing isolated fxpmath 0.4.10.
The original source and dependency are unchanged.

## Responsibilities and ownership

The native teaching helper is `examples/algorithms/fixed_point_error_analysis.py`.
It implements the notebook's numerical experiments with explicit stored integers
and reuses Simulation's existing `fixed_point_real_value` decoder. This is a
finite scalar analysis asset, with no third-party numerical objects entering
Core or Compiler and no production dependency addition. The native notebook
runs all four executable cells while blocking fxpmath, Qualtran and Cirq imports.

| Source responsibility | Native disposition |
| --- | --- |
| Introduction and paper sections D4/D5/D6 | Preserve all three goals, equations and source section ordering. |
| `fxp` helper | Preserve `0 <= x < 1`, unsigned stored words, fixed-width truncation and saturation. Native return is the raw integer. Word width and fractional width remain explicit. The external Fxp object API is confined to the reference boundary. |
| `assert_allclose` | Compare decoded scalar values with NumPy's original absolute epsilon and default relative tolerance. Do not silently tighten or remove the relative term. |
| D7 integer-product precision helper | Preserve `d_B + ceil(log2(d_B/eps))`. |
| Real times integer | Original wide input, bit-index loop, shift and conversion to the result's `d_A-d_B` fraction before each addition. |
| D17 real-product precision helper | Preserve the original parentheses, ceil location and final integer truncation. |
| Real times real | MSB-first multiplier traversal excludes the final bit. The left operand shifts before each conditional addition. |
| D36 square precision helper | Preserve the original bound, including its failing experiments. |
| Square algorithm | Preserve both loops, evolving bit mask, same-width literal multiplication and even-width tail. |
| Three seeded experiment helpers | Preserve RNG seeds, 100 samples per configuration, NumPy tolerance, first-assertion stop and Success/Failed report. |

## Exact numerical checks and retained failure

The complete original and native notebooks print exactly the same 28 report
lines. All 2800 seeded scalar results are also compared directly, including
later square samples that the original report helper stops before evaluating.
All dyadic inputs of widths one through six additionally compare the odd/even
square paths. Constructor tests cover ordinary, wider and narrower storage,
values next to one and both original input-domain rejections.

A draft using exact integer multiplication by literal one initially reported
success where the source reported failure. The source `const_op_sizing='same'`
first converts one to the pure-fraction format, saturating it to `1-2**(-d)`.
Each multiplication then truncates at the same fractional width. The native
helper now preserves that operation at every original multiplication site.
For example, input `0.4106776946403047` at 14 bits produces stored word 2754,
exactly matching the untouched reference, rather than the exact-literal draft's
2762. No bound, reference function or assertion was weakened.

The original D36 experiment reports Failed at `(d, eps)` equal to
`(14, 0.001)`, `(18, 0.0001)` and `(21, 0.00001)`. It reports Success at
`(25, 0.000001)` and `(29, 0.0000001)`. These remain visible in the native
notebook. The source comment speculates about the paper's bound. This audit
records the actual constant-saturation behavior and does not infer that the
paper's quantum square circuit has the same numerical defect.

## Physical and model boundaries

The notebook emits zero quantum primitives, owns no quantum workspace and has
no measurement branches. Full complex amplitudes and quantum work reset are
not applicable to this scalar asset. Existing executable fixed-point product
and square suites retain their independent phase, branch, workspace and
resource evidence. These fxpmath experiment outputs do not certify their gate
algorithms or create a new resource estimate. The normal quantum entry remains
those existing operations through `circuit.append(op)`.

The native helper covers the notebook's unsigned finite scalar experiments and
positive Python integer widths. It is not a substitute for fxpmath's public
object, configuration, vector arithmetic or arbitrary dtypes. Those external
library interfaces are outside this source notebook's transplant scope. Q-0197
closes the complete teaching/analysis responsibility. Existing quantum source
and admission gaps remain under their original implementation parents.

The concentrated batch passed 466 native/reference cases without skips or
warnings. Native helper/test and extracted notebook code passed strict mypy,
Black and Ruff. Compiler scope, architecture, multilevel IR and protected API
checks passed. The helper explicitly rejects nonpositive storage and negative
fraction widths. Notebook variable names follow the repository's lowercase
policy, while printed source labels remain identical. The sampled integer is
converted to a Python int without changing its value or RNG consumption.
Logs are `ftqc-reference-env/g2-fxp-analysis-*.log`.
