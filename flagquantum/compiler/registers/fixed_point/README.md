# Fixed-point arithmetic

This package owns polynomial approximations, reciprocal and inverse-square-root steps.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [fixed_point_polynomial.py](fixed_point_polynomial.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_fixed_point_polynomial.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`fixed_point_square_component.py` composes the retained D24/D28 terms through
shared weighted columns. Its signature preserves both pure-fraction THRU inputs
a and b, creates a RIGHT result, and consumes the matching LEFT image on erase.
The existing paper construction reads a and preserves b. The typed result owner
keeps the fixed-point descriptor while measured And calls restore scratch.
Run `tests/unit/test_fixed_point_square_component.py` for the stored/expanded
call, complex-state, measurement-history and ordinary append workflow.

`fixed_point_scaling_component.py` and `fixed_point_product_component.py` compose
D4/D5 through the shared `truncated_product_component.py` owner. They preserve
ScaleIntByReal's pure-fraction real_in, unsigned int_in and mixed fixed-point
result, and MultiplyTwoReals' three pure-fraction formats. Results use RIGHT
creation and matching LEFT image consumption. Initial And copies occupy the
result itself. Each stored CAdd owns its temporary word and carry lifetimes,
using the accepted exact Add inverse during erasure. Physical binding returns
the result mapping, which can change after call expansion. Run
`tests/unit/test_truncated_product_components.py` for direct source-child
comparison, full complex states and ordinary circuit.append(op) round trips.
Scaling supports every pair of positive concrete input widths. When int_in is
wider, the result retains its negative fractional metadata and the construction
omits multiplier terms that shift entirely past the retained word. The unchanged
flat emitter still covers its original narrower width domain. Source model
extrapolation and native retained-term counts are reported separately.

`reciprocal_seed_component.py` owns a RIGHT pure-fraction seed and its LEFT
inverse, calling the existing stored multi-control cleanup.
`reciprocal_newton_component.py` composes owned integer square, live product
subtraction, a retained XOR slice and exact restoration. Its source and estimate
are THRU words. `reciprocal_component.py` preserves InvertRealNumber's original
mixed fixed-point THRU a/result formats. It owns and erases every history word
and XORs the lower approximation into the result's fractional suffix. The
occupied integer prefix survives. Nonpositive fractional widths have a zero
quantized mask. The unchanged source supplies a model and lacks root
decomposition. Native numerical construction and source model remain separate.
Run `tests/unit/test_reciprocal_components.py` for stored/expanded calls,
complex occupied results, channel certificates and ordinary append replacement.

`arcsine_declaration.py` owns the frozen ArcSin constructor, unsigned fixed-point
ports and degree/iteration model. Its source has no physical decomposition.
Simulation's scalar callback and the independent Appendix D approximation keep
their separate word contracts. The trigonometric catalogue consumes this
metadata owner. Run `tests/unit/test_trigonometric_declarations.py` and
`tests/unit/test_trigonometric_foreign_file.py` for symbolic/concrete declarations
and actual foreign libraries. See `TRIGONOMETRIC_FILE_AUDIT.md` for source overflow,
missing coefficients and the independent executable approximation.
