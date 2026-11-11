# Opaque values in classical component graphs

`ComponentClassicalState` now dispatches Core `opaque` ports to the shared
`bind_opaque_register_values` owner in `classical_registers.py`. It previously
sent them through unsigned integer validation, incorrectly rejecting values that
Qualtran v0.7.0 `QAny` explicitly leaves unrestricted. For example, a scalar
`0.5`, `-1` or an arbitrary object now survives identity evaluation unchanged.

The locked source is
[`data_types.py::QAny`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py)
and
[`classical_sim.py`](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py).
Both QAny scalar and array validators are no-ops. This is source behavior, not
an inferred integer range. No frozen reference file is edited.

## Ownership and execution

Simulation retains numerical binding, distributions and storage. The new binder
reuses existing lookup, concrete shape validation, symbolic bypass, row-major
assignment and scalar distribution resolution. It does not duplicate traversal,
sample twice, add a type hierarchy or import Compiler. Graph collection still
uses the shared source storage policy: uint8/16/32/64 through width 64 and object
storage above that width. Hence unrestricted array binding can subsequently
truncate a fractional value, warn, or fail during NumPy collection. Tests compare
that conversion behavior instead of normalizing or repairing it.

This is a `single_device_fast_path` numerical inspection fix. Core admission,
physical word encoding and protected public APIs do not expand. In particular,
an opaque scalar object accepted by the classical interpreter is not implicitly
an encodable quantum basis state. Users still use `circuit.append(op)` for
physical execution; the identity consumer verifies that workflow for valid basis
values and full complex superpositions.

## Source comparisons

The new 121 tests cover:

- 70 scalar cases at widths 0/1/5/64/65, preserving exact object identity for
  integers, floats, NaN, complex values, strings, bytes, None, objects,
  containers, NumPy scalars and symbols.
- 36 vector/matrix cases at widths 1/9/65, comparing storage dtype, every
  collected value, exception categories and ordered warning categories. Object
  arrays containing NaN are compared elementwise without changing the data.
- Eight fixed-distribution cases and three seeded RNG cases (16 draws each).
  Output fixed values preserve source object identity; source input fixed
  distributions fail with `AttributeError` because LeftDangle has no `i`.
  Native input ID -1 remains an explicit representation difference. RNG input
  distributions agree with source values and complete generator state.
- Shared missing-key, shape, no-handler and missing-fixed-ID refusal checks.
- Three physical identity consumers at widths 1/2/3: full classical matrix
  reconstructed from all basis inputs agrees with the real source identity
  decomposition tensor. Ordinary append preserves full complex amplitudes and
  an unused clean spectator remains zero.

The source identity Bloq and its empty decomposition are test fixtures using
the actual pinned simulator/tensor dispatch, not edited upstream implementations.
The scalar unrestricted-value tests also compare source's direct numerical
callback path; physical tensor comparison uses the fixture's declared identity
decomposition.

## Separate resource accounts and open boundaries

| Consumer | Measured primitive instructions | Model estimate | Work and measurement |
| --- | --- | --- | --- |
| Numerical opaque binding | 0 | Not applicable | 0 quantum work bits |
| Physical width 1/2/3 identity | 0, verified from the bound stream | Empty identity decomposition | 1 unused clean spectator; no measurements |

There are no measurement branches in this identity circuit. Existing measured
component consumers remain in concentrated regression. No numerical acceptance
case is presented as a physical gate synthesis result.

Q-0882 and Q-0883 remain open for broader dtype admission, foreign decomposition
error translation and complete original-test replacement. Source fixed-input
and empty-array differences remain explicit. The batch extends existing child
evidence and does not increase source-file acceptance or its denominator.

## Local verification

- 121 new tests, 3,055 concentrated regressions and 84 CPU/tensor-network checks passed. Existing concentrated warnings: 207; integration warnings: 2; the new tests have no unhandled warnings.
- Strict typing of all three changed Python files, Black, Ruff, team scope, architecture, internal multi-level IR and protected API checks passed.
- Coverage: state 97% (129 statements, 54 branches), shared binder 95% (57 statements, 22 branches), combined 96%. Six reference blobs match the frozen inventory.
- Default smoke/unit: `7 failed, 22011 passed, 300 skipped, 1037 warnings in 120.13s (0:02:00)`. All observed failures match the seven recorded baseline IDs; 0 baseline IDs were not observed before the max-failure stop. The full suite is not green.
- Evidence: `ftqc-reference-env/g2-component-opaque-values-{source,default-comparison}.json` and corresponding verification logs.
