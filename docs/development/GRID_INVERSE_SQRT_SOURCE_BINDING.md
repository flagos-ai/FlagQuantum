# Grid inverse-square-root source binding

Frozen source: Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`bloqs/chemistry/trotter/grid_ham/inverse_sqrt.py`.

## Interface and ownership before implementation

Both original roots are resource-only Bloqs. Polynomial evaluation has ordered
THRU opaque ports x_sq, in_coeff shaped (4,), and out; Newton has x_sq, poly,
and target. Each scalar word retains its supplied bit width; shaped words use
row-major elements and the existing MSB-first binder. No source work ownership,
measurement, ordered primitive body, numerical fixed-point contract, or zero
output promise is supplied. A bodyless Program must remain non-executable.

Compiler owns their shared typed signature and exact constructor-wire binding.
Algorithms retains the existing concrete model entry points and independently
specified polynomial/Newton execution. It delegates signature construction to
the shared owner without changing its positive-width admission. Formats must
not import Algorithms, numerical execution or an optional SDK. Symbolic and
empty source declarations reuse existing logical opaque storage contracts.
Explicit native execution remains a separately selected operation appended by
users; constructor binding must not silently select an approximation.

The whole-file audit must disposition two coefficient helpers, both classes and
their signature/model/drawing methods, two factories and both DocSpecs. Existing
coefficient table, resource, drawings and coherent execution evidence is reused.
Generic user-supplied coefficient behavior and large-chain measurement histories
remain distinct obligations until their actual evidence has been inspected.


## Complete source disposition

| Original definitions and assets | Owned implementation and evidence |
|---|---|
| get_inverse_square_root_poly_coeffs | Existing Algorithms decimal literals and binary64 pairs match both complete source arrays exactly; no second constants owner is introduced. |
| build_qrom_data_for_poly_fit | Ecosystem inverse_sqrt_coefficient_qrom_data retains supplied coefficient banks, original zip truncation, four-row output, repeated addresses, binary64 scaling, optional fixed-point conversion, wrapping and platform-integer failures. Plain tuple output stops external arrays at the boundary. |
| NewtonRaphsonApproxInverseSquareRoot and signature | Shared Compiler opaque signature; actual original constructor libraries at depths 0/1/2 bind without an invented numerical format or primitive body. |
| NewtonRaphsonApproxInverseSquareRoot.wire_symbol | Existing drawer custom root and all ordinary register symbols match the unchanged original. |
| NewtonRaphsonApproxInverseSquareRoot.build_call_graph | Existing concrete Algorithms cost model and new generic resource formula retain square/scale/two-product/add dependencies and concrete or symbolic QEC categories/width fallback. |
| PolynmomialEvaluationInverseSquareRoot and signature | Shared Compiler x_sq/in_coeff[4]/out signature and actual source constructor libraries retain THRU ownership, including symbolic and empty logical words. |
| PolynmomialEvaluationInverseSquareRoot.wire_symbol | Existing drawer custom root and all four indexed coefficient boxes match the unchanged original. |
| PolynmomialEvaluationInverseSquareRoot.build_call_graph | Concrete model and generic formula preserve three same-width real products and signed adds without claiming an executable source order. |
| _nr_inv_sqrt and _poly_inv_sqrt | Both actual factories resolve their original 7/8/12 declarations and complete source-wire signatures. |
| _NR_INV_SQRT and _POLY_INV_SQRT | Actual DocSpecs retain the original classes and example objects; frozen complete source/test blobs are checked. |

The two classes and their six methods, two helpers and two factories account
for all twelve original definitions. No original file is modified. The real
Kickmix 0.1 Simulator and Qualtran 0.7.0 are asserted before reference evidence.
The prior unchanged original suite (12 passes, six original skips), coefficient,
drawing, native polynomial/Newton and SDK evidence are reused where unchanged.

## Exactness and failure boundaries

Source-compatible coefficient conversion covers original and supplied negative,
wrapped, short, empty and excessive coefficient sequences. Nine address/target
profiles include zero-width, small address, 63/64/128-bit conversion cases.
Original successes and exception types match; recursive Newton serialization
with polynomial width below three raises the original SquareRealNumber error.
Shallow root declarations still exist. Numerical quantum execution is refused
for every bodyless root. Source resource counts cannot invent a primitive body.

Unconstrained symbolic source width accounting retains both input and output
widths of square, scale and product children inside Max. Dropping an input
width because a concrete positive width would dominate changes the symbolic
expression and is not permitted. Generic formula comparisons verify all gate
categories and peak width against original QECGatesCost/QubitCount.

The independently specified cubic/Horner/Newton implementation now has complete
branch certificates for the full 4-distance-bit/5-fractional-bit chain and the
3-distance-bit/16-fractional-bit chain. Every actual temporary-And compute/erase
block and its conjunction invariant is checked. Both local erase outcomes have
the same positive branch scalar, so composition proves the same complex linear
operator in every combined raw history. This is not enumeration of 2^1026 or
2^4670 trajectories. All distance and occupied target words are checked in the
first profile. In the wide profile all distance words, zero/basis-vector/all-one
targets and a structural proof that targets only receive CX extend the result to
every occupied target. A coherent spectator is retained; all private work is zero.
Existing sparse complex trajectories remain independent checks of execution.
The separate small polynomial/Newton all-branch certificates are reused.

## Measured inventories and original estimates

| Path | Measured primitives and storage | Independent source estimate |
|---|---|---|
| Supplied four-bank QROM, select 2 / target 1 each | 6 X, 6 H, 4 T, 4 Tdg, 20 CX, 2 S, 2 measure, 2 CZ, 2 reset; one reserved clean wire | Compare literal original QROM operands, order and feedback; no original inverse-root gate recipe |
| Independent complete cubic/Newton, distance 4 / fraction 5 | 44 X, 3078 H, 2052 T, 2052 Tdg, 10284 CX, 1026 S, 1026 measure, 1026 CZ, 1026 reset, 6 CCX; 208 clean, 13 slots | Not the source model's primitive decomposition |
| Independent complete cubic/Newton, distance 3 / fraction 16 | 28 X, 14010 H, 9340 T, 9340 Tdg, 49367 CX, 4670 S, 4670 measure, 4670 CZ, 4670 reset, 2 CCX; 1322 clean, 34 slots | Not the source model's primitive decomposition |
| Original polynomial model 7/8/12 | No measured primitive body | 165 Toffoli, 21 And, 138 Clifford, 21 measurements, peak 51 |
| Original Newton model 7/8/12 | No measured primitive body | 345 Toffoli, 11 And, 74 Clifford, 11 measurements, peak 36 |

Receipts use g2-grid-inverse-source-*-20261126. This closes the actual source
capability mapping and the previously outstanding representative complete-chain
branch proof. It does not give the original opaque ports a new arithmetic
meaning, extend bounded native convenience APIs, or claim arbitrary-size dense
simulation. Q0224's missing original minimax coefficients remain open.


Final concentrated regression: 287 passes; Cirq 1.6 lane: 167 passes; local
integration: 145 passes. Eight changed Python files pass strict typing,
formatting and lint. Three owner scopes, repository language, documentation
links, architecture, Program IR and public API checks pass. Both frozen blobs
and the real native execution extension are checked in the reference lane.
