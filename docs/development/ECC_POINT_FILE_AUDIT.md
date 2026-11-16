# Elliptic-curve affine point file audit

Q-0463 and Q-0464 cover `ec_point.py` and `ec_point_test.py` from
Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
Both files matched their Git blobs. The two unchanged original tests
passed. The native point is the classical operand provider for the
subsequent reversible addition and window-table modules.

The native point preserves the affine `(0, 0)` infinity encoding,
modulus and curve coefficient, negation, identity and inverse cases,
and the separate slopes for point addition and doubling. The auditable
field inversion uses the same modular inverse as the source. The
original symbolic multiplication names `ec_times_x` and `ec_times_y`
remain opaque and retain the same arguments and metadata. General
symbolic point addition still requires a concrete invertible
denominator and modulus.

Concrete multiplication uses binary double-and-add. Every point pair
and every scalar from zero through 24 matched the frozen source on the
curves `(p, a, b) = (17, 0, 7)` and `(7, 3, 4)`. These cases include
doubling at a zero ordinate, opposite points, infinity and a nonzero
curve coefficient. Tests also cover mismatched curve metadata,
positive scalar reduction by the fixture's order and the affine
coordinate display. The coordinate API follows the source assumption
that the supplied points are already on the intended curve, with the
source sentinel reserved for infinity.

For the curve over 17, the original negative scalar `-1` returns
`(15, 13)` because the repeated-addition loop is empty. The native
signed convention returns its inverse `(15, 4)` and applies inversion
before multiplying by the scalar's magnitude. The symbolic case keeps
the original opaque representation also for negative arguments.

This shared classical module emits no quantum gates, uses no
work registers or feedback, and contributes no stand-alone quantum
resource estimate. Gate and branch evidence belongs to its reversible
consumers.
