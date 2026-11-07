# Classical modular multiplication values

`simulation/classical_modular_multiplication.py` ports ModDbl and CModMulK's
numerical methods from locked Qualtran v0.7.0. Doubling uses the original `x+x`
when `x < p`, with no lower-bound check. Constant multiplication first tests
control truthiness, then `x < p`. It does not normalize a nonzero control to one,
validate a bit, cast operands, or repair unsigned overflow. Inactive and exterior
objects retain identity. Register and constant-constructor validation is separate
from these direct value methods.

The existing basis/coherent doubling and controlled-multiplication builders now
consume these numerical functions and retain independent mathematical asserts.
Source primitive/feedback/resource assertions, output port aliases, all-history
clean-work certificates and forced complex histories remain in the original
quantum suites. This does not broaden their quantum residue domains.

G2-VAL10 acceptance adds 271 source value tests, including Python/NumPy types,
arrays, symbolic relations, exterior values and control truthiness. Compiler gate
implementations are unchanged; their previously recorded source discrepancies
outside the valid domain remain open.
