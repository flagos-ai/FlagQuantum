# Rotation synthesis numerical contracts

The internal synthesis scalar and arithmetic modules cover these Qualtran
v0.7.0 files at commit `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Item | Frozen file | Git blob |
| --- | --- | --- |
| Q-0820 | `rotation_synthesis/_math_config.py` | `ea435fa802e0dabc272d3ac52065b709fd441b55` |
| Q-0821 | `rotation_synthesis/_typing.py` | `66bdc937ad691a24bb523d64c2d77cc34de36031` |

`flagquantum/synth/_rotation_types.py` supplies the numerical annotation
types and the runtime integral predicate. The predicate admits Python
bool and NumPy integer scalars as the source does. It rejects NumPy bool
and integer-valued floating and arbitrary-precision scalars.

`flagquantum/synth/_rotation_math.py` supplies immutable float, NumPy and
mpmath backends with the same callback injection, constant construction,
scalar operations and rounding. The NumPy constructor uses the platform's
`longdouble`. On the acceptance host that type has a 52-bit mantissa
field. The independent tolerance helper preserves the strict absolute or
first-operand relative comparison. The arbitrary-precision backend
preserves the inclusive decimal tolerance and the source's process-global
precision change. Previously created backends observe changes in later
operation calls. Tests scope this state with `mpmath.workdps`.

The native NumPy and float paths load without mpmath. Calling the
arbitrary-precision factory requires mpmath in the environment, as in the
locked reference environment. These files declare numerical operations.
Their direct rotation-synthesis consumers have separate ledger items for
channel, lattice, integer ring and matrix work. Circuit amplitudes,
primitive counts and scratch checks apply at those consumers.
