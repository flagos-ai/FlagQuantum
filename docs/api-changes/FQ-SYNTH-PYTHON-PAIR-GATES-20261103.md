# Native Python qfunc scalar pair gates

The user authorized direct native Python qfunc calls to CX, SWAP, and CRX on
2026-11-03. The unreleased synth namespace exposes `CX(control, target)`,
`SWAP(first, second)`, and `CRX(angle, control, target)`. The spelling and
positional order match Qmod. Core Circuit signatures remain unchanged.

The gates accept distinct initialized one-bit operands. CX and CRX preserve
the control's basis value, while SWAP writes both operands. The angle is a
finite static real in radians, including signed values. The shared pair-gate
record owns the operand and effect checks and lowers to the existing Core
operators. These markers are captured in qfunc and raise TypeError on host
calls. The supported within/apply and adjoint subsets are unchanged.
