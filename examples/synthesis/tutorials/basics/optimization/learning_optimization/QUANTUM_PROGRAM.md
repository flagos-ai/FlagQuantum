# Integer and slack optimization quantum program

The complete one-layer quantum program retains two two-bit unsigned integers,
four slack bits and both real parameters. Native ports `x0`, `x1`, `slack` replace
source fields `v.x[0]`, `v.x[1]`, `v.constraint_slack_var` explicitly. The cost is
`-(3*x0+2*x1+20*(s0+2*s1+4*s2+3*s3-3*x0-x1+2)**2)`.
The unusual fourth slack weight is three, not eight; the offset is two.
The mixer applies RX(beta), retaining the original angle convention.

Two nonzero parameter vectors and both objectives match an independent classical
cost table and Circuit parity/RX rotations. Tests compare all raw complex128
amplitudes, absolute phase, declared bit order and zero workspace. Native resources
are 8 qubits, 45 static instructions and depth 16; the source has the same width
and count with depth 15. Maximum instruction width is 2 in both. Local evolution
fusion and storage ordering can alter these IR statistics; no hardware cost or
optimality claim follows.

Run the commands in [README](README.md) for the native program and Circuit comparison.
The complete related Notebook optimizer workflow remains pending.
