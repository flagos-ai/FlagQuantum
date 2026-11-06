# Relative-phase exchange and indexed routing

The shared register compiler ports the concrete Qualtran v0.7.0
`CSwapApprox` and `SwapWithZero` decompositions. Both emit native
Clifford+T instructions on caller-provided wires. They use zero scratch
qubits, zero measurement events and zero classical slots.

`relative_controlled_swap_ir` accepts one control and two equal nonempty
words. It applies the source rotation and CNOT layers around a dirty-target
fanout. This conjugation is self-adjoint and retains the original relative
signs. Sign extension and both modular negation emitters use the same
`bitwise.fanout_ir` helper.

`swap_with_zero_ir` accepts big-endian selection words, axis extents and
row-major target words. It visits the child subtrees before each axis'
exchange layers. The selected row arrives at row zero for in-range
addresses. Its adjoint reverses the exchange sequence and applies the same
self-adjoint relative-phase exchange at each edge. The source omits zero-bit
selection registers and fails to bind a later axis with actual exchanges.
The native entry point rejects that configuration explicitly. Truncated
axes, trailing zero-bit singleton axes and an identity singleton are
covered by source comparisons.

Gate tests read the locked Cirq decomposition of each relative-phase
exchange and the original SwapWithZero Soquet connections. Complex tests
compare every output amplitude of a dense input entangled with a spectator,
including padded addresses of truncated axes, and verify inverse restoration
at absolute tolerance 2e-12. The largest primitive exchange checked has
eight bits per word. The network checks cover one, two and three axes.

| Scope | Literal T + T-dagger | Literal CX | Source T model | Total qubits |
| --- | ---: | ---: | ---: | ---: |
| One exchange of eight-bit words | 32 | 47 | 32 | 17 |
| Six two-bit words on a 2-by-3 grid | 40 | 55 | 40 | 15 |

For word width b and N target rows, the network emits N-1 exchanges.
Each contributes 4b T-family gates, 6b-1 CX and 16b single-qubit Clifford
gates. The source QEC and qubit models are checked independently of the
instruction scan.

The local dependency regression passed 249 tests, including 15 new cases
and the paired lookup measurement tests. Both new emitters have 100%
statement coverage. The locked blobs for `cswap_approx.py`,
`swap_with_zero.py` and `multi_target_cnot.py` match the inventory.
