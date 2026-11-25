# Induced-subgraph quantum program

Complete twelve-bit, eight-layer quantum program. Two nonzero parameter vectors, both width/depth objectives and independent complete complex128 Circuit states. All4096 classical graph-cost inputs checked. Complete Notebook optimizer workflow remains pending.

Native explicit six-bit rows a and b replace qstruct x[6][2]. The source directed-edge sum counts each undirected edge twice; retain its full square, same-color penalties, constant6, gamma-first/beta-last parameters and every RX layer. Split the commuting square into within-row squares and cross-row product to stay within existing bounded phase expansion. No Compiler/IR/API/budget changes or source importer in user examples. Independent reference uses graph edges and at-most-four-bit truth tables with Walsh spin-parity Circuit construction; no compiler polynomial implementation reused.

The independent classical oracle evaluates
`6 - equal_color_penalties - directed_edge_sum**2` on every twelve-bit input.
It is checked against the small reference cost factors before full-state
Circuit comparison. This protects edge multiplicity, cross-row terms, constant
phase, row/bit order and the empty/all-selected boundary inputs.

Absolute complex128 amplitudes match within 1e-12 without phase alignment.
Compiler work bits are zero. Actual qubit, operation and memory budgets reject
undersized requests. Both nonzero parameter vectors retain all eight layers.

Static IR resources for the first binding are native twelve qubits,34185
operations,depth23477 versus the original twelve qubits,33592 operations,
depth26085. Splitting the cost changes instruction scheduling. These are
instruction-level counts, not hardware-basis costs or a claim of optimum or
uniformly improved resources.

A separate facility-location candidate exceeded the existing4096 captured-effect
budget when its empty-row projectors were expanded directly. It remains
unregistered: preserve its full dimensions and constraints rather than raising
the budget or using it to claim coverage. Numeric local computation over array
views remains a shared follow-on semantic gap.

[README](README.md) lists runnable native/Circuit commands. Optimizer training
and complete Notebook workflows remain separately pending. No PR, remote CI or
release is authorized by this local result.
