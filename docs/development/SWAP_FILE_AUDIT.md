# Basic SWAP source family

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original files retain their
Apache-2.0 attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0259 `swap.ipynb` | `b2ccb653c700e5a2628a6b6649218dbc68d8a118` | Closed by the complete native tutorial. |
| Q-0260 `swap.py` | `5c2481f7a1ee0a592d1192999ddf7e6dba8facc3` | The concrete and symbolic register families and exact Fredkin stream are verified. Generic foreign wrappers and the source nested control limitation remain open. |
| Q-0261 `swap_test.py` | `9f5d563a9e642e05e46bb4141438686a7b349a08` | The original suite and native counterparts for executable paths run. Generic factory and serialization compatibility remain open. |

## Signatures and execution

The atomic SWAP binds `x, y`, and atomic Fredkin binds `ctrl, x, y`.
The wider versions keep both data words through-going and exchange corresponding
bits in most-significant-bit order. Both exact operations are self-adjoint.
The native tutorial retains the two atomic signatures and the symbolic, 4-bit,
and 64-bit signatures for both register operations. It presents the symbolic
one-level graphs with weight `n` on the corresponding atomic child. The
4-bit and 64-bit signatures and graph weights are verified as metadata.

The tutorial renders the locked exact Fredkin sequence in the native circuit
drawer. A two-bit pair is appended as a register operation and checked against
the entire expected complex permutation in both control sectors. The inverse
restores identity. Existing tests cover noncontiguous operands, full
amplitudes with a spectator, and every tested branch of the outer control
measurement cleanup. Exact and relative-phase Fredkin paths retain their
separate behavior.

## Emission and resource accounting

| Path | Emitted instructions | Declared cost |
|---|---|---|
| Width `n` SWAP | `n` native SWAP instructions | `n` Cliffords, peak `2n` |
| One exact Fredkin | 8 CX, 2 H, 4 T, 3 Tdg | One Fredkin leaf. The source legacy exact synthesis reports 7 T and 10 Cliffords. Its default QEC T-equivalent policy assigns 4 to one Fredkin. |
| Width `n` exact CSwap | `n` copies of that exact Fredkin stream | `n` Fredkin leaves, peak `2n + 1` |

These paths use no scratch or measurements. The conditional wider or nested
paths have separately verified shared predicate work and feedback. The locked
source raises `TypeError` when a merged multi-bit control group is wrapped
again in the tested case. The native adapter retains an explicit refusal.

The original tests cover matrices, classical permutations, Cirq and PennyLane
conversion, bound decomposition, exact legacy costs and the small, large and
symbolic factories. Existing native controlled and classical tests provide the
gate and value evidence. Generic source serialization, foreign method
dispatch and symbolic factory equivalence remain separate from that coverage.

The original 37 cells and 39 native cells cover all four source examples.
The physical Clifford+T circuit and the register permutation checks run via
the ordinary `circuit.append(op)` path.

Both notebooks passed independent kernel execution. The native kernel rejected
Qualtran, Quimb and Cirq imports. All six SVGs and the Fredkin PNG were inspected.
The original suite recorded 38 passes and eight mode skips. The concentrated
suite passed 623 cases and the final focused suite passed all ten cases,
including the six source factory shapes and graphs. All 113 integration cases
passed. Strict typing, formatting, team scope, architecture, multilevel
Program IR and public API checks passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_swap.ipynb
```
