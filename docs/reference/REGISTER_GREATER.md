# Two pinned unsigned greater-than decompositions

Both paths port unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, with Google copyright and
Apache-2.0 notices retained. They compute the same arbitrary-target XOR
`target ^= (a>b)` while preserving big-endian unsigned inputs and clean work.
They are different source algorithms; their resources are not conflated.

## Ownership and explicit work

`linear_comparison.py` implements `LinearDepthGreaterThan(signed=False)`.
`linear_greater_ir(left,right,target,workspace,classical_bits)` follows the
source carry-only addition prefix, copies the sign result to target, and
uncomputes all carries. Both input registers have the same positive width.
`linear_greater_requirements(n)` returns quantum work and classical slots:
`(n+2,n)` for n>1, with two zero sign bits and n clean carry flags. At width
one it retains the source `MultiControlX(1,0)` special case using `(1,1)`.
The existing shared And and measured erasure own the leaf decomposition.
Signed and controlled variants are not included in this interface.

`ordering.py` implements the original `GreaterThan` wrapper: call the existing
`less_equal_ir`, then apply X to target. It does not copy the comparison tree.
`greater_ir(left,right,target,workspace,classical_bits)` accepts unequal input
widths with exactly the shared `less_equal_requirements(m,n)` allocation.
All prefix/sign/carry flags start and end in zero. Arbitrary target values and
relative phases are retained apart from the specified XOR.

## Runnable user example

```sh
python -m examples.algorithms.register_greater
```

The example uses `circuit.append(Greater("linear"))` or `Greater("tree")` for
three-bit inputs. Each path checks all complex input/target amplitudes, clean
work and repeated-comparison restoration, without either reference package.
Printed gate counts cover one comparison. The route is explicit so the two
source algorithms and their work budgets can be compared without a silent
compiler substitution.

## Measured gates and independent model

| Route and width | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean work / classical slots |
|---|---:|---:|---:|---:|---:|---:|
| Linear, 1 | 7 | 4 | 4 | 4 | 1 / 1 | 1 / 1 |
| Linear, 3 | 35 | 10 | 12 | 12 | 3 / 3 | 5 / 3 |
| Tree, 3 | 75 | 12 | 20 | 20 | 5 / 5 | 8 / 5 |
| Linear, 8 | 95 | 20 | 32 | 32 | 8 / 8 | 10 / 8 |
| Tree, 8 | 225 | 22 | 60 | 60 | 15 / 15 | 23 / 15 |

These are executable X/H/S, CNOT, T/TDG, conditional CZ, measurement and reset
counts. Each path matches its own unmodified source stream, including operands,
per-wire dependencies, measurement slots and feedback; none is asserted to be
the gate sequence of the other. Independent `QECGatesCost` T uses explicit
And-to-T conversion; coarse Clifford model units remain separate from primitive
gate totals. No cancellation of source X gates or carry/prefix operations is
used to obtain agreement.

The linear reference materializes the original Bloq call chain before primitive
conversion, retaining its measured And decomposition instead of a generic
unitary-controlled-X synthesis. The tree wrapper retains its original composite
call into the unmodified Cirq-style LessThanEqual generator. The reference
adapter binds temporary names deterministically, without patching algorithm
source. Disjoint source operations may commute during DAG export.

## Verification scope

The linear suite has six source widths (1,2,3,4,5,8), four complete native
amplitude cases (1 through 4 bits), and invalid-binding checks. The tree wrapper
suite covers eleven equal/unequal width pairs and five native amplitude cases,
including both unequal directions. Shared independent amplitude checks cover
every input pair and both target values, clean all quantum work, observe every
small-case measurement result in 256 trajectories and verify double-XOR
restoration. Exact branch probabilities are not inferred from finite samples.
These 27 new tests are run alongside the previous modules at batch acceptance.

Batch acceptance: all 702 combined G1/G2 regression cases pass, with 27 new
cases in this batch. Both new modules have 100% statement coverage. Strict mypy
passes on 936 source files. Both Black versions, Ruff, architecture boundaries
and the multi-level IR contract pass. Whole-PR remote CI/GPU and review remain
independent final acceptance conditions.
