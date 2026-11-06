# Tree register less-or-equal comparison

This batch ports the unmodified Qualtran v0.7.0 source at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, retaining Google copyright and
Apache-2.0 notices. The native dependency chain is:

`SingleQubitCompare / BiQubitsMixer -> LessThanEqual`, with shared
`And / measured And erase` and `MultiAnd` for longer prefixes.

## Component ownership and intermediate outputs

`comparison_tree.py` owns two concrete compute/adjoint components:

- `single_qubit_compare_ir` preserves a, stores equality in b, and writes a<b
  and a>b into two initially zero logical output wires. Its source adjoint
  restores b and clears those outputs; only the less-than flag is measured.
- `bi_qubits_mixer_ir` acts on two two-bit inputs and three initially zero
  work wires. It preserves the sign of the comparison in the low input bits,
  retaining the two source And-assisted swaps. Its adjoint restores inputs and
  clears all three work wires, measuring the last two only.

Forward outputs include logical flags/junk. They are not described as clean
scratch until the correct adjoint has run. The source reverses component order
while using measured And adjoints internally; native lowering does the same,
without reversing the primitive measurement instructions themselves.

`less_equal.py` owns only allocation binding and composition of those components.
`less_equal_requirements(m,n)` returns `(quantum_work, classical_slots)`.
`less_equal_ir(left,right,target,workspace,classical_bits)` XORs unsigned a<=b
into an arbitrary target, with big-endian inputs of positive widths. Every work
wire starts and ends in zero. Quantum operands must be disjoint and classical
slots distinct/nonnegative. There is no new runtime allocator or IR contract.

Equal widths use the literal divide-and-conquer tree and final single-bit
compare. Unequal widths retain each source prefix-zero path: bit difference 1
uses X/CNOT, difference 2 uses And(0,0), larger differences use MultiAnd. The
right-longer and left-longer target updates differ exactly as in source. A final
And combines prefix equality with the suffix result before reverse cleanup.
Symbolic widths and other comparison classes remain pending.

## Runnable user example

```sh
python -m examples.algorithms.register_less_equal
```

`circuit.append(LessEqual())` compares a two-bit input with a three-bit input.
The example checks every complex input/target amplitude, all seven quantum work
wires in zero at exit, and restoration by repeating the comparison. Four
classical measurement slots suffice. No reference package is needed. Printed
gate counts cover one comparison, separately from the restoration run.

## Executable gates versus model

| Input widths | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Quantum work / classical slots |
|---|---:|---:|---:|---:|---:|---:|
| 3, 3 | 75 | 11 | 20 | 20 | 5 / 5 | 8 / 5 |
| 8, 8 | 225 | 21 | 60 | 60 | 15 / 15 | 23 / 15 |
| 3, 8 | 106 | 35 | 40 | 40 | 10 / 10 | 13 / 10 |
| 8, 3 | 105 | 34 | 40 | 40 | 10 / 10 | 13 / 10 |
| 1, 5 | 40 | 27 | 20 | 20 | 5 / 5 | 6 / 5 |

The primitive gate set includes X/H/S, CNOT, T/TDG, conditional CZ, measurement
and reset. Complete source/native gate multisets, operands, per-wire ordered
dependencies and classical feedback agree. All gates touching quantum work and
all measured work flags are checked, with actual measurement counts separate
from total work. Independent `QECGatesCost` T uses its explicit And-to-T
conversion. Raw Clifford model units are not asserted equal to primitive totals.

For this Cirq-style source, parity tests read the original
`decompose_from_registers` generator directly. Exporting its already-flattened
DAG can reorder independent allocations and hence temporary wire names. The
reference allocator deterministically recycles temporary RIGHT-output mapping
names in caller order before later generator allocations; final primitive gates
are unchanged, and no algorithm source is patched. This binding detail is
separate from the physical work count in the table.

## Verification scope

The component suite compares source compute/adjoint pairs, verifies all basis
truth/sign cases on forward logical outputs, and checks complete complex states
and work cleanup. Component T/measurement costs are independently compared.
Less-or-equal source parity covers equal widths 1, 2, 3, 4, 5 and 8; unequal cases
cover both directions and prefix differences 1, 2, 3, 4 and 5. Native small cases
check every input pair and both target values with complex amplitudes, clean
work and repeated-comparison restoration. Sampling checks all observed branch
states without inferring exact probabilities. Aliases, invalid widths, work
layouts, adjoint flags and classical bindings are rejected.

Batch acceptance: 29 new cases and all 675 combined G1/G2 regressions pass.
Both new modules reach 100% statement coverage. The small native comparator
cases observe every measurement result in 256 trajectories, checking the full
complex output and clean work on each trajectory. Strict mypy passes on 934
source files. Both Black versions, Ruff, architecture and multi-level IR checks
pass. Whole-PR CI/GPU and review remain independent final requirements.
