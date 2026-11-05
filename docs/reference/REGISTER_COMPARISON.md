# Unsigned constant comparison

`compiler/registers/comparison.py` ports `LessThanConstant` from Qualtran v0.7.0,
commit `bcb17bb25360ed68ce3bc084645db3e929799389`, retaining Google copyright
and Apache-2.0 source notices. It consumes the shared `and_compute_ir` and
`and_uncompute_ir`; no comparison-specific And or measurement implementation is
introduced.

`less_than_constant_ir(source, threshold, target, workspace, classical_bits)`
preserves the big-endian unsigned source and XORs `(x<threshold)` into target.
The target can start in either state or be entangled with the source. Thresholds
are concrete nonnegative integers and source width n is positive.

`less_than_constant_workspace_size(n, threshold)` returns n+1 for the normal
source path: one initially zero equality flag and n clean prefix wires. Callers
also provide n distinct nonnegative measurement slots. The forward scan and
reverse cleanup order follow the source; all quantum work ends in zero. The
cleanup reverses component calls, not the internal instruction order of a
measurement-assisted And erase.

When threshold is at least `2**n`, the source's always-true branch emits just X
on target, with no quantum workspace or classical slots. Threshold zero retains
the complete source prefix network, despite implementing a false predicate.
No constant-specific optimization is introduced. Other comparison classes,
symbolic parameters and general controlled construction remain pending.

## Runnable example

```sh
python -m examples.algorithms.register_comparison
```

The example uses `circuit.append(ConstantComparison(threshold))` without either
reference dependency. It checks every source/target complex amplitude for
three-bit x and threshold 5, and clears four work wires. Repeating the comparison
restores the original state. Threshold 8 checks the single-X, zero-work path.
Printed gate counts describe one comparison, independently of the round trip.

## Measured gates and independent model

| Width, threshold | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean quantum work |
|---|---:|---:|---:|---:|---:|---:|
| 3, 5 | 26 | 6 | 12 | 12 | 3 / 3 | 4 |
| 8, 83 | 68 | 10 | 32 | 32 | 8 / 8 | 9 |
| 8, 0 | 64 | 2 | 32 | 32 | 8 / 8 | 9 |
| 8, 256 | 0 | 1 | 0 | 0 | 0 / 0 | 0 |

The primitive gate set is X/H/S, CNOT, T/TDG, conditional CZ, measurement and
reset. Complete executable source and native gate counts, operands, per-wire
ordered dependencies and all classical metadata agree. The source DAG exporter
may commute disjoint operations; no cancellations or rewriting are used to
obtain equality. Output wire mappings are checked as well.

Independent `QECGatesCost` retains n And nodes and n measured inverses for normal
paths. T values use its explicit `total_t_count()` conversion at 4 T/And.
The raw model's Clifford field excludes parts of the And primitive expansion
and is not treated as a primitive gate total. Model T and measurement agreement
are separate from the direct executable comparison. The always-true model has
one Clifford and no T or measurements, matching its sole X instruction.

## Local verification

51 new cases compare unmodified source at widths 1, 2, 3, 5 and 8 against zero,
one, middle, maximum, full-range and above-range thresholds. Native widths 1
through 4 compare every source/target amplitude with an entangled spectator,
work cleanup and double-XOR restoration. Every measurement result in each small
case appears in the 256 sampled trajectories; no exact probability assertion
is inferred from those samples. Invalid thresholds, widths, work layouts,
wire aliases and classical slots are rejected.

All 560 combined regression tests pass. The new module has 100% statement
coverage. Strict mypy passes on 930 source files. Both Black versions, Ruff,
architecture boundaries and the multi-level IR contract pass. Whole-PR remote
CI/GPU and review remain separate acceptance conditions.
