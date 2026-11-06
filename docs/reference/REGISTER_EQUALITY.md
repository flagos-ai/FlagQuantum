# Shared control reduction and equality

The native chain ports pinned Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:
`MultiAnd -> ControlledViaAnd -> MultiControlX -> Equals / EqualsAConstant`.
Source notices retain Google copyright and Apache-2.0 attribution.

`multi_control.py` owns only concrete control-pattern reduction. It reuses
`and_compute.py`'s exact four-T compute and measured erasure. `multi_and_ir`
computes prefix conjunctions into n-1 initially zero workspace wires. These are
logical outputs at the intermediate boundary, not clean scratch. Its explicit
uncompute mode requires those correct prefix values and clears them in reverse
source order. At least two controls are required for the ladder.

`multi_control_x_ir` consumes that compute/erase pair, applies one CNOT from
the final flag to an arbitrary target, and returns all work to zero. For zero
controls it emits X, and for one control it emits CNOT, with source X conjugation
for control-on-zero. Both special cases need no work or measurements. Larger
cases need n-1 clean wires and n-1 distinct classical slots, all caller-owned.
Control values are explicit zero/one tuples. Symbolic controls and non-X Pauli
operations are not included in this acceptance scope.

`equality.py` owns two narrow compositions:

- `equality_ir(left, right, target, workspace, classical_bits)` follows the
  source `Xor(left,right) -> MultiControlX(all zeros) -> Xor(left,right)` chain.
- `equality_constant_ir(source, value, target, workspace, classical_bits)` uses
  the constant's exact unsigned bit pattern as control values.

Registers are big-endian, equal positive width for register equality, with
n-1 clean work wires and n-1 measurement slots (none at width one). Target is
XOR-updated and need not be zero. Inputs and all other amplitudes are retained.
No parallel control implementation, new IR type or runtime allocator is added.

## User example

```sh
python -m examples.algorithms.register_equality
```

The example uses `circuit.append(Equality())` or `Equality(5)` for three-bit
register and constant equality. All input pairs and both target values carry
complex amplitudes. Every sampled output matches the full expected complex
state, both work wires return to zero, and repeating equality restores the
input. Printed counts describe one comparison, not the round trip. No reference
package is needed to run it.

## Executable counts and separate model

| Operation | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean work |
|---|---:|---:|---:|---:|---:|---:|
| Eight-control X, all ones | 43 | 0 | 28 | 28 | 7 / 7 | 7 |
| Eight-control X, all zeros | 43 | 32 | 28 | 28 | 7 / 7 | 7 |
| Eight-bit register equality | 59 | 32 | 28 | 28 | 7 / 7 | 7 |
| Eight-bit constant equality, 83 | 43 | 16 | 28 | 28 | 7 / 7 | 7 |

The primitive gate set includes X/H/S, CNOT, T/TDG, conditional CZ, measurement
and reset. The full executable upstream and native gate multiset, operands,
per-wire order and feedback metadata agree. Disjoint operations may commute in
the upstream DAG export; no cancellation or simplification is used in parity
checks. Model T uses independent `QECGatesCost.total_t_count()` at 4 T per And;
raw model Clifford fields remain coarse operation counts, not primitive totals.

The 41-case control suite covers 0 through 8 controls with all-zero, all-one and
alternating patterns, plus small complete complex states, entangled spectator,
all sampled measurement outcomes, clean work and involution. Separate prefix
compute tests check its intermediate logical outputs without calling them clean.
The 45-case equality suite compares source at widths 1, 2, 3, 5 and 8, including
register QUInt/QInt/QMontgomeryUInt/QAny and concrete constants. Small native
checks cover all complex input/target states, spectator entanglement, cleanup
and double-XOR restoration. Validation rejects aliases, invalid bit patterns,
workspace and measurement slots. Source is not patched for these comparisons.

Batch acceptance: all 646 combined G1/G2 regression cases pass. Both new modules
reach 100% statement coverage. Strict mypy passes on 932 source files. Floor and
current Black, Ruff, architecture boundaries and the multi-level IR contract
pass. Local module acceptance does not replace whole-PR CI/GPU or review.
