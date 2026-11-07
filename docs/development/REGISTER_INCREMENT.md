# Measured register increment

G2-KADD06 owns the locked `gen_iadd1` schedule in Compiler registers. The
interface binds a big-endian target, an optional quantum control (`True` means
uncontrolled), caller-provided clean workspace in borrowing order, and one
reusable classical measurement slot. It adds one modulo the target width when
enabled. The control and all supplied work wires are preserved; used clean
wires must enter and leave zero. There is no dirty workspace and no new Core
operation or public export.

The upstream helper traverses little-endian target wires. Reverse the bound
target once, preserve every CCX/CX operand and loop order, and map HMR plus
conditional CZ through the existing measured-carry erasure primitive. Source
reset operations are represented by the explicit clean-input precondition.
Uncontrolled width n borrows max(0,n-3) clean wires; controlled width n borrows
max(0,n-2). A measured path requires the result slot and overwrites it at each
HMR. Scope the output classical bit independently of quantum work ownership.

Reference: Kickmix `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
`gen/adders/gen_iadd1.cc`, blob `890b484fb6043d45d8e60a91dd241ab8092f7b21`.
No reference source is modified. Runtime and Simulation keep using existing IR
execution; source-native generation is a test oracle, not a production dependency.

## Source conformance and resources

The native `LoopBuilder` emits the carry-building CCX with the target-word
control before the prior carry control. The port preserves that emitted operand
order, not just the commutative Boolean effect. Both standalone cleanup and loop
feedback retain their original CZ order and reusable `b0` slot.

The fixture records 68 complete stream digests and opcode counts: widths 0–16,
controlled/uncontrolled, and minimal/two-extra clean workspace. Only source
register annotations, INFO tags and resets of explicitly pre-cleared carry
inputs are omitted. All other primitive names, ordered operands, feedback
slots and repeated operations contribute to the digest. Regenerate it with:

```sh
python tests/unit/fixtures/generate_kickmix_increment.py \
  --source-root /path/to/locked/kickmix \
  --output /tmp/kickmix_increment.json
```

The generator checks the increment source blob, compiles the supplied unchanged
C++ source and the checked-in oracle, and does not edit the reference tree.
It includes native gate implementations from `py/circuit/ops`, excluding Python
bindings. Compare the resulting JSON with
`tests/unit/fixtures/kickmix_increment.json`.

For an effective controlled-helper width m >= 3, measured primitive counts
are m-1 CCX, m-1 CX, m-2 HMR and m-2 conditional CZ. An uncontrolled increment
uses helper width n-1 and appends one X. These are emitted native primitive
counts, not synthesized T counts or a source model estimate. The four-bit
controlled example has 3 CCX, 3 CX, 2 HMR, 2 CZ, two clean carries and one reused
classical result slot.

Run `python -m examples.algorithms.measured_register_increment` for a complete
`circuit.append(op)` example. Thirty-three new tests cover all small complex
measurement branches, unchanged control, clean and unused workspace, native
basis/absolute phase, invalid bindings and source digests. The concentrated
addition/predicate/native-format regression passes 716 tests without skips.
Strict mypy covers the emitter, example and regeneration tool; Black, Ruff,
compiler scope and architecture checks pass. Reference defects are not repaired
or hidden by a different executable path.
