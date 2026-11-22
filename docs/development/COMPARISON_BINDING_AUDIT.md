# Complete comparator binding dispatch

The frozen reference is Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Copyright 2026 Google LLC,
Apache-2.0. The immutable Qualtran reference remains v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. All eight source blobs in
`tests/unit/fixtures/comparison_binding.json` are checked before regeneration.

## Complete file ownership

| Parent | Complete source file | Native owner and disposition |
| --- | --- | --- |
| K-0222 | `py/circuit/subs/flip_if_cmp.pybind.cc` | `compiler/registers/comparison/physical_comparison.py` delegates all three relations to existing generators. Full dispatch is verified within the safe workspace domain. Keep the parent open for the original mixed-word empty-workspace crash below. |
| K-0223 | `py/circuit/subs/flip_if_cmp.pybind.h` | All three declarations map to the existing physical comparator's relation argument, target, equality, control and tolerance ports. Complete. |
| K-0224 | `py/circuit/subs/flip_if_cmp_test.py` | All 33 unchanged original tests execute with the actual reference extension. Native complete physical streams, complex branch and append evidence are retained by the existing comparison corpus and the new binding suite. Complete. |
| K-0185 / K-0186 | `py/circuit/control_helper.pybind.cc` and header | Previously accepted `registers/primitives/control_binding.py` owns Z-enable and X-output resolution. Reused unchanged. `lowering/physical_trace.py` now also preserves the exact nested physical scopes and shifts every descendant call range. Field addition shares this scope owner. The earlier source move-over-live-owner boundary remains in REGISTER_CONTROL_BINDING.md. |
| K-0261 / K-0262 | `py/val/converted_array.pybind.cc` and header | Previously accepted `registers/layout/operand_word.py` owns scalar/sequence conversion, typed words, integer bounds, singleton distinction and owned immutable views. Reused unchanged. |
| K-0225 | `py/circuit/subs/fuzz_test_util.py` | Existing independent branch/primitive execution replaces the comparator's immediate test consumer. Full generic helper remains open, including arbitrary reference-code contexts, batch scheduling and diagnostic behavior. |

Compiler owns port resolution, deterministic instruction generation, call ranges
and lowering. Runtime executes the resulting native instructions. Foreign
objects stay in the optional fixture generator. No reference package is required
for ordinary execution or the fixture regression.

## Interface, bit order and ownership

The native boundary uses MSB-first words and existing typed operand identities.
A scalar integer RHS is a literal bounded by the LHS width. Quantum RHS wires
are supplied as a word. A scalar quantum LHS is admitted by the existing quantum
word binder. None retains the existing phase shorthand. Existing X-basis
operands and `|+>` / `|->` select guarded phase, skip and unconditional phase.
Boolean and classical Z-controls use the existing control binder.

Source arrays are LSB-first. Reference fixtures reverse native views while
retaining physical IDs. Source workspace comes from `min(num_free_qubits,
width)` and is leased LIFO. Native callers supply explicit clean workspace in
that lease order and retain its ownership, including unused wires. The source
builder's default available capacity is infinite. Setting an explicit maximum
is necessary to compare a bounded pool. Every successful source fixture calls
the operation twice, checks that allocated quantum ownership returns to its
entry count and compares both complete records against two native emissions.

The source opens the X-target guard before the Z-enable guard and closes them
in reverse order. Skip occurs after input conversion and guard construction,
before generator marks and workspace leasing. This can leave a balanced empty
classical scope. Repeated reads of one bit remain nested. Every classical input,
including sparse target/enable slots, is reserved before choosing HMR results.
The shared physical scope adapter keeps generator marks inside those scopes.
The effective-condition path used by earlier equality bindings retains its
existing ownership and behavior.

## Gate, state and execution evidence

The fixture contains 5,520 exact complete record comparisons across widths
0, 1, 2, 4 and 8, quantum/classical/mixed/integer RHS, quantum/phase/classical/
skipped output, quantum/literal/classical/shared-bit control, finite tolerance,
equality inclusion and full/empty clean pools. There are 2,480 less-than,
2,480 greater-than and 560 equality cases. Another 480 source workspace
refusals agree with native ValueError. The 120 unsafe source configurations
are excluded before calling C++, with the observed witness retained below.
The fixture retains operations, ordered operands, marks, resets, HMR, bit IDs
and nested conditions. Only register declarations and the native transport
header are excluded. KMX encode/decode round trips preserve every native call.

Reuse the accepted 1,262-case generator corpus, specialized generator tests,
wide integer-word conversion and the unchanged 33 reference binding tests.
The new small coherent cases exhaust every reset/HMR history for each relation,
quantum/phase target and all target/enable classical values. Unnormalized
complex amplitudes, total probability and both work wires are checked. Inactive
guards suppress measurement and only the all-zero inactive history has weight.
Independent normal `DynamicCircuit.append(op)` execution verifies phase sign.

## Emitted resources and analytical models

Two-bit classical RHS, classical phase target b5, classical control b6 and two
clean workspace wires give the following static physical records.

| Relation | Operations | X | CX | CCX | Z | CZ | NEG | R | HMR | Push/pop pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| less than | 26 | 13 | 1 | 0 | 1 | 1 | 2 | 1 | 1 | 3 |
| greater than | 29 | 13 | 2 | 0 | 2 | 1 | 3 | 1 | 1 | 3 |
| equal | 21 | 8 | 1 | 1 | 1 | 2 | 0 | 2 | 2 | 2 |

All three examples use five physical quantum wires and eight classical slots,
including the sparse external guard IDs. Scope wrapping contributes zero
quantum primitives and four scope records for two guards. Counts describe the
emitted stream. An inactive guard executes no guarded quantum primitive.

The binding and control helper declare no analytical resource formula. Existing
comparison model owners remain independent of these measured counts. The prior
256-bit primitive and source benchmark/model evidence remains in
KICKMIX_COMPARISON_FILE_AUDIT.md. No T-equivalent substitution is presented as
an emitted primitive count.

## Remaining source boundary

The unchanged native extension exits with SIGSEGV for width two, a mixed RHS
(q2, b0 in source LSB order), quantum output q3, quantum control q4, zero clean
workspace, less-than, exact tolerance and false equality. The source's full
mixed ladder indexes `Q_clean[0]` after taking an empty span. The native
comparison already rejects insufficient clean workspace with ValueError.
No reference source was changed. K-0222 stays open under the ledger's original
source-defect acceptance policy. Widths four/eight with the same unsafe
workspace shape are excluded conservatively, together with affected target/
control/relation variants. Those exclusions are separate from proven crashes.

Earlier dependency warnings, pending Proposals 116–119, other original source
failures and complete G2 acceptance remain independent. The C++ move assignment
boundary and generic fuzz helper remain as documented by their owning files.

## Reproduction

Run native checks with
`python -m pytest -q tests/unit/test_comparison_binding.py tests/unit/test_kickmix_comparison_file.py tests/unit/test_kickmix_field_addition_file.py tests/unit/test_kickmix_control_binding.py`.

Regenerate using the already built frozen extension with
`python tests/unit/fixtures/generate_comparison_binding.py --source-root <frozen-kickmix-root> --output tests/unit/fixtures/comparison_binding.json`.
The wrapper must expose the repository and frozen extension paths, and the
extension must expose version 0.1 and Simulator. The generator checks all eight
source blobs, runs actual bound methods, checks pool return and rejects silent
reference absence.
