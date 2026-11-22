# Complete scalar and broadcast Pauli bindings

Frozen Kickmix revision `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`,
Copyright 2026 Google LLC, Apache-2.0. Immutable Qualtran v0.7.0 remains
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Source files and native ownership

| Parents | Complete source scope | Native owner |
| --- | --- | --- |
| K-0212 / K-0213 | `py/circuit/ops/x.pybind.cc` and header | `registers/primitives/broadcast_binding.py::broadcast_x_ir` |
| K-0214 | `py/circuit/ops/x_test.py` | Native complete binding matrix and append execution in `test_pauli_broadcast_binding.py`, plus all original tests unchanged |
| K-0215 / K-0216 | `py/circuit/ops/z.pybind.cc` and header | `registers/primitives/broadcast_binding.py::broadcast_z_ir` |
| K-0218 | `py/circuit/ops/z_test.py` | Same complete matrix, ordered classical feedback and native complex append evidence |
| K-0217 | `py/circuit/ops/z_pow_test.py` | Existing PhaseAngle128, KMX codec and exact reversible-phase execution. Both complete original test consumers are retained and independently checked. |
| K-0207 | `py/circuit/ops/op_testing_util.py` | Direct original helpers execute through the unchanged X/Z tests. Existing typed port/stream fixtures and native independent state tests support these consumers. Full generic helper disposition remains open. |

All eight source blobs are recorded and checked by the fixture generator.
Already accepted operand conversion, tagged identities, mixed primitive dispatch,
classical XOR and exact scalar-sign owners are reused. The two new compiler
functions bind an existing operand word and concatenate its existing lane
emitters. Ordinary execution uses no external package or new contract.

## Complete dispatch

X on a quantum wire emits X. X on a classical Z-bit inverts that bit. X on a
classical X-bit emits a phase conditioned on its same stored bit. X on the minus
state emits NEG and on the plus state emits no gate. Ordinary Boolean targets
are rejected before emission, including an invalid lane after a valid lane.

Z on a quantum wire emits Z. Z on a classical bit emits its conditioned scalar
sign. True emits NEG and false emits nothing. X-basis targets are rejected.
Empty words emit nothing. Homogeneous and mixed words preserve all selected
physical IDs, lane order, repeated operands and separate scalar signs. Two
literal signs remain two NEG records even when their total phase cancels.

The source's mutable vector fast paths map to immutable native lane emission.
The conversion and output owner fully validates a word before returning an
instruction tuple. Runtime handles execution and classical state. Compiler
allocates no work, performs no measurement and chooses no backend.

## Source and native evidence

The actual frozen extension passes all 32 unchanged X, Z and ZPow tests.
The new fixture records 232 accepted/rejected scalar, empty, homogeneous,
reversed/strided-ID and mixed binding configurations. Every ordered emitted
primitive and effective classical predicate agrees. The underlying exact KMX
primitive stream contains no component marks or measurement scopes for these
bindings. Existing binary-gate fixture verification supplies the shared typed
port and stream normalization owner.

The original ZPow simple consumer distinguishes 0.25, rational 1/10 and binary
float 0.1. Native exact angle words and decimal KMX round trips retain their
difference. The complete original seven-ZPow CCZ circuit is extracted from its
unchanged AST, then executed through native append. A superposition with complex
amplitudes verifies the full CCZ operator, including the global phase. This
closes the complete test file and leaves the broader angle and builder parents
under their existing acceptance decisions.

Normal DynamicCircuit.append execution checks all four classical input values,
ordered bit inversion followed by phase feedback, entangled spectator amplitudes,
quantum X/Z operations and untouched classical slots. There are no work wires
or measurement branches in these source files. The exact-phase execution
consumer has no work or measurement either.

## Resources and models

| Bound operation | Emitted physical result |
| --- | --- |
| X quantum lane | One X |
| X classical Z-lane | One BIT_INVERT |
| X classical X-lane | One conditioned NEG |
| X minus / plus | One NEG / zero records |
| Z quantum lane | One Z |
| Z classical lane | One conditioned NEG |
| Z true / false | One NEG / zero records |
| Original phase CCZ consumer | Seven Z_POW and six CX on three quantum wires |

These figures count emitted records. Scalar-sign native IR expands on an
arbitrary-state anchor and lowers back to one exact NEG. That intermediate IR
expansion is separate from the emitted target count. X/Z binding files declare
no analytical model. Existing component costs and source phase estimators
retain their separate ownership and are reused without equating model output
with measured primitive records.

## Reproduction and boundaries

Run `python -m pytest -q tests/unit/test_pauli_broadcast_binding.py`.
Regenerate with
`python tests/unit/fixtures/generate_pauli_broadcast_binding.py --source-root <frozen-kickmix-root> --output tests/unit/fixtures/pauli_broadcast_binding.json`.
The generator requires the actual built extension, version 0.1 and Simulator,
verifies every original blob and retains atomic source errors.

K-0207 still needs its full general broadcast/mask/read-write utility disposition.
K-0222 retains the original empty-workspace mixed-comparator SIGSEGV. Pending
Proposals 116–119 and previous source defects remain separate G2 gaps.
