# API Change Proposal 098: Gate Coverage Batch A

## Status

**Proposal prepared; implementation is not approved.**

This document requests an additive Stable Core API change. It creates no gate,
changes no runtime behavior, and makes no provider or QPU support claim.

## User outcome

After implementation, users can write the same named gates on static and dynamic
builders without dropping to raw opcode strings or custom matrices:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

static = fq.Circuit(3).h(0).iswap(0, 1).ecr(1, 2)

dynamic = (
    DynamicCircuit(3)
    .ch(0, 1)
    .csx(1, 2)
    .dcx(0, 2)
    .measure(2, classical_bit=0)
)
```

The example is a proposed API sketch and is not executable until implementation
approval and merge.

## Scope

Batch A adds five fixed two-qubit unitary opcodes:

- `iswap`
- `ecr`
- `ch`
- `csx`
- `dcx`

These names address broadly used algorithmic, interoperability, and superconducting
target vocabulary while remaining exactly representable by FlagQuantum's existing
unitary IR. They do not imply that a selected backend executes the operation
natively.

The implementation must update the existing authoritative operator registry. It
must not create a second gate registry or handwritten copies of the fluent methods.
The schema installer and API Change 097 typing contract then expose the additions
to both `Circuit` and `DynamicCircuit`.

## Semantic contract

For every opcode, implementation must specify and test:

1. one canonical matrix convention, including qubit order and any global phase;
2. arity, parameter list, aliases, adjoint rule, control rule, and power rule;
3. an exact decomposition into the current built-in basis for targets that do not
   declare the opcode natively;
4. preservation through IR serialization and deserialization;
5. agreement between direct matrix execution and decomposed execution in both
   supported complex precisions; and
6. deterministic refusal from a backend or exporter that has neither a native
   implementation nor an approved lowering.

`ecr` is especially phase-sensitive. Its matrix and decomposition must be compared
as operators, not only by measurement probabilities. If an external ecosystem has
multiple conventions, each adapter must map its named convention explicitly.

## Vertical implementation slices

### Batch A1: `iswap`

Add the schema, dense matrix, exact current-basis decomposition, builder typing,
IR round-trip, local engine conformance, OpenQASM behavior, and ecosystem adapter
tests. `iswap` is the first slice because it is a common interaction gate and does
not require a parameter or a global-phase-bearing controlled-unitary contract.

### Batch A2: `ch` and `csx`

Add the two single-control gates together. Their control metadata must agree with
the existing `h` and `sx` semantics, and any further-control synthesis must stay
inside the existing bounded, ancilla-free policy or fail closed.

### Batch A3: `dcx`

Add the double-CNOT operation with an exact two-`cx` lowering and prove the chosen
operand order through asymmetric basis-state tests.

### Batch A4: `ecr`

Add `ecr` last, after its matrix convention and ecosystem mappings are recorded.
Prove direct/decomposed operator equality including global phase and require target
legalization before any remote admission.

Each batch is independently reviewable. A later batch must not be used to make an
earlier batch's unsupported backend silently work.

## Required coverage matrix

For each added opcode, tests must record one of `native`, `exact_lowering`, or
`unsupported` for:

- PyTorch statevector;
- MPS;
- tensor network;
- density matrix;
- JAX where advertised;
- distributed execution where advertised;
- OpenQASM;
- QCIS;
- QIR;
- Qiskit, Cirq, PennyLane, Braket, and CUDA-Q adapters that are present; and
- provider serialization and post-routing capability matching.

An `unsupported` cell is acceptable when it fails before execution with the opcode
and target named. An empty or assumed cell is not acceptable.

## Compatibility

The change is additive: existing opcode meanings, aliases, method signatures,
serialized instructions, and provider behavior remain unchanged. The new names
become protected Stable Core API only after implementation approval and merge.

Adding a named opcode can still affect broad registry-derived behavior. The
implementation therefore must keep snapshot changes intentional and must not edit
contracts merely to silence a failing test.

## Non-goals

- no general `cu` or arbitrary multi-control convenience API;
- no `fsim`, phased-X, ion-trap, neutral-atom, or pulse-native family;
- no new channel opcode;
- no `barrier`, `delay`, pulse instruction, or global-phase IR field;
- no provider SDK, credential, discovery, network, submission, or live-QPU work;
- no claim that Quafu or another QPU supports any Batch A gate natively; and
- no change to DynamicCircuit measurement or remote-execution admission.

## Acceptance criteria

- [ ] All five canonical schemas are complete and registry validation passes.
- [ ] `Circuit` and `DynamicCircuit` expose typed lowercase and uppercase methods.
- [ ] Direct and decomposed operators agree under explicit qubit-order tests.
- [ ] Local executors either pass conformance or fail closed by opcode and backend.
- [ ] Compiler targets contain no assumed support cells.
- [ ] Ecosystem round-trips preserve semantics where a corresponding gate exists.
- [ ] Remote capability matching uses post-routing opcodes and remains fail closed.
- [ ] API snapshots, reference documentation, and release notes are updated.
- [ ] Every implementation PR contains a runnable user-facing example.
- [ ] Existing gate and dynamic-circuit suites remain green.

## Implementation approval requested

```text
approve API-CHANGE-098-GATE-COVERAGE-BATCH-A-IMPLEMENTATION
```

Approval authorizes the four local, offline implementation slices above. It does
not authorize provider access, credentials, network requests, task submission,
charging, or a remote-QPU capability claim.

