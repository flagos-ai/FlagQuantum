# API Change Proposal 098: Gate Coverage Batch A

## Status

**Implementation and adjoint-completeness amendment approved; all four batches
are implemented on `feat/gate-coverage-batch-b` and pending review.**

The API owner approved the original five-name implementation scope with:

```text
approve API-CHANGE-098-GATE-COVERAGE-BATCH-A-IMPLEMENTATION
```

Pre-implementation validation then found that three proposed fixed gates cannot
satisfy FlagQuantum's existing composition contract under that surface alone:
`iswap` and `csx` are not self-inverse, while the inverse of `dcx(q0, q1)` is
`dcx(q1, q0)`. The amendment below is therefore required before implementation.

The API owner approved that amendment with:

```text
approve API-CHANGE-098-GATE-COVERAGE-BATCH-A-ADJOINT-AMENDMENT
```

This document governs an additive Stable Core API change. Its implementation adds
the approved local gate semantics while making no provider or QPU support claim.

## User outcome

After implementation, users can write the same named gates on static and dynamic
builders without dropping to raw opcode strings or custom matrices:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

static = fq.Circuit(3).h(0).iswap(0, 1).ecr(1, 2)
restored = static.adjoint()  # uses iswapdg for the iSWAP inverse

dynamic = (
    DynamicCircuit(3)
    .ch(0, 1)
    .csx(1, 2)
    .dcx(0, 2)
    .measure(2, classical_bit=0)
)
```

The example is executable on this implementation branch and becomes part of the
documented public surface when the implementation merges.

## Scope

Batch A adds seven fixed two-qubit unitary opcodes:

- `iswap`
- `iswapdg`
- `ecr`
- `ch`
- `csx`
- `csxdg`
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

### Batch A1: `iswap` and `iswapdg`

Add the paired schemas, dense matrices, exact current-basis decompositions, builder
typing, IR round-trip, local engine conformance, OpenQASM behavior, and ecosystem
adapter tests. The two opcodes name each other as their adjoint, so
`Circuit.adjoint()` remains closed over registered opcodes. `iswap` is the first
slice because it is a common interaction gate and does not require a parameter or
a global-phase-bearing controlled-unitary contract.

### Batch A2: `ch`, `csx`, and `csxdg`

Add the single-control gates together. `ch` is self-inverse; `csx` and `csxdg`
name each other as adjoints. Their control metadata must agree with the existing
`h`, `sx`, and `sxdg` semantics, and any further-control synthesis must stay inside
the existing bounded, ancilla-free policy or fail closed.

### Batch A3: `dcx`

Add the double-CNOT operation with an exact two-`cx` lowering and prove the chosen
operand order through asymmetric basis-state tests. Extend the internal adjoint
rewrite result so this one schema can request reversed operands; do not add a
duplicate inverse opcode. Existing gates continue to preserve operand order.

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

- [x] All seven canonical schemas are complete and registry validation passes.
- [x] `Circuit` and `DynamicCircuit` expose typed lowercase and uppercase methods.
- [x] Direct and decomposed operators agree under explicit qubit-order tests.
- [x] Local executors either pass conformance or fail closed by opcode and backend.
- [x] Compiler targets contain no assumed support cells.
- [x] Ecosystem round-trips preserve semantics where a corresponding gate exists.
- [x] Remote capability matching uses post-routing opcodes and remains fail closed.
- [x] API snapshots, reference documentation, and release notes are updated.
- [x] Every implementation PR contains a runnable user-facing example.
- [x] Existing gate and dynamic-circuit suites remain green.

## Implementation approval requested

```text
approve API-CHANGE-098-GATE-COVERAGE-BATCH-A-IMPLEMENTATION
```

Approval authorizes the four local, offline implementation slices above. It does
not authorize provider access, credentials, network requests, task submission,
charging, or a remote-QPU capability claim.

## Adjoint-completeness amendment approval requested

```text
approve API-CHANGE-098-GATE-COVERAGE-BATCH-A-ADJOINT-AMENDMENT
```

This amendment adds only the two inverse spellings required to keep named-gate
adjoints closed and an internal operand-permutation rule for `dcx`. It does not
add another gate family or broaden the provider and hardware boundaries above.
