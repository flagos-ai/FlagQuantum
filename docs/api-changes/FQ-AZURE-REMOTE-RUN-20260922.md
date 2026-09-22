# Azure Quantum remote run contract

Status: approved and implemented on 2026-09-22.

## Decision

The user explicitly authorized extending the stable `fq.run` behavior to one
Azure Quantum counts workflow without changing its signature. The canonical
call is:

```python
result = fq.run(
    circuit,
    outputs=fq.counts(),
    target="azure:<target-id>",
    shots=100,
)
```

`AZURE_QUANTUM_RESOURCE_ID` selects the workspace. The process must also set
`AZURE_QUANTUM_TARGET_QUBITS` from an authoritative target description; the
adapter does not infer hardware capacity from the submitted circuit.

The integration emits a sealed OpenQASM 3 deployment package, compiles it to
QIR internally through the Microsoft QDK, submits it to the named workspace
target, and normalizes counts or probabilities into the stable
`ExecutionResult` contract.

## Initial capability boundary

The initial root workflow supports one full-register `fq.counts()` request.
It rejects expectation values, partial-register counts, dynamic circuits,
`target_qubits`, explicit compiler selection, missing workspace configuration,
and missing target-width evidence before creating a cloud job. Provider-native
usage remains available through `AzureQuantumProvider` for advanced workflows.

Credentials remain owned by the QDK and Azure identity configuration. They are
not accepted by `fq.run`, embedded in `target`, or stored in deployment receipts.

## Compatibility

The `fq.run` signature is unchanged. Existing local, Jiuding, and Quafu routes
keep their prior validation and behavior. The only Stable Core delta is that a
previously rejected `azure:<target-id>` target becomes valid with the required
environment configuration. QDK remains an implementation detail recorded in
result provenance rather than a user-selected compiler.

## Verification

API contract tests prove the successful counts path, stable result metadata,
pre-submission rejection boundaries, unchanged public signature, and absence of
network or paid hardware use in the test suite. Provider tests independently
cover dry-run behavior, QIR submission, status, cancellation, identity evidence,
and deterministic probability-to-count conversion.
