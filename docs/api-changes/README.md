# API Change Proposals

One document per approved or proposed change to the Stable Core public API,
named `FQ-<AREA>-<DATE>.md`. Each records the decision, authorization,
compatibility analysis, and the evidence that the change is verified.

Follow [public API protection](../development/PUBLIC_API_PROTECTION.md) before
adding a proposal here. The current stable surface is defined by
[`docs/public_api_v1.json`](../public_api_v1.json) and the
[API reference](../reference/API.md); a proposal does not change that surface
until it is approved and the contracts are updated through the integration
process.

- [Public typing corrections for Circuit, execution scope, and CPU probes](FQ-API-TYPING-20260910.md)
- [Public optimizer factory contract](FQ-API-OPTIMIZER-FACTORY-20260910.md)
- [Parameter expression operand validation](FQ-PARAMETER-ARITY-20260910.md)
- [Parameter mapping key contract](FQ-PARAMETER-MAPPING-20260910.md)
- [Runtime Module object typing](FQ-RUNTIME-MODULE-TYPES-20260910.md)
- [Qubit terminology migration](FQ-QUBIT-NAMING-20260913.md)
- [Complete the qubit terminology migration on the public surface](FQ-QUBIT-TERMINOLOGY-COMPLETION-20261002.md)
- [Azure Quantum remote run contract](FQ-AZURE-REMOTE-RUN-20260922.md)
- [CPU noisy-MPS counts through `fq.run`](FQ-CPU-NOISY-MPS-COUNTS-20260924.md)
- [Circuit expressiveness contract](FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md)
- [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md)
- [Compiler boundary responsibility split](FQ-COMPILER-BOUNDARY-SPLIT-20260930.md)
- [Pauli algebra and quantum_info ownership boundary](FQ-PAULI-QUANTUM-INFO-BOUNDARY-20260930.md)
- [Quafu credential presence fails closed before submission](FQ-QUAFU-CREDENTIAL-FAIL-CLOSED-20260930.md)
- [Quafu service compilation through `fq.run`](FQ-QUAFU-SERVICE-COMPILATION.md)
- [Remote job lifecycle](FQ-REMOTE-JOBS.md)
- [Zero-noise extrapolation as a native capability](FQ-ERROR-MITIGATION-ZNE-20261002.md)
