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
- [One word for a qubit, across the whole package](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md)
- [The other two doors: attribute names and definition names](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md)
- [The runtime slice of the qubit vocabulary migration](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md)
- [The runtime executor slice of the qubit vocabulary migration](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md)
- [The algorithms slice of the qubit vocabulary migration](FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md)
- [The drawer reads a legacy device's spelling without publishing it](FQ-DRAWER-LEGACY-QDEV-20261012.md)
- [The Twin region composer relabels through the one qubit-relabelling rule](FQ-QUBIT-VOCABULARY-TWIN-REGION-20261018.md)
- [The primitives package admits an export on three dimensions, and one export was deleted](FQ-ALGORITHMS-PRIMITIVES-ADMISSION-20261019.md)
- [The primitive append forms are placements, and placement is `Circuit.compose`](FQ-ALGORITHMS-PRIMITIVES-PLACEMENT-20261023.md)
- [Azure Quantum remote run contract](FQ-AZURE-REMOTE-RUN-20260922.md)
- [CPU noisy-MPS counts through `fq.run`](FQ-CPU-NOISY-MPS-COUNTS-20260924.md)
- [Channel instruction parameters on the public `fq.Circuit` surface](FQ-CHANNEL-INSTRUCTION-PARAMETERS-20261002.md)
- [Circuit expressiveness contract](FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md)
- [Circuit composition](FQ-CIRCUIT-COMPOSITION-20261002.md)
- [Circuit adjoint](FQ-CIRCUIT-ADJOINT-20261003.md)
- [Circuit control](FQ-CIRCUIT-CONTROL-20261022.md)
- [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md)
- [The batch parameter-shift profile reads the opcode declaration](FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md)
- [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md)
- [The density-matrix output](FQ-DENSITY-MATRIX-OUTPUT-20261006.md)
- [Compiler boundary responsibility split](FQ-COMPILER-BOUNDARY-SPLIT-20260930.md)
- [Pauli algebra and quantum_info ownership boundary](FQ-PAULI-QUANTUM-INFO-BOUNDARY-20260930.md)
- [Quafu credential presence fails closed before submission](FQ-QUAFU-CREDENTIAL-FAIL-CLOSED-20260930.md)
- [Quafu service compilation through `fq.run`](FQ-QUAFU-SERVICE-COMPILATION.md)
- [OpenQASM import subset and `fq.from_openqasm`](FQ-OPENQASM-IMPORT-20261002.md)
- [Remote job lifecycle](FQ-REMOTE-JOBS.md)
- [Zero-noise extrapolation as a native capability](FQ-ERROR-MITIGATION-ZNE-20261002.md)
- [A native SPSA optimizer for objectives with no gradient](FQ-SPSA-OPTIMIZER-20261002.md)
- [A global phase on the program](FQ-IR-GLOBAL-PHASE-20261006.md)
- [The optimization-level parameter on the compiler entry points](FQ-COMPILER-OPTIMIZATION-LEVEL-20261006.md)
- [Core-lane dependency discipline for optional imports and `tomllib`](FQ-CORE-LANE-DEPENDENCIES-20261025.md)
