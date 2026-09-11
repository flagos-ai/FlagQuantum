# Architecture

Start with the repository [architecture map](../../ARCHITECTURE.md) and the
[compiler implementation map](../development/IR_IMPLEMENTATION_STATUS.md).
These describe code ownership and execution paths in this checkout. Support
levels belong to the [capability catalog](../generated/CAPABILITIES.md);
validation scope belongs to [publication validation](../reference/PUBLICATION_VALIDATION.md).

## Implementation and operating contracts

- [Runtime architecture](RUNTIME_ARCHITECTURE.md)
- [Dependency direction and enforcement](ARCHITECTURE_DEPENDENCIES.md)
- [Hybrid runtime architecture](HYBRID_RUNTIME_ARCHITECTURE.md)
- [Hybrid parallel semantics](HYBRID_PARALLELISM.md)
- [Training state, precision, and determinism](TRAINING_STATE.md)
- [Rank-owned MPS forward](RANK_OWNED_MPS_FORWARD.md)

Domain READMEs under `flagquantum/` identify the owning source files and focused
tests. An implemented path is not automatically a release-certified capability.

## Design direction and historical decisions

- [Long-term target architecture](LONG_HORIZON_ARCHITECTURE.md)
- [Multi-level IR design](MULTI_LEVEL_IR_ARCHITECTURE.md)
- [Historical Phase 0–1 execution plan](MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
- [Noise code organization review](NOISY_SIMULATION_CODE_ORGANIZATION.md)
- [Discrete-event simulation design](DES_FRAMEWORK_ARCHITECTURE.md)
- [Architecture decisions](decisions/README.md)

These documents retain design rationale, proposed interfaces, and recorded
approval boundaries. Their original baselines and phase checklists do not
establish what is implemented today; follow the implementation map before
using a proposed module or API.
