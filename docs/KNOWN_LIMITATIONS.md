# Known limitations

FlagQuantum publishes support claims per capability, not for the package as a
whole. The authoritative status and evidence requirements are defined in
[`capability-maturity.toml`](../capability-maturity.toml) and explained in
[`CAPABILITY_MATURITY.md`](CAPABILITY_MATURITY.md).

| Capability | Current level | Limitation |
| --- | --- | --- |
| IR v1 | `release_certified` | Incompatible schema changes require an explicit migration. |
| Local statevector | `production_supported` | Capacity is bounded by one device; larger statevectors require the sharded runtime. |
| Sharded statevector training | `production_supported` | Multi-node release certification requires promoted, audited hardware evidence. |
| Sharded MPS training | `development_evidence` | Single-node 2/4/8-GPU forward and boundary transport plus a 100-step 8-GPU SGD soak are validated; other optimizers, multi-node operation, and release payload evidence remain incomplete. |
| Tensor-network training | `experimental` | General reverse contraction and production distributed transport are not certified. |
| Cloud deployment | `development_evidence` | Provider behavior and credentials vary; no provider is release certified. |
| Dynamic circuits | `experimental` | Provider-neutral active-reset, conditional-flip, and qubit-reuse conformance vectors pass on the FlagQuantum trajectory runtime and Qiskit Aer. Amazon Braket IQM `measure_ff`/`cc_prx` serialization, real SDK `Program` construction, dry-run checks, and mocked submission are tested, but no AWS account or real QPU task was used. Device availability, published dynamic groups, credentials, billing, and hardware results remain unverified; persistent-layout/ancilla routing and MPS, TN, distributed, or differentiable execution remain unsupported. |
| Extension SDK | `experimental` | Compatibility is not guaranteed before stabilization. |

These statements are support boundaries, not a development backlog. A
capability is promoted only when all evidence required by its target level
passes `python tools/check_capability_maturity.py`.
