<!-- BEGIN GENERATED KNOWN_LIMITATIONS -->
# Known limitations

Do not edit this generated region. Capability boundaries come from
[`capability-maturity.toml`](../../capability-maturity.toml).

| Capability | Current level | Limitation |
| --- | --- | --- |
| Unified circuit API and FlagQuantum IR | `release_certified` | IR v1; incompatible schema changes require an explicit migration. |
| Local statevector simulation and training | `production_supported` | Capacity is bounded by one device; distributed capacity claims use the sharded capability. |
| Sharded statevector training | `production_supported` | Multi-node release certification remains dependent on promoted audited hardware evidence. |
| FlagOS local statevector CUDA reference | `development_evidence` | CUDA-backed development reference only. It does not certify a domestic accelerator, prove absence of Torch-FL host fallback, establish production performance, or authorize a scalability claim. |
| Differentiable and sharded MPS training | `development_evidence` | Single-node and dual-node execution plus matched checkpoint/restart have development evidence. The only public capacity measurement is emitted from the validated claim below; it is one exact-workload result, not general scalability or release evidence. Boundary instructions still execute serially by owner, and layer-parallel contraction/SVD, capacity multi-step soak, a sealed fault matrix, repeated evidence, and the release payload remain incomplete. |
| Constrained local MPS TEBD | `experimental` | Static real one-site and adjacent two-site Pauli terms on an open chain, batch one, second-order imaginary-time evolution, and product initial states only. Real-time evolution, periodic and nonlocal terms, gradients, TDVP, distributed execution, and production or scalability claims are unsupported. |
| Tensor-network execution and training | `experimental` | General reverse contraction and production distributed transport are not certified. |
| Exact and trajectory-based noisy simulation | `experimental` | Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-wire MPS channels use an explicitly dense correctness fallback. |
| Circuit packaging and cloud deployment | `development_evidence` | Provider support and credential/runtime behavior vary; no provider is release-certified by this matrix. |
| Qiskit IR interoperability | `experimental` | Validated with Qiskit 2.5.2 and Aer 0.17.2. Qiskit control flow and arbitrary ParameterExpression import are rejected; named or multiple registers require explicit lossy flattening; custom multi-qubit unitary matrices remain blocked until basis ordering is specified. Conversion does not make Qiskit a runtime dependency or certify any provider hardware. |
| Dynamic circuits and IQM Braket preflight | `experimental` | Provider-neutral conformance vectors pass on the FlagQuantum trajectory runtime and Qiskit Aer. IQM dialect serialization, SDK Program construction, sealed packaging, and mocked provider submission are tested. No AWS account or real IQM QPU task was used, so device availability, published qubit groups, billing, credentials, and hardware results remain unverified. |
| Extension SDK | `experimental` | Extension compatibility is not guaranteed before stabilization. |

## Validated public performance claims

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `efb0d34c741d1196a7f7c89404cbf033c655f14bf5d575470bbe605bb93b0cbc`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |

These statements are support boundaries, not a development backlog. Promotion
requires all evidence for the target level to pass
`python tools/check_capability_maturity.py`.
<!-- END GENERATED KNOWN_LIMITATIONS -->
