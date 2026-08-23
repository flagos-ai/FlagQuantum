# FlagQuantum Roadmap

> **Status:** Future intent
> **Direction:** Build, train, and scale quantum AI through one PyTorch-facing
> programming model and the FlagOS unified multi-chip backend.

This roadmap describes what FlagQuantum plans to deliver next. It is not a
statement of current support. For capabilities available today, see the
[capability catalog](../generated/CAPABILITIES.md) and
[known limitations](../reference/KNOWN_LIMITATIONS.md).

## The destination

FlagQuantum is moving toward one durable quantum AI workflow:

```text
build once
  → train with PyTorch
  → choose statevector, MPS, or tensor-network execution
  → scale across chips when the workload requires it
  → deploy the trained program to quantum hardware
```

Backend choice should not require users to rewrite their model. Planning,
execution, gradients, and deployment should remain visible through stable,
typed interfaces.

## Next: a unified quantum AI framework

The next release focuses on making the current product model coherent and easy
to adopt.

Planned capabilities:

- stable `fq.Circuit`, `fq.Module`, `fq.run`, `fq.train`, and `fq.plan`
  interfaces;
- one versioned FlagQuantum IR shared by execution, compilation, and
  deployment;
- statevector, MPS, and tensor-network execution selected from the same
  program;
- PyTorch-native gradients and optimizer workflows across supported modes;
- explainable runtime planning with explicit blockers and no silent fallback;
- reproducible installation, examples, tests, and migration guidance.

The result should be a reliable local development experience on CPU or one GPU,
with advanced execution modes clearly labeled by maturity.

## Then: the FlagOS unified multi-chip backend

The following stage connects FlagQuantum to a unified FlagOS execution layer
instead of exposing accelerator-specific behavior to users.

Planned capabilities:

- a stable, Torch-FL-backed FlagOS provider integration with no core-package
  hard dependency;
- unified device, stream, memory, kernel, and collective contracts;
- optimized contraction, complex linear algebra, QR, and SVD kernels for MPS
  training;
- consistent execution across NVIDIA and FlagOS-supported accelerators;
- automatic detection of unsupported operations and hidden CPU fallback;
- reproducible cross-accelerator correctness and performance evidence.

Users should be able to move from local development to a supported accelerator
without changing the quantum AI model.

## Scale: train workloads that do not fit on one device

FlagQuantum will extend the same programming model to genuinely sharded
training.

Planned capabilities:

- sharded statevector forward and backward execution;
- rank-owned MPS forward, backward, and optimizer state;
- tensor-network slicing and reduction with differentiable reverse
  contraction;
- topology-aware placement and communication through FlagCX;
- distributed checkpointing, recovery, and failure diagnostics;
- runtime evidence that distinguishes sharded capacity from replicated
  throughput.

This stage is complete only when one logical workload exceeds a single-device
capacity limit and trains successfully across devices. Replicated execution
will not be presented as capacity scaling.

## Deploy: connect training to quantum hardware

The final product loop carries a trained parameterized program from simulation
to a provider or QPU without leaving FlagQuantum.

Planned capabilities:

- bind trained parameters into a versioned deployment artifact;
- compile and validate programs against target-device constraints;
- submit, monitor, and retrieve provider jobs through a common interface;
- preserve shots, calibration data, routing, and result provenance;
- compare ideal, noisy, and hardware results through one result model;
- support hardware-in-the-loop evaluation and future training workflows.

Provider and QPU support will remain target-specific until it has been
validated on real hardware.

## What users can expect

| Stage | User-visible outcome |
| --- | --- |
| Unified framework | One API for circuits, training, planning, and execution |
| FlagOS backend | The same model runs across supported accelerator stacks |
| Distributed training | A single oversized workload trains across chips |
| Quantum deployment | A trained program reaches a QPU with traceable results |

## Release standard

A planned capability moves into supported documentation only when:

- its public interface and compatibility boundary are defined;
- forward execution, gradients, and optimizer behavior pass the relevant
  correctness tests;
- supported hardware paths have reproducible evidence;
- fallback, limitations, and distribution semantics are explicit;
- installation and an end-to-end example are verified.

Until then, the capability remains experimental, development evidence, or
future intent.
