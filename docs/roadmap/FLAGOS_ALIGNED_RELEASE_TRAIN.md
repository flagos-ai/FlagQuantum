# FlagQuantum Roadmap

> **Status:** Future intent
> **Direction:** A general quantum computing framework with shared program,
> simulation, and execution capabilities, PyTorch-native training, and the
> FlagOS unified multi-chip backend.

This roadmap describes what FlagQuantum plans to deliver next. It is not a
statement of current support. For capabilities available today, see the
[capability catalog](../generated/CAPABILITIES.md) and
[known limitations](../reference/KNOWN_LIMITATIONS.md).

## The destination

FlagQuantum is moving toward one durable quantum computing workflow:

```text
express a circuit or physical model
  → declare outputs, constraints, and accuracy requirements
  → plan a supported representation and execution method
  → simulate or execute and inspect results
  → optionally optimize with PyTorch, scale, or deploy
```

Backend choice should not require users to rewrite their model. Planning,
execution, gradients, and deployment should remain visible through stable,
typed interfaces.

Domain-native programs may express physical controls, optical modes, or anyon
fusion and braiding instead of gates. Shared semantics must preserve those
distinctions until a supported lowering is selected. Arbitrary backend switching
is not promised: the planner must reject unsupported operations, state spaces,
outputs, or gradients before numerical execution.

## Shared programming and M0-M4 delivery

This is the delivery sequence for the general-framework scope in the
[vision](FLAGQUANTUM_VISION.md). The existing [architecture](../../ARCHITECTURE.md),
[implementation map](../development/IR_IMPLEMENTATION_STATUS.md), and
[ARCH-012](../architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md)
remain authoritative. No new public API, IR schema, top-level package, numerical
core, or compatibility exception is approved by this roadmap.

The baseline reviewed for M0 is public main commit
`0f389c9d8f30179557eb14920bbc9e9dd49b1bd0` (2026-10-19). Supported behavior
continues to come from `capability-maturity.toml`; the CUDA-Q comparison continues
to come from `contracts/cudaq-parity-matrix.toml`. Update their generated views
with the existing tools. Do not create another capability registry or scoreboard.

### Four independent dimensions

| Dimension | Examples |
| --- | --- |
| Physical system | Superconducting circuits, neutral atoms, trapped ions, optical systems, Majorana systems |
| Computing model | Gate circuits, continuous evolution, measurement feedback, anyon braiding and fusion |
| Numerical representation | Statevector, density matrix, MPS/TN, stabilizer, truncated Fock space, fusion space |
| Execution target | CPU, accelerator, distributed resources, external simulator, real device |

Only evidenced combinations are supported. In particular, local simulation,
device discovery, cloud submission, and a physics-native SDK are separate
capabilities. Topological computation is not synonymous with a topological
error-correcting code or a provider backend.

### Milestones and acceptance

The ordering below is binding for this workstream; durations are planning
estimates for a four-to-six-person team with physics support, not release dates.
Milestones advance by evidence, not elapsed time. M0 is a baseline repair and
documentation change; M1-M4 remain future work until their own checks pass.

| Milestone | Estimate | Owning domains | Deliverable and exit gate |
| --- | --- | --- | --- |
| M0: reliable baseline | 1-2 weeks | Integration; Algorithms for the numerical repair | Reproduce and repair the Python 3.10/3.11 CPU failures without weakening numerical assertions; reconcile capability descriptions and generated views; keep public APIs and architecture unchanged; run supported-version regressions and required repository checks. |
| M1: shared mathematical execution | 4-6 weeks | Core, Operators/Observables, Simulation | Extend existing state-space and dynamics boundaries for mixed dimensions and bounded time-dependent controls; connect the existing boson algebra to execution; two different physical models share a solver and result path, with independent references, ordering, truncation, convergence, and gradient checks. |
| M2: shared developer workflow | 3-4 weeks | Integration, Runtime, Core | Reuse ExecutionOptions, output requests, parameter binding, planning, and PyTorch lifecycle; demonstrate two models using the same workflow without a second trainer or backend registry; explicitly validate result axes, gradient availability, serialization, and exact-plan execution. |
| M3: domain-native complete scenarios | 6-10 weeks | Domain implementation owners under existing boundaries | Deliver the reference scenarios below with input, validation, planning, execution, outputs, diagnostics, and examples; domain additions reuse the common path and carry their own support limits. |
| M4: performance and device validation | Continuous after correctness gates | Simulation, Compute, Runtime, Remote, Twin | Measure acceleration, batching, approximation, calibration, or hardware execution against fixed accuracy requirements; distinguish throughput from capacity and simulation from hardware evidence. |

The M0 baseline's failing Python 3.10/3.11 jobs reproduced two failures in
`tests/unit/test_algorithms_chemistry_integrals.py`: ordinary float summation
lost precision in a contracted overlap and in its independent closed-form
reference. The repair uses `math.fsum` in those accumulations, retains the
existing accuracy and rotation assertions, and adds an analytic cancellation
regression over every primitive ordering. Reproduce the focused acceptance
with `python -m pytest tests/unit/test_algorithms_chemistry_integrals.py -q`
on each supported Python minor; the broader testing tiers still apply. A local
pass does not relabel the original failed CI run as successful.

M1 starts with static mixed-dimension CPU references before time-dependent
controls. Preserve CircuitIR 1.0 and the working qubit fast path. Reuse
`core/ir/`, `compiler/pass_manager.py`, `lindblad/`, and `simulation/lindblad*`;
do not introduce a parallel IR hierarchy, pass manager, or dynamics engine.
The existing boson algebra is a starting point, not a complete Fock backend.
Trainable coefficients and live tensor bindings require explicit validation;
serializing parameter values does not serialize an autograd graph.

An internal state-space proposal must be tested against both tensor-product
spaces and an anyon fusion-space example with fixed total charge. Product
dimensions alone do not describe all constrained spaces. Introduce only the
representation required by a verified consumer; do not prebuild a universal
type system. Domain-only model construction can start in its current owning
domain. A new `physics/` package requires a separate boundary decision and
architecture-policy change justified by multiple consumers; this roadmap
creates neither it nor empty packages for each physical route.

### Reference scenarios

| Scenario | Shared capability under test | Additional acceptance |
| --- | --- | --- |
| Existing variational circuit training | CircuitIR, Module, planning, gradients | Preserve current APIs and the local CPU fast path. |
| Driven three-level superconducting model | Mixed dimensions, controls, dynamics | Validate populations, leakage, and time-grid convergence against an independent reference. |
| Moving neutral-atom pair | Geometry, time dependence, noise and events | Separate atom/trap identity and physical loss/detected loss; report unconditional outcomes, survival, and postselection. |
| Spin coupled to a motional mode | Mixed dimensions and boson operators | Validate basis ordering, truncation convergence, and shared evolution with the superconducting example. |
| Two-mode photonic interference | Fock states, optical operations, counting and loss | Validate interference and number statistics; do not imply Gaussian-backend support from a truncated Fock example. |
| Small anyon braiding and fusion measurement | Constrained spaces, sectors, basis changes, sampling | Check braid relations, unitarity, total charge, and measurement probabilities; Ising braiding alone is not a universal gate-set claim. |

The topological scenario participates in M1 design review even when its full
execution arrives in M3. Majorana parity measurement and conditional feedback
are a subsequent bounded scenario using fermionic and measurement primitives;
they are not evidence of a physical Majorana device or a complete microscopic
model. QEC owns codes and decoders, not all topological-program semantics.

### Work packaging and verification

1. Repair the reproduced CPU regression; retain the independent physics oracle
   and its accuracy bound. Run the focused tests on Python 3.10, 3.11, and 3.12.
2. Reconcile current implementation, capability fields, parity gaps, and their
   generated views without promoting maturity or claiming unmeasured support.
3. Review a narrow state-space and parameter-binding contract change before
   dependent implementations. Include migration/refusal behavior and a
   contract fake or replacement test under the existing boundary.
4. Deliver static mixed-dimension evolution, then the boson consumer, then
   time-dependent controls. Each change extends a complete CPU path.
5. Prove reuse with two physical models before expanding domain packages.
6. Add neutral-atom motion/loss, then the remaining bounded scenarios, each with
   a local ten-minute example and independent scientific checks.

Every implementation task names its user outcome, owning domain, existing
components reused, unsupported scope, and reproducible validation. Integration
owns shared/protected contract changes; the owning domain implements after that
baseline lands. Parallel work must not share an unproven contract.

Use the existing [testing tiers](../development/TESTING.md), architecture and
dependency checks, public-API protection, capability checker, and generated-doc
checks. Report solver error, Hilbert-space truncation, statistical uncertainty,
and gradient error separately. No single global tolerance establishes them all.
Performance claims require matched accuracy and environment evidence.

The FlagOS, distributed-training, and deployment sections below remain existing
workstreams. They do not require each new physical model to invent its own
execution or resource-management layer.

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
