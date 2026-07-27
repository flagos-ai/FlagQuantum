# FlagQuantum Ecosystem Development Plan

This document defines the ecosystem layer around FlagQuantum: learning content,
examples, demos, integrations, benchmarks, contribution paths, and release
rhythm. The goal is to make FlagQuantum feel like a usable quantum AI product,
not only a runtime library.

> **Strategy and intent, not capability evidence.** This document describes how
> the ecosystem should be built. An integration, benchmark, partnership, or
> certification is not available until its implementation and required evidence
> exist in the repository.

## Ecosystem Thesis

FlagQuantum should build an ecosystem around a narrow product identity:

> **PyTorch-native infrastructure for developing, scaling, validating, and
> deploying quantum AI workloads.**

The ecosystem should not compete on the raw number of circuit APIs or provider
logos. It should make one workflow progressively more valuable:

```text
scientific problem
        |
        v
model and dataset -> FlagQuantum training -> runtime or hardware integration
        ^                                      |
        |                                      v
reusable package <- reproducible evidence <- measured result
```

The desired long-term user perception is **the distributed training
infrastructure for quantum AI**, rather than another general-purpose circuit
toolbox or an isolated high-performance simulator.

## Target Participants

The initial ecosystem should optimize for a small number of deeply engaged
participants instead of broad but shallow awareness.

| Participant | Primary need | FlagQuantum offer |
| --- | --- | --- |
| Quantum-AI researchers | Differentiable models and reproducible experiments | PyTorch workflows, reference tasks, evidence artifacts |
| HPC and multi-GPU teams | Workloads beyond one accelerator | Honest sharding semantics, profiling, recovery, capacity tests |
| Backend and hardware teams | Real workloads and a stable integration boundary | Versioned extension SDK, conformance, certification |
| Model and algorithm authors | Distribution and reuse of research outputs | Model packages, task templates, paper reproductions |
| Cloud and QPU providers | Train-to-hardware workflows | Provider contract, capability discovery, provenance |

Education and broad introductory quantum programming remain useful adoption
paths, but they should not displace the core quantum-AI and distributed-systems
audience during the beta stage.

## Ecosystem Principles

1. **Depth before breadth.** One complete provider or accelerator integration is
   more valuable than many logos attached to partial adapters.
2. **Stable boundaries before a marketplace.** The SDK, compatibility policy,
   conformance suite, and ownership model must mature before promoting a large
   plugin catalog.
3. **Tasks before feature showcases.** Ecosystem assets should solve a user or
   scientific problem, not merely demonstrate an API call.
4. **Evidence before promotion.** Compatibility, performance, and scalability
   labels require executable checks and measured artifacts.
5. **External ownership by default.** Optional integrations and model families
   should live in independently maintained packages once their contracts are
   stable, rather than continuously expanding the core package.
6. **Local success remains the entry point.** Every distributed or provider
   journey should begin with a bounded local workflow where possible.

## Four Ecosystem Layers

### Developer And Extension Ecosystem

The extension SDK should become the common boundary for independently released
packages. Recommended package families are:

| Package family | Purpose | Example naming |
| --- | --- | --- |
| Execution backend | Simulator or optimized runtime | `flagquantum-backend-*` |
| Provider | Cloud or QPU access | `flagquantum-provider-*` |
| Operators and kernels | Hardware-optimized primitives | `flagquantum-ops-*` |
| Compiler integration | Lowering, mapping, or optimization passes | `flagquantum-compiler-*` |
| Models | Reusable quantum-AI architectures | `flagquantum-models-*` |
| Datasets and tasks | Reproducible scientific workloads | `flagquantum-datasets-*` |

Each family should eventually provide:

- a generated project scaffold and a ten-minute reference implementation;
- versioned manifests and capability discovery;
- reusable conformance tests and compatibility matrices;
- a named maintainer and support window;
- installation, security, failure, and cleanup documentation;
- release automation independent of the core package.

An installed extension must report more than availability. Where applicable it
must declare gradient, dtype, device, distribution, checkpoint, fallback, and
deployment semantics before activation.

### Runtime, Hardware, And Provider Ecosystem

The first partnership portfolio should be deliberately small:

1. one deeply optimized NVIDIA/PyTorch/NCCL reference path;
2. one non-NVIDIA or domestic accelerator path that tests portability;
3. one real QPU provider path covering discovery, compilation, submission,
   polling, retry, normalized results, and provenance.

Integration status should use three distinct labels:

| Label | Meaning |
| --- | --- |
| Compatible | The maintainer reports that the basic public contract passes |
| FlagQuantum Verified | The integration passes the official compatibility and conformance suite on a declared environment |
| Production Certified | Correctness, performance or capacity where claimed, recovery, security, and support gates pass with auditable evidence |

Allocation, import success, or a smoke run alone must never qualify an
integration as production certified.

### Research And Model Ecosystem

FlagQuantum should maintain a small flagship task suite that creates reusable
models, benchmark evidence, and publication opportunities. Initial task
families should include:

- distributed VQE and variational energy estimation;
- quantum classification and hybrid classical-quantum models;
- MPS-based many-body or Hamiltonian identification;
- quantum dynamics and ground-state search;
- structured workloads that compare statevector, MPS, and tensor-network
  execution under an explicit accuracy budget.

Every flagship task should include a problem statement, scientific relevance,
small local example, one-device baseline, distributed or capacity variant when
supported, environment manifest, correctness tolerance, result artifact, and
citation guidance.

Candidate community programs include a FlagQuantum Research Challenge,
student research fellowships, compute grants, joint publications, and awards
for models, integrations, and measured optimizations. These programs should
start only when the corresponding task and result-reproduction workflow is
stable.

### Learning And Adoption Ecosystem

Content should follow user jobs rather than repository modules. The primary
journeys are:

- train a first parameterized quantum model;
- move a workload from CPU to one GPU;
- diagnose a model that does not fit on one GPU;
- select statevector, MPS, or tensor-network execution;
- integrate a backend or provider;
- reproduce a FlagQuantum research result;
- bind trained parameters and deploy to a QPU.

Each journey should offer a short quick start, a workshop-length guide, and a
reusable project template where justified. Models and examples that become
substantial should move to separately versioned packages rather than turn the
core repository into an application monorepo.

## Ecosystem Flywheels

| Flywheel | Loop | Product effect |
| --- | --- | --- |
| Model | Research model -> reusable package -> users and results -> new contributors | More valuable workloads |
| Backend | User demand -> partner integration -> verified workload -> optimization | Better hardware reach and performance |
| Evidence | Standard task -> measured artifact -> independent reproduction -> trusted claim | Stronger credibility and partnerships |

The evidence flywheel is the primary differentiation. A public result catalog
should make it easy to distinguish local performance, replicated throughput,
true sharded speedup, and capacity expansion.

## Positioning

FlagQuantum should be presented as a quantum AI framework with one coherent
programming model across:

- Local statevector simulation for exact small and medium circuits.
- Local MPS simulation for structured low-entanglement workloads.
- Local tensor-network simulation for contraction-friendly circuits.
- PyTorch-first training loops.
- JAX quantum kernels as an optional acceleration path.
- FlagQuantum IR as the bridge to compilation, visualization, cloud packaging,
  and future distributed runtimes.

The ecosystem should keep a clear boundary between local performance evidence
and distributed scalability. Local MPS/TN examples can show capacity and speed
for structured workloads on one CPU/GPU. Distributed scalability claims require
one logical workload sharded across ranks and explicit runtime evidence.

## Ecosystem Pillars

| Pillar | Purpose | Primary artifacts |
| --- | --- | --- |
| Learning | Help new users build correct mental models | Tutorials, concept guides, API recipes |
| Runnable workflows | Give developers copyable starting points | Examples, smoke commands, templates |
| Research demos | Show algorithmic and application value | Gallery demos, paper reproductions |
| Integrations | Make FlagQuantum fit existing AI stacks | PyTorch layers, JAX kernels, deployment adapters |
| Evidence | Make performance and correctness claims citable | Benchmarks, result JSON, reports, audits |
| Contribution | Help external users add value safely | Contributor guide, example templates, issue labels |

## Learning Ecosystem

Learning content should be organized by user intent instead of source module.

| Track | User question | Content to build |
| --- | --- | --- |
| Getting started | How do I build and measure my first circuit? | Install, first `fq.Circuit`, states, gates, measurements |
| Differentiable circuits | How do I train parameters? | PyTorch gradients, `fq.Module`, runtime policies |
| Runtime choice | Should I use statevector, MPS, or TN? | Decision guide, memory estimates, examples by circuit structure |
| Algorithms | How do I implement VQE/QAOA/kernel/QNN workflows? | VQE, QAOA, quantum classifier, kernel methods |
| Large structured simulation | How do I use MPS/TN responsibly? | Bond dimension, truncation, contraction path, structured 1000q demo |
| Deployment | How do I move from training to cloud/hardware? | IR export, QASM/QCIS, provider package, mock provider |

Each tutorial should include a matching runnable example, a short "when to use
this" note, and a "what this does not prove" note for performance-sensitive
topics.

## Examples Ecosystem

Examples should be small, tested, and practical. They should avoid becoming
mini-frameworks.

### Curated Example Families

| Family | Example ideas | Notes |
| --- | --- | --- |
| Basic circuits | Bell state, GHZ, measurement, custom gates | CPU-only, under 10 seconds |
| Algorithms | VQE, QAOA, Hamiltonian expectation, parameter binding | Use `fq.Circuit` and `Hamiltonian` |
| QML | Quantum classifier, kernel classifier, Torch layer | PyTorch-first, JAX optional |
| MPS | Low-bond ansatz, dimerized chain, scale report | Report bond dimension and truncation |
| Tensor network | Circuit-to-TN, contraction strategy comparison, slicing | Report contraction profile and memory target |
| Noise | Density matrix, depolarizing channel, amplitude damping | Keep small and exact |
| Deployment | Train, bind parameters, export, package for provider | Use mock/local provider by default |

### Example Quality Bar

Every curated example should:

- Run from the repository root with one command.
- Accept `--steps`, `--n-qubits`, `--device`, and relevant runtime options.
- Print structured final metrics.
- Have a smoke-test command suitable for CI.
- Avoid hidden downloads unless clearly documented.
- State whether execution is local, replicated, sliced, or sharded.

## Demo And Gallery Ecosystem

Demos can be longer and more narrative than examples. They should be grouped by
why a user would care.

| Gallery | Purpose | Candidate demos |
| --- | --- | --- |
| Algorithm gallery | Show canonical quantum algorithms | VQE H2/Ising, QAOA MaxCut, time evolution |
| QML gallery | Show AI training loops | Classifier, quantum kernel, hybrid Torch model |
| Tensor-network gallery | Show structure-aware simulation | Low-bond MPS, TN contraction comparison |
| Deployment gallery | Show full workflow closure | Train-to-QASM/QCIS/provider package |
| Performance stories | Explain measured wins and limits | Single-device MPS, JAX kernel speedups |

Performance stories must include environment, device, dtype, runtime mode, and
claim boundaries. For example, a structured 1000-qubit MPS demo should say that
it is favorable to low-bond MPS and is not evidence for arbitrary high-
entanglement circuits.

## Integration Ecosystem

FlagQuantum should meet users where they already work.

| Integration | User value | Ecosystem artifact |
| --- | --- | --- |
| PyTorch | Standard AI training loops | `fq.Module` tutorials and examples |
| JAX | Fast quantum kernels | `compile_quantum_kernel` guides and fallback notes |
| OpenQASM/QCIS | Hardware and cloud portability | Export tutorials and deployment examples |
| Provider adapters | Real or mock cloud execution | Provider matrix, credential-free mock demos |
| Visualization | Explain circuits and IR | Drawer examples and gallery screenshots |
| Benchmarks | Compare runtime choices | Reproducible commands and result reports |

Optional integrations should degrade gracefully. If JAX, a provider SDK, or a
visualization package is missing, examples should skip or explain the fallback
instead of failing opaquely.

## Evidence And Benchmark Ecosystem

Benchmarks should be separated from tutorials and examples. Tutorials teach;
examples template workflows; benchmarks preserve evidence.

Evidence artifacts should include:

- Command line used to produce the result.
- Hardware, OS, Python, PyTorch, JAX, and backend versions.
- Runtime mode: statevector, MPS, tensor network, sliced TN, sharded statevector,
  or sharded MPS/TN.
- Device and dtype.
- Memory, time, gradient, and correctness metrics where applicable.
- Explicit claim wording and limitations.

Recommended benchmark result groups:

| Group | Purpose |
| --- | --- |
| `benchmarks/results/local/` | Citable single-machine statevector/MPS/TN results |
| `benchmarks/results/comparison/` | PennyLane/TensorCircuit/etc. comparisons |
| `benchmarks/results/distributed/` | Sharded distributed evidence only |
| `benchmarks/results/historical/` | Engineering logs not suitable for claims |

## Contribution Ecosystem

Contributors should have a low-friction path to add tutorials and examples
without weakening quality.

### Contributor Templates

| Template | Should include |
| --- | --- |
| Tutorial template | Objective, prerequisites, minimal cell, explanation, runtime notes, next links |
| Example template | Docstring, argparse, deterministic seed, structured summary, smoke command |
| Demo template | Narrative, dataset/source notes, expected runtime, screenshots or results |
| Benchmark template | Environment capture, JSON output, claim boundary, audit command |

### Review Checklist

- Does this content teach a real user task?
- Does it run locally without hidden services?
- Is the expected runtime clear?
- Are optional dependencies handled?
- Are performance claims backed by benchmark evidence?
- Are local, replicated, sliced, and sharded execution described accurately?
- Is there a smoke test or documented manual verification command?

## Release Rhythm

| Cadence | Output |
| --- | --- |
| Every PR | Keep examples and notebook JSON health checks passing |
| Every minor release | Refresh curated examples and README entry points |
| Every benchmark refresh | Archive command, JSON, report, and claim wording |
| Every feature milestone | Add one tutorial, one runnable example, and one smoke test |
| Every ecosystem review | Move mature demos into gallery and archive stale experiments |

## Phased Execution And Success Metrics

Dates should be set at execution time; the phases below are readiness-based so
they do not imply commitments unsupported by staffing or partner access.

### Stage E0: Foundation

Exit conditions:

- the stable public API and extension boundary pass their repository gates;
- backend and provider reference extensions are documented and tested;
- a scaffold, conformance command, and compatibility policy exist;
- three flagship research tasks have local reproducible baselines;
- a new user can run a curated example in 30 minutes;
- an experienced external developer can implement a minimal extension in one
  working day.

### Stage E1: Seed Ecosystem

Exit conditions:

- 5-10 recurring external contributors or research users are active;
- 2-3 university or research collaborations run maintained workloads;
- one real provider integration completes an end-to-end non-secret test;
- one additional accelerator ecosystem completes a declared compatibility run;
- the first public benchmark collection is independently reproducible;
- beta support boundaries and the compatibility matrix are published.

### Stage E2: Recognized Ecosystem

Exit conditions:

- a verified integration catalog has multiple independently maintained owners;
- at least one flagship scientific result and technical report are published;
- an annual research or optimization challenge has a reproducible task suite;
- model, backend, and evidence flywheels each produce repeat contributors;
- ecosystem releases follow the compatibility and deprecation policy without
  requiring uncontrolled root-API growth.

The leading indicators should be retained-user and contribution quality, not
only downloads, stars, chat membership, or plugin count. Track:

- successful first-run rate and time to first result;
- monthly recurring research users and external maintainers;
- externally owned packages passing conformance;
- reproducible flagship tasks and independent result reproductions;
- median issue/PR response and review time;
- compatibility failures by release and supported-version coverage;
- partner workloads reaching verified or certified status.

## Governance

Before opening a broad extension catalog, define:

- core maintainers, component owners, and contributor progression;
- an RFC process for public contracts and architecture changes;
- compatibility windows, deprecation periods, and support branches;
- security reporting and credential-handling requirements;
- benchmark submission, reproduction, audit, and withdrawal rules;
- naming, trademark, and `FlagQuantum Verified/Certified` usage policy;
- ownership transfer and archival rules for abandoned integrations;
- coexistence rules for open-source, research, and commercial extensions.

No extension should be promoted without an accountable maintainer. Stale or
incompatible integrations should be visibly archived rather than silently left
in the supported catalog.

## Deliberate Non-Goals For The Current Stage

- building a large plugin marketplace before stable compatibility contracts;
- maximizing the number of partially implemented cloud providers;
- treating every internal benchmark script as a supported product surface;
- measuring ecosystem health primarily through repository stars or group size;
- accepting unowned integrations without conformance and maintenance policy;
- simultaneously targeting introductory education, every quantum workflow,
  enterprise applications, and leadership-scale HPC with equal priority;
- branding FlagQuantum as the "PyTorch of quantum computing" before adoption,
  compatibility, and production evidence justify the comparison.

The near-term outcome should be concrete: ten genuine research users, three
recurring external contributors, two hardware or provider partners, and one
flagship reproducible result are more valuable than one hundred unmaintained
plugins.

## Near-Term Backlog

1. Clean rendered text and outputs in the existing tutorial notebooks.
2. Add QAOA, tensor-network contraction, noise, and train-to-deploy examples.
3. Create tutorial templates and example templates.
4. Add a benchmark results README that labels citable, comparison, distributed,
   and historical files.
5. Convert the strongest single-machine examples into narrative tutorials.
6. Create a gallery index for application demos and performance stories.
