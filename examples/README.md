# FlagQuantum Examples

[Coherent predicate synthesis](synthesis_predicate.py): run
`python -m examples.synthesis_predicate` for named basis and uniform results of
`x*x+y<5`, using the maintained synth facade without gate design or bit decoding.

Private constrained-synthesis contributor workflows:

- [Candidate selection](synthesis_selection.py): run
  `python -m examples.synthesis_selection` to select both schedules through
  different final constraints, then check and execute their real Runtime plans.
- [Complete static costs](synthesis_selection_costs.py): run
  `python -m examples.synthesis_selection_costs` for both candidates of four
  computation families at two widths and two targets, plus the manual baseline.
  Larger rows are Compiler costs, not dense-state or hardware measurements.

[`synthesis_execution_contract.py`](synthesis_execution_contract.py) demonstrates
the private CS06-A seam: `python -m examples.synthesis_execution_contract` rejects
planning-induced basis drift, checks an actual routed plan and verifies its
nonidentity output placement and complex phase. This expert plan example leaves
the normal synthesis facade unchanged.

[`synthesis_workspace.py`](synthesis_workspace.py) is the private CS-05 Compiler
diagnostic: `python -m examples.synthesis_workspace` compares retained and
cleanup/recompute workspace costs for one high-level source program and executes
both small coherent blocks on the existing CPU path. This is an advanced explicit
port example; the normal named CS-04 workflow is unchanged. It does not expose
CS-06 constraint-based candidate selection.

Runnable, copy-ready workflows for building and training quantum AI programs
with the current FlagQuantum API.

Examples driven by the root-level `fq` alias use:

- `import flagquantum as fq` as the public entry point;
- `fq.Circuit(n_qubits=...)` for circuit construction;
- PyTorch for parameters, gradients, and optimizers;
- explicit runtime and distribution semantics when making performance claims.

These examples do not use that alias:

- `qec/matched_primary.py`: guarded fixed primary capture and externally hash-bound replay/32-comparison summaries; reuses shared prediction shards and existing statistics; no trusted controller or formal execution CLI.
- `qec/matched_parity.py`: verifies frozen decisions before a single shared probe and composes explicit CNN/MLP parity paths; hardware-only receipt guards and two-host verification; no primary sampler or CLI.
- `qec/matched_validation.py`: internal shared physical-validation capture and all sixty threshold trials with explicit architecture identities; no primary sampler or launch CLI.
- `qec/matched_selection.py`: internal final-six temperature fitting on shared, saved independent calibration streams; no physical validation, CLI or formal launch authorization.
- `qec/matched_training.py`: internal fixed-budget paired Trainer composition with observed batch hashes and shared stored held-out scoring and a complete-artifact gate returning all six final models; no CLI or formal launch authorization.
- `qec/matched_initialization.py`: freezes and verifies all six seed/architecture initial states and Torch RNG without samples or optimizer updates.
- `qec/matched_checkpoints.py`: binds unchanged private Trainer checkpoints to the verified layout, both prospective documents and an externally frozen inventory; no standalone study launch.
- `qec/ising_layout.py`: verifies pinned Ising measurement-record geometry and the private CNN input mapping without sampling, training or loading weights.

- `qec/run_matched_assessment.py`: inspected-container zero-sample preflight and internal fixed-deadline prepare/peer/finish/run composition; trusted SSH orchestration and launch CLI remain pending.
- `qec/run_continued_assessment.py`: controller-bound, fixed-budget A800 study coordination with preflight and immutable handoffs; see the frozen physical protocol.

- `qec/continued_primary.py`: guarded fixed primary prediction shards and registered offline summaries; no formal execution CLI.

- `qec/continued_parity.py`: internal frozen-probe and two-host bit-parity prerequisites; no primary sampler or CLI.

- `qec/continued_selection.py`: internal calibration and physical-validation decision stages for the registered continued-checkpoint study; no sampling CLI.

- `qec/continued_assessment.py`: verifies six frozen QEC checkpoint inputs; the CLI does not expose physical sampling.

- `qec/study_training_budget.py`: preregistered CPU continuation of audited QEC checkpoints; see `docs/development/QEC_AI_TRAINING_SUFFICIENCY.md`.

- [`qec/diagnose_predecoder.py`](qec/diagnose_predecoder.py) inspects archived
  CPU QEC checkpoints with exact H/L equivalence and MLP feature-label diagnostics.
  `--fixture-smoke` runs an exact cycle probe with Core Torch alone. Archived
  runs verify the previous study's inputs and producing environment before new
  fixed-budget samples; they never retrain or access primary logical tests.

- [`qec/compare_ising.py`](qec/compare_ising.py) connects the private QEC reference
  to a pinned optional Ising checkout and a shared batched global decoder. Smoke
  uses untrained weights and does not establish a pretrained-model ranking.

- [`qec/assess_predecoder.py`](qec/assess_predecoder.py) runs the private paired
  QEC study through domain imports.

- [`qec/calibrate_predecoder.py`](qec/calibrate_predecoder.py) diagnoses label/loss
  sparsity and fits confidence calibration with separate validation and test
  streams; it uses private QEC domain imports.

- [`qec/correlated_circuit_noise.py`](qec/correlated_circuit_noise.py) uses the
  public QEC and Noise namespaces to build exclusive two-qubit Pauli faults,
  sample their declared detector model and demonstrate independent-only export
  refusal. Core Torch is sufficient; circuit execution separately needs Stim.

- [`qec/ai_predecoder.py`](qec/ai_predecoder.py) trains and evaluates the private
  QEC predecoder through domain imports; it does not use the root-level alias.

- [`visualize_rydberg_transport.py`](visualize_rydberg_transport.py) uses Core
  commanded positions and the existing optional Matplotlib visualization extra.
- [`terminal_instrument_sampling.py`](terminal_instrument_sampling.py) samples
  explicit presence and finite-Fock instruments through the CPU Runtime API.
- [`terminal_instrument_reference.py`](terminal_instrument_reference.py) uses
  existing SuperOperator maps and illustrative Torch draws for proposal 083.
- [`absorptive_photon_counting.py`](absorptive_photon_counting.py) uses existing
  Lindblad evolution, square vacuum-reset Kraus branches and terminal sampling
  for a finite absorptive counting reference. Run
  `python -m examples.absorptive_photon_counting`; it retains measured-mode vacuum
  coordinates and makes no detector-event or hardware claim.
- [`boson_interference.py`](boson_interference.py) — it uses the existing
  `flagquantum.lindblad` namespace and boson algebra for two-photon interference
  and loss with explicit mode truncations. Run `python -m examples.boson_interference`.
  Unconditional probabilities, survival, plan replay and explicit QND number
  samples share the existing dynamics and Runtime paths; this is a bounded CPU model.
- [`superoperator_channel_algebra.py`](superoperator_channel_algebra.py) — it
  reads one amplitude-damping channel as a superoperator through
  `flagquantum.operators.SuperOperator`: the action, the dense map, the Choi
  matrix, the complete-positivity certificate, the two normalization registers,
  the Pauli transfer matrix, the local application to one qubit of a larger
  register, and the Kraus list read back off the map. Run
  `python -m examples.superoperator_channel_algebra`. It builds no circuit and
  runs no evolution; the algebra is a numerical primitive.
- [`kaiwu_matrix_boundary.py`](kaiwu_matrix_boundary.py) — it imports the
  experimental Kaiwu ecosystem surface directly to demonstrate local QUBO and
  Ising interoperability without credentials, provider submission, or quota.
- [`qdiffusion_kaiwu/minimal_live_sampling.py`](qdiffusion_kaiwu/minimal_live_sampling.py)
  — the shortest live `FlagQuantum -> Kaiwu -> QBoson` path. It requires an
  explicit ten-credit acknowledgement, starts from a FlagQuantum
  `QuboProblem`, and uses the provider-specific ecosystem and remote
  boundaries without exposing a matrix or enabling a local fallback.
  The same reviewed workflow also contains the following directly imported
  provider, evidence, and validation modules: `a800_sampler_smoke.py`,
  `assemble_acceptance.py`, `audit_readiness.py`, `build_environment_lock.py`,
  `build_transfer_bundle.py`, `failure_evidence.py`, `inspect_sdk_wheel.py`,
  `offline_guard/sitecustomize.py`, `plan_quota.py`,
  `preflight_protein_artifacts.py`, `private_io.py`, `provider_inputs.py`,
  `provider_reconciliation.py`, `provider_resources.py`,
  `qboson_live_smoke.py`, `qboson_optimization_probe.py`, `qboson_resume.py`,
  `qboson_sampling_probe.py`, `qdiffusion_portability_replay_live.py`,
  `qdiffusion_protein_evaluate.py`, `qdiffusion_protein_training_live.py`,
  `qdiffusion_system_development_probe.py`, `qdiffusion_system_live.py`,
  `sdk_approval.py`, `source_preflight.py`, `stable_source_tree.py`,
  `stream_development_evidence.py`, `strict_json.py`,
  `validate_acceptance.py`, `verify_environment_lock.py`,
  `verify_extracted_bundle.py`, and `verify_transfer_bundle.py`.
- [`algorithms/`](algorithms/README.md) — `pca.py`, `kmedians.py`,
  `quantum_kernel.py`, `feature_selection.py`, `qarm.py`, `svd.py`,
  `error_mitigation.py`, `pec.py`, `cdr.py`, `readout_mitigation.py`,
  `folding.py`, `variational_solvers.py`, `vqe_solvers.py`,
  `trotter.py`, `block_encoding.py`
  and `linear_combination.py`, which
  import the unit they demonstrate from the subpackage surface because
  `flagquantum.algorithms.<unit>` carries no root-level `fq.` name.
  [`spsa_optimizer.py`](algorithms/spsa_optimizer.py) is the exception inside
  that directory: it imports its optimizer from the subpackage surface and also
  `import flagquantum as fq`, because the objective it minimizes is a circuit it
  has to build and run. [`docs/guides/ALGORITHMS.md`](../docs/guides/ALGORITHMS.md)
  is the per-unit reference they follow, and the place each unit's advantage
  premise is recorded in full.
- [`extensions/reference_extensions.py`](extensions/reference_extensions.py) and
  [`extensions/reference_compiler_extension.py`](extensions/reference_compiler_extension.py)
  — they import the extension and ecosystem APIs, and the second also imports
  `CircuitIR` from the root rather than through the alias.
- [`extensions/reference_backend_extension.py`](extensions/reference_backend_extension.py)
  — a third-party execution backend that imports only
  `flagquantum.ecosystem.extensions`, so it demonstrates the published extension
  surface rather than the root alias.
- [`extensions/reference_target_extension.py`](extensions/reference_target_extension.py)
  — a third-party target declaration that imports the published extension
  namespace and the Core capability vocabulary that describes evidence. It builds
  no circuit, so it needs no root alias either.
- [`extensions/reference_pass_extension.py`](extensions/reference_pass_extension.py)
  — a third-party compiler pass. It imports `CircuitIR` and `Instruction` from
  `flagquantum.core.ir` and its manifest and request types from
  `flagquantum.ecosystem.extensions`, which is what a pass package outside this
  repository can reach; the root alias would only give it a second name for the
  circuit type it is handed.
  [`extensions/declared_target_execution.py`](extensions/declared_target_execution.py)
  is the run that follows it: one call takes that declaration and locally executes a
  circuit the declared device can actually run, so a provider sees the whole path
  end to end without FlagQuantum changing to accept the device.
- [`remote/jiuding_submit.py`](remote/jiuding_submit.py) — it imports its client.
- [`remote/realtime_device_call.py`](remote/realtime_device_call.py) — it imports
  the realtime messaging module directly, because the feedback loop it models is
  a channel between two programs rather than a program to run.
- [`single_machine_quantum_ai/common.py`](single_machine_quantum_ai/common.py) —
  a shared helper for the examples beside it, which imports no FlagQuantum at all.
- [`qec/restructure_compatibility.py`](qec/restructure_compatibility.py) imports
  the QEC public surface to run the existing repetition-memory feedback workflow
  and check its correction, logical error rate and decoded bits.
- [`qec/code_family_imports.py`](qec/code_family_imports.py) imports QEC code
  records through root and package imports to check one surface class, pickle
  paths, Steane matrices and the existing memory noise sweep.
- [`qec/memory_layout_and_sampling.py`](qec/memory_layout_and_sampling.py)
  imports QEC declarations, maps and sampling directly, then runs all three
  existing feedback modes; actual measurement sampling requires the Stim extra.
- [`qec/dem_model_and_chunks.py`](qec/dem_model_and_chunks.py) uses the QEC
  public imports to read exact rates, build a decoding graph, close a complete
  chunk and verify the historical pickle class; no optional SDK is required.
- [`qec/decoder_registry_and_pickle.py`](qec/decoder_registry_and_pickle.py)
  uses QEC registry imports to compare three in-tree decoders and verify the
  canonical matcher pickle class, without an optional SDK.
- [`qec/installed_imports.py`](qec/installed_imports.py) checks existing QEC
  exports, one registered rotated-class pickle path, decoding and a noiseless
  memory sweep; run it outside the checkout to verify an installed wheel.
- [`qec/independent_batch_decoding.py`](qec/independent_batch_decoding.py)
  compares complete sparse-shot batch records with direct single-shot decoding
  across five certified concrete decoders. Requires the PyMatching extra; this
  operation retains correction fields and is separate from logical-only Tensor
  batch predictions. It makes no speed claim.
- [`qec/memory_experiment.py`](qec/memory_experiment.py) composes the approved
  fixed-shot experiment through its canonical QEC execution import. It uses one
  original source and exact final partial batches with five default decoders;
  install the Stim and PyMatching extras. Reports counts without a speed claim.
- [`qec/circuit_noise_dem.py`](qec/circuit_noise_dem.py) builds and samples a
  distance-five circuit-location surface-code DEM without a dense statevector
  or optional SDK; the physical circuit has 49 qubits.
- [`qec/stim_user_migration.py`](qec/stim_user_migration.py) — it imports the QEC
  subpackage surface, where the detector error model reader and the decoder
  registry live, and `stim`, because the circuit layer stays Stim's.
- [`qec/stim_experiment_bridge.py`](qec/stim_experiment_bridge.py) uses the QEC
  domain and opt-in Stim adapter imports rather than the root-level alias. Native
  memory export and external annotated text share the existing sampler, decoder
  and fixed-shot statistics; both original circuits generate their own samples.
- [`qec/stim_pymatching_baseline.py`](qec/stim_pymatching_baseline.py) imports
  QEC sampling and decoder APIs from the subpackage to evaluate streamed Stim
  memory samples with PyMatching batch predictions. It reports fixed-shot logical
  failures, per-observable counts, a Wilson 95% interval and experiment metadata;
  both optional SDKs are required. `--code surface --distance 5 --basis x`
  selects three rounds of the existing rotated Surface code in a certified
  all-data X frame; `--basis z` uses its declared Z logical. The report records
  the logical support and reuses the same fixed-shot statistics.
- [`qec/css_code_from_matrices.py`](qec/css_code_from_matrices.py) — it imports
  the QEC subpackage surface, because the code record built from parity-check
  matrices lives there and has no root-level `fq.` name.
- [`qec/mixed_check_code.py`](qec/mixed_check_code.py) — it imports the QEC
  subpackage surface, because a mixed-check code record, its memory circuit, its
  detector error model and the matching decoder all live there and none has a
  root-level `fq.` name.
- [`dynamic_density_matrix.py`](dynamic_density_matrix.py) — it imports
  `flagquantum.dynamic` and `flagquantum.simulation.density_matrix`, because the
  exact mixed-state route for a mid-circuit measurement is reached through
  `DynamicCircuit.density_matrix()`, while `fq.run` returns sampled trajectories.
- [`dynamic_disjunctive_condition.py`](dynamic_disjunctive_condition.py) — it
  imports `flagquantum.dynamic` and `flagquantum.experimental.dynamic`, because a
  disjunctive condition is a `DynamicCircuit.conditional` keyword, and the two
  public readers of it are the exact route through
  `DynamicCircuit.density_matrix()` and `run_dynamic`.

For exact support levels, consult the
[capability catalog](../docs/generated/CAPABILITIES.md).

## Matrix expectation dynamics

Run `python -m examples.matrix_expectation_dynamics` to evolve a three-level
model using the shared `fq.expectation` request, retrieve a named expectation,
and differentiate its coupling with PyTorch. It uses the existing CPU
Lindblad planner and integrator. Expected output:

```text
Leakage expectation: 0.11757891
Coupling derivative: 0.32210884
Circuit matrix output: refused; use flagquantum.lindblad
```

## Shared dynamics training and replay

Run `python -m examples.shared_dynamics_workflow` to train a three-level and a
spin-mode model with the same output requests, batch axes and PyTorch optimizer,
then replay detached numeric plans. Each model prints axes `(51, 2)`, initial
transfer `0.11757891`, coupling derivative `0.32210884`, `Loss decreased: True`
and `Exact numeric replay: True`. See the
[M2 workflow review](../docs/development/M2_SHARED_WORKFLOW_ACCEPTANCE.md) for
physical references, supported scope and focused tests.

## Lindblad output names

Run `python -m examples.lindblad_output_names` to retrieve two named expectation
trajectories and see an explicit/generated name collision refused with
`duplicate_output_name`. Expected output:

```text
Polarization: 1.0
X expectation: 0.0
Collision refused: duplicate_output_name
```

This uses the existing public Pauli output requests. Full-space matrix request
construction is demonstrated by the matrix dynamics example above.

## Static finite-mode training

Run `python -m examples.train_boson_coupling` to fit a static exchange coupling
with existing `fq.Parameter.bind`, `fq.ExecutionOptions`, and PyTorch Adam.
The example supplies boson operators with live scalar coefficients directly to
the existing Lindblad path and rebuilds the bound request each optimizer step.
No time-dependent pulse or separate trainer is introduced.

## Local basics

Every command on this page runs from the root of a clone of this repository.
The published wheel contains the `flagquantum` package only, so a
`pip install flagquantum` environment has no `examples/`: clone the repository
and run the commands from that working copy.

Start with these small local workflows. The first three use only the stable
`fq.*` API; the dynamic example uses the candidate and experimental surfaces
named below:

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
python -m examples.local.dynamic_feedback
python -m examples.local.dynamic_many_shots
```

They cover local statevector simulation, exact and sampled measurements,
PyTorch-native training, and experimental local dynamic feedback without
credentials, remote resources, or optional backends. See the
[annotated local guide](../docs/guides/LOCAL_WORKFLOWS.md) before moving to
configurable research examples.

To inspect the local QBoson Kaiwu matrix boundary without installing the Kaiwu
SDK or submitting a remote task, run:

```bash
python -m examples.kaiwu_matrix_boundary
```

## First CPU execution

Start with the complete local execution path:

```bash
python -m examples.cpu_statevector
```

The example builds a Bell-state circuit, creates an inspectable execution plan,
runs that exact plan on the PyTorch CPU statevector engine, and checks the
double-precision result against its analytical state. It disables backend
fallback so a successful run proves the reported CPU path was actually used.

To inspect target-independent compiler optimization separately:

```bash
python -m examples.compiler_optimize
```

This example removes redundant gates, verifies that optimization reaches a
fixed point, and compares the optimized program with the original numerical
result. It uses `compiler.optimize`; target-aware lowering and routing belong to
`compiler.compile`.

It is also the compiler's optimization-level path. It prints the gate count at
every implemented level, proves each level preserves the source state, shows a
circuit where level `2` reaches two gates where level `1` stops at three and
where level `0` returns the program as submitted, forwards the same level through
`compiler.compile`, and prints the reserved level failing closed with its reason.
The level vocabulary is read from
`flagquantum.compiler.optimization_levels`, so adding a level changes one module
and this path reports it.

To rewrite gates a target cannot run into the gates that target publishes:

```bash
python -m examples.compiler_synthesis
```

This example declares a target basis (a z-rotation, a pi/2 pulse about `x`, and
one entangler) and then spells an `h`, a `swap`, and a four-amplitude state in
that basis. Each case executes the rewrite next to the original on the shipped
statevector engine, removes the global phase a synthesis result cannot record
in FlagQuantum IR, and prints the residual that remains. It also prints what the
boundary refuses -- a non-supercontrolled entangler, a z-rotation that is not
entrywise exact, and a pulse opcode the module cannot emit -- because a refusal
is the boundary's answer, not a fallback. The three entry points are reached by
module path; none of them is a stable `fq.compiler` export.

To exercise the optional single-GPU Triton kernel for local one-qubit gates:

```bash
FQ_STATEVECTOR_TRITON_LOCAL_1Q=1 \
  python -m examples.triton_statevector_local_1q
```

The example uses the public `flagquantum.runtime.run_distributed` entry point.
It compares the rank-owned CUDA state with a CPU reference, requires the runtime
to select Triton, and prints the measured compiler distribution and integration
path. On one GPU, every circuit qubit is local. In a sharded statevector, this
kernel is eligible only for a qubit whose amplitude pairs remain on the same
rank. The ordinary single-GPU `fq.run(..., mode="statevector")` path currently
uses the local simulator instead of this distributed-statevector kernel.

Compile non-local gates for a concrete hardware topology with:

```bash
python -m examples.target_aware_compilation
```

The example targets a five-qubit line, checks every emitted two-qubit operation
against that connectivity, verifies logical-qubit restoration, and executes the
compiled IR against the original result.

Route onto a device by what its links cost rather than by how far apart they are:

```bash
python -m examples.weighted_routing
```

A `CouplingMap` says which qubits a device couples, so the router can only count
hops. `LinkErrorRates` is the same couplings with the measured two-qubit error of
each link, which makes the cheapest route and the cleanest route different
questions: here both routes between two qubits are two hops and both insert the
same two SWAPs, and only the calibration separates them -- 0.003 expected link
faults against the 0.04 the hop-count route would have paid. The example prints
both programs, the equal SWAP counts, the fault total before and after routing,
and the refusal a planner that never reads the calibration receives. A calibrated
device also changes which strategies are ranked, so the automatic choice is a
choice between fault totals.

## Start in one minute

From an editable development installation:

```bash
python examples/quick_start.py --mode sv --steps 40
```

This trains a hybrid model with a `torch.nn.Linear` encoder and an `fq.Module`
quantum layer in one PyTorch optimizer loop. The example uses an analytical
target, so it reports correctness as well as training loss.

Switch the simulation representation without rewriting the model:

```bash
python examples/quick_start.py --mode mps --steps 40
python examples/quick_start.py --mode tn --steps 40
```

Statevector is the recommended first run. MPS and tensor-network support
boundaries are listed in the capability catalog.

## Choose a workflow

| Goal | Recommended entry | Scope |
| --- | --- | --- |
| Learn circuits, measurements, gradients, and QML | [Tutorials](tutorials/README.md) | Guided notebooks |
| Run one quantum algorithm unit end to end | [Algorithm examples](algorithms/README.md) and the [algorithms guide](../docs/guides/ALGORITHMS.md) | Demonstration-scale units, subpackage surface |
| Verify the local CPU or one-GPU path | [Single-machine quantum AI](single_machine_quantum_ai/README.md) | Supported local workflows |
| Train a local statevector VQE | [`01_vqe_statevector.py`](single_machine_quantum_ai/01_vqe_statevector.py) | Exact differentiable simulation |
| Compare gradient methods and read the one that ran | [Gradient methods](gradient_methods/README.md) | One entry point, reported method |
| Train with MPS | [`03_mps_training.py`](single_machine_quantum_ai/03_mps_training.py) | Low-entanglement systems |
| Use a JAX kernel through PyTorch | [`04_jax_kernel_torch_layer.py`](single_machine_quantum_ai/04_jax_kernel_torch_layer.py) | Optional accelerator path |
| Run a noisy circuit exactly instead of by trajectory | [`density_matrix_execution.py`](density_matrix_execution.py) | Dense exact oracle, 2**n by 2**n state |
| Measure mid-circuit and keep the exact mixture | [`dynamic_density_matrix.py`](dynamic_density_matrix.py) | Dynamic subpackage surface, one branch per read bit |
| Fire a gate on either of two readouts | [`dynamic_disjunctive_condition.py`](dynamic_disjunctive_condition.py) | Dynamic subpackage surface, disjunction of condition clauses |
| Inspect sharded statevector ownership | [Distributed statevector](distributed_statevector_topologies/README.md) | One logical statevector across ranks |
| Inspect rank-owned MPS execution | [Distributed MPS](distributed_mps/README.md) | Development evidence |
| Train and package a circuit | [`train_parameterized_circuit_then_deploy.py`](train_parameterized_circuit_then_deploy.py) | Deployment bridge |
| Build an extension | [`extensions/reference_extensions.py`](extensions/reference_extensions.py) | Experimental API |
| Decode a Stim detector error model here | [`qec/stim_user_migration.py`](qec/stim_user_migration.py) | Experimental QEC surface |
| State a stabilizer code as matrices | [`qec/css_code_from_matrices.py`](qec/css_code_from_matrices.py) | Experimental QEC surface |
| Measure what a mixed-check patch costs | [`qec/mixed_check_code.py`](qec/mixed_check_code.py) | Experimental QEC surface |

Larger application and research examples are intentionally not presented as
minimal getting-started paths.

## Emulate a target without submitting

Compile a program for a declared third-party target and run the compiled program
on this machine before spending anything on a submission:

```bash
python -m examples.remote.emulate_local_target
```

The example declares a provider listing, reads what the target's own passes
changed -- routing, native-gate decompositions, the emitted payload -- and then
executes the compiled program locally. It contacts no provider and needs no
credentials. The local device, the precision, and the result are the machine's;
the qubit capacity, native gate set, and program limits are the profile's
declaration, and the record keeps those two sources apart.

## Attach your own operator to a declared gate

`Circuit.gate(name, qubits, matrix=...)` attaches an operand: from that point the
instruction is the matrix you wrote and `name` is only the spelling it was
registered under. Three executors used to pick a kernel from the name before they
read the operand, so an `x` carrying a matrix applied `X`, and the identity pass
deleted a `rz(theta=0.0)` that carried one.

```bash
python -m examples.carried_operand_execution
```

The example attaches one real symmetric unitary -- not `X`, not `Z`, not a phase
-- to `x`, `y`, `z` and `rz`, and prints what `Circuit.state()` and `run_dynamic()`
each produce next to the operand's own image of `|0>`. That image differs from the
named operators' images by a whole amplitude, so a route that substitutes the name
is visible in the transcript rather than inferred. It prints the two controls too:
`x` with no operand is still `X|0>`, and a zero-angle `rz` that carries nothing is
still removed. Everything runs locally in one process.

## Decode a Stim circuit's error model without Stim's decoder

Replace the decoding half of a Stim QEC workflow while keeping Stim for the
circuit layer (requires the `stim` extra, and the `pymatching` extra for the
cross-check section):

```bash
python -m examples.qec.stim_user_migration
```

The example reads the detector error model Stim wrote, samples it, decodes it
with this package's matcher, and prints the two readings of Stim's decomposed
`^` separator side by side with the observable marginal each one implies. It
runs locally in one process, contacts nothing, and establishes no threshold. The
route it follows, the four reader refusals, and the tradeoff between the two
readings are documented in
[Migrating a Stim workflow](../docs/guides/STIM_USER_MIGRATION.md).

## Declare a code this package does not ship

Three code records ship here. This example takes the other route, writing a code
down as parity-check matrices, and runs the result end to end:

```bash
python -m examples.qec.css_code_from_matrices
```

It states the Steane code as matrices and holds the record it builds against the
record this package ships, then builds the toric code -- which no record here
declares -- on two lattices, derives its distance from the matrices rather than
from a declared logical operator, and runs it through a memory circuit, a
detector error model, sampling, and the matcher. It prints the matrices that are
refused beside the ones that are admitted. It runs locally in one process,
contacts nothing, and establishes no threshold and no scalability claim. The
record it demonstrates is documented in
[FlagQuantum QEC](../flagquantum/qec/README.md).

## Run on remote resources

After completing the local examples, use the provider-specific golden paths:

```bash
python examples/remote/quafu_bell.py
python examples/remote/jiuding_workspace_bell.py
```

The Quafu path compiles and validates a circuit before submitting it to a QPU.
The Jiuding path reuses a running workspace for low-latency remote compute.
Both require provider credentials and configured remote resources.
Use `--list-workspaces` first when the account can see more than one Jiuding
workspace; a single visible workspace is selected automatically.

## Plan before execution

Use the runtime planner when you need to inspect representation choice,
gradient support, or blockers before running:

```python
import flagquantum as fq

circuit = (
    fq.Circuit(n_qubits=4)
    .h(0)
    .cx(0, 1)
    .rzz(1, 2, theta=0.2)
)

plan = circuit.runtime_plan(prefer_jax=True, require_gradients=True)
print(plan.summary())
```

A plan is an explanation of intended execution, not benchmark evidence.
Performance and scalability statements must use runtime-generated records and
report their `distribution_semantics`.

## Recommended smoke runs

These small commands are suitable for checking a development environment:

```bash
python examples/single_machine_quantum_ai/00_local_fast_path_check.py
python examples/single_machine_quantum_ai/01_vqe_statevector.py \
  --steps 2 --n-qubits 3
python examples/single_machine_quantum_ai/02_quantum_classifier.py --steps 2
python examples/single_machine_quantum_ai/03_mps_training.py \
  --steps 2 --n-qubits 4 --max-bond 8
```

Optional JAX check (requires the `jax` extra from `pyproject.toml`; without it
the script reports `status: skipped`):

```bash
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py \
  --steps 1 --bench-iters 1
```

The curated single-machine examples do not initialize distributed backends and
make no distributed scalability claim.

## Example quality contract

A curated example must:

1. state the task, runtime mode, and maturity boundary;
2. expose practical arguments for problem size, steps, and device;
3. run end to end from a documented installation;
4. report a correctness metric or an explicit reference value;
5. identify local, replicated, sliced, or sharded execution accurately;
6. use stable public API unless explicitly labeled experimental.

Shared helpers belong next to the examples that use them. Example-only
convenience code must not become a framework abstraction without a separate API
review.


## LaTeX and hybrid-model Notebook

[`notebooks/hybrid_latex_drawing.ipynb`](notebooks/hybrid_latex_drawing.ipynb)
shows existing text/Matplotlib drawing, LaTeX export, inline Notebook images,
a real CPU CNN/quantum/head training step, and a portable Overleaf bundle.
Source export uses the default install; image display additionally requires
a TeX engine and Poppler on the kernel PATH.

## Terminal instrument mathematical references

Run `python -m examples.terminal_instrument_reference` for proposal 083's
current-API references. Existing SuperOperator Kraus maps calculate branch
probabilities/backaction; Torch supplies illustrative samples. Expected:

```text
presence labels: ('no_flag', 'flag')
presence probabilities: [0.68, 0.32]
presence Torch draw replay: True
No-flag conditional ground: 0.88235294
Unread measurement changes coherence: True
fock labels: ('0', '1', '2')
fock probabilities: [0.6, 0.3, 0.1]
fock Torch draw replay: True
Framework terminal instrument API: flagquantum.runtime.sample_terminal
```

The coherent g/lost example distinguishes physical loss from observed flags and
shows that an unread instrument can change coherence. The finite three-level
Fock example explicitly models QND number projection. Neither is a calibrated
photodetector or framework event API. The approved
[proposal 083](../docs/development/API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md)
is implemented by the Runtime-only sample_terminal primitive. Existing Program
IR fields, plan formats and MeasurementResult fields remain unchanged.

Run `python -m examples.terminal_instrument_sampling` for real Runtime samples:

```text
presence labels: ('no_flag', 'flag')
presence probabilities: [0.68, 0.32]
presence indices shape: (4096,)
presence private-seed replay: True
fock labels: ('0', '1', '2')
fock probabilities: [0.6, 0.3, 0.1]
fock indices shape: (4096,)
fock private-seed replay: True
global RNG unchanged: True
```

```python
from examples.terminal_instrument_reference import presence_reference
from flagquantum.runtime import sample_terminal

rho, instrument = presence_reference()
result = sample_terminal(rho, instrument, shots=4096, seed=7)
print(tuple(result.metadata["outcome_labels"]))  # ('no_flag', 'flag')
print(result.value.shape)  # torch.Size([4096])
```

The caller declares the complete Kraus instrument and its coordinate basis.
Sampling is CPU-only and does not carry gradients; use analytic evolution
probabilities and expectations for training. With identical inputs, effective
seed, Torch version and CPU runtime, samples replay. No cross-version bitwise
replay, feedback, detector trajectory or hardware submission is implied.


## Two-mode interference and number counting

Run `python -m examples.boson_interference` on CPU. The existing boson algebra
builds the passive coupling `0.7 * (a0_dagger * a1 + a1_dagger * a0)` from
`|1, 1>`, for duration `T = pi / (4 * 0.7)`. Each mode has three levels.
The same Lindblad plan supports uniform pure loss with rate 0 or 0.2 per mode;
the Runtime terminal primitive samples the tensor product of the declared
three-level QND number projectors. There is no second evolution or sampling engine.

Expected analytic outputs:

| Loss rate | P(0,2) | P(2,0) | P(1,1) | Two-photon survival |
| --- | --- | --- | --- | --- |
| 0.0 | 0.50000000 | 0.50000000 | 0.00000000 | 1.00000000 |
| 0.2 | 0.31919722 | 0.31919722 | 0.00000000 | 0.63839443 |

The subsequent lines print number labels in mode-0/mode-1 order:
`['0/0', '0/1', '0/2', '1/0', '1/1', '1/2', '2/0', '2/1', '2/2']`.
Each nine-element count vector sums to `4096`, followed by `Shots: 4096`
and `Number sample replay: True`. In Python 3.12 / Torch 2.13 CPU, seed 7 gave
`[0, 0, 2035, 0, 0, 0, 2061, 0, 0]` without loss and
`[174, 659, 1281, 705, 0, 0, 1277, 0, 0]` with loss. The exact count vectors
are not a cross-version guarantee. Vacuum and one-photon outcomes are retained;
two-photon survival is not an implicit postselection step.

An independent closed form uses transmission `eta = exp(-rate * T)`:
vacuum probability `(1-eta)^2`, each one-photon probability `eta*(1-eta)`,
each bunched probability `eta^2/2`, and zero coincidence probability.
The two bunched states retain coherence `eta^2/2` before measurement. The
unread QND instrument removes it, which distinguishes the declared measurement
backaction from merely reporting populations or assuming a destructive detector.

Integration tests compare the full complex density matrix with this formula and
embed the three-level result in a four-level calculation at absolute error
`1e-10`. For this fixture the passive coupling preserves total photon number
and loss only lowers it; the initial total is two, so occupation above two is
unreachable. This explains the exact truncation closure for this input, rather
than certifying a general infinite-Fock simulation. Shot-frequency uncertainty
is checked separately with binomial standard errors. This finite CPU reference
does not provide a Gaussian backend, optical program frontend, destructive
photodetection, event timestamps or hardware submission.

## Native moving-atom dynamics

Run `python -m examples.rydberg_native_dynamics` for a real RydbergProgram on
the existing CPU Lindblad planner. The source declares two atoms, a linear move,
C6 and absorbing loss; Simulation constructs the interval matrices. Expected:

```text
Native plan version: 1.7
Physical survival: 0.81873075
Unconditional RR: 0.40936538
No-flag probability: 0.85353566
Exact numeric replay: True
```

The one/two-actor profile uses g/r/lost states and interval-midpoint geometry.
Refine the grid to check the imposed-path approximation independently of solver
accuracy. The no-flag effect assumes independent detection with efficiency 0.8;
it is distinct from physical survival and unconditional RR. There are no sampled
events, heating, quantum motion or hardware submission. This native example
replaces the fixed matrix adapter from the
[contract baseline](../docs/api-changes/FQ-RYDBERG-TRANSPORT-PROGRAMS-20261022.md).

## Native Rydberg program geometry

Run `python -m examples.rydberg_program_geometry` to construct a real native
RydbergProgram, append named moves and pulses, and query verified commanded
positions. Expected output:

```text
Atom order: ('a', 'b')
Product dimensions: (3, 3)
Commanded b position at t=0.25: (3.0, 0.0)
Survival effect shape: (9, 9)
Circuit route: refused; use flagquantum.lindblad for native evolution
```

This is the Core construction slice of proposal 081. Commanded positions are
not evidence of physical atom presence. Use the moving-atom example above for
native numerical evolution; detector events and hardware submission remain open.

## Train native moving-atom controls

Run `python -m examples.train_rydberg_controls --steps 40 --intervals 128 --shots 4096`
on CPU.
The source declares a moving atom pair and one named pulse amplitude. Two
per-interval amplitude bindings train through the existing Lindblad planner,
Runge-Kutta integrator and PyTorch Adam; geometry, C6, loss and time boundaries
remain fixed. Every optimizer step rebuilds its live plan. Expected output:

```text
Runtime: CPU, two actors, interval_midpoint
Initial unconditional RR: 0.04305292
Initial amplitude gradient: [0.07881346, 0.07879645]
Training reduced loss: True
Final unconditional RR: 0.12278807
Survival-conditioned RR: 0.14997368
No-flag-selected RR: 0.14385816
Exact numeric replay: True
Observed flag labels: ['no_flag/no_flag', 'no_flag/flag', 'flag/no_flag', 'flag/flag']
Observed flag probabilities: [0.85353566, 0.07033428, 0.07033428, 0.00579579]
```

The remaining lines report four observed counts summing to `4096`,
`Detection shots: 4096`, and `Detection replay: True`. Counts depend on the Torch
version; seed 7 replays in the same CPU environment. Each frequency has binomial
standard error `sqrt(p * (1 - p) / shots)`, separate from integration error.

Survival conditioning and no-flag selection use different explicit denominators.
Training uses analytic expectations; only the detached final plan requests dense
states for Runtime terminal sampling. Detection uses the declared efficiency 0.8
and zero false positives, with outcome order following atoms a/b. With equal
g/r loss rates, each observed flag has probability `0.8 * (1 - exp(-0.1))`,
independent of the optimized amplitudes. Consequently, a no-flag result includes
undetected physical loss and is not a survival certificate. The no-flag RR ratio
matches conditioning with the declared square-root Kraus branch. This is one
terminal measurement, without timestamps or feedback.

Detached plan replay preserves numbers rather than a training graph. The
128-interval default meets the existing 1e-10 frozen
unitary/gradient reference budget for this fixture. At 64 intervals one gradient
differed by 2.28e-10 and failed that unchanged budget; refinement fixes the
integration error. Imposed-path midpoint convergence remains a separate check.
The finite-dimensional example is not heating, hardware or scalability evidence.

## Native transport geometry and schedule view

Install the existing `flagquantum[viz]` extra and run:

```bash
MPLBACKEND=Agg python -m examples.visualize_rydberg_transport --output transport.svg
```

Expected output:

```text
Commanded b at t=0.5: (3.0, 0.0)
Schedule: move b (aod); pulse a,b (laser)
Saved: transport.svg
```

Use --output transport.png for a raster image and --samples 101 for a finer
**display** sampling. The two panels show declared sites/commanded paths and
source operation durations/channels. Positions call Core's commanded_position;
the renderer does not duplicate interpolation or infer presence, trap activity,
heating or gate fidelity. Pulse bars show duration, not amplitude: the named
omega control is bound per interval by the training example. Display samples
are not the numerical integration grid. This example reuses the moving-pair
source from train_rydberg_controls and adds no public drawing API or renderer
registry. The exported SVG/PNG is suitable for inspection without a GUI.

### Ising fusion observation

Run `python -m examples.ising_fusion_measurement` after
`python -m examples.ising_braid_execution` to inspect ideal pair-charge QND
measurement. Both initial fusion channels use the same native braid, sealed
plan replay and explicit terminal instrument. The example shows input coherence
0.5 becoming zero when the result is unread, while each normalized conditional
state preserves its measured charge on a second observation. These conditional
states are calculated explicitly using existing Kraus maps; the sampler itself
returns outcome indices, not post-measurement states or feedback events.

At 20,000 shots, both probabilities are 0.5 and their binomial standard error is
about 0.003536. The numerical probability error is checked separately against
1e-12; sampled frequencies use a six-standard-error bound plus two counts. A
private seed reproduces outcomes without changing caller RNG state. This is the
fixed four-sigma, total-vacuum CPU model, not hardware, other anyon sectors, or a
universal gate set.

### Spin-mode thermal truncation

Run `python -m examples.spin_mode_convergence` for resonant, lossless spin-mode
exchange, with spin first and a thermal oscillator second. Unlike the vacuum
one-excitation training example, this input has a nonzero tail at every Fock
cutoff. The example compares mode dimensions 3, 5 and 7 to independent closed
Rabi blocks and a 64-term geometric thermal sum (omitted mass below 0.2**64).
It checks full complex density matrices, total excitation conservation, coupling
gradients, named outputs, optimizer updates and detached numeric JSON replay.

Finite initial thermal weights are explicitly normalized after truncation. The
highest excited spin/mode state is artificially dark because the next Fock level
is absent. Both conditioning away the tail and this blocked transition enter
the reported probability-error bound `q**N + (1-q)*q**(N-1)/(1-q**N)`. Increasing
cutoff reduces the observed final-time error for this declared parameter set;
this is not a universal monotonic-convergence rule. Integration error is checked
separately at 1e-10. The fitted target uses the same finite model, so fitting is
not evidence that its truncation error disappeared.

The example composes existing mixed dimensions, boson algebra, parameter binding,
Lindblad planning, Krylov evolution and PyTorch. It adds no ion-trap frontend,
heating, laser calibration, hardware execution or distributed-capacity claim.

### Driven three-level leakage and time-grid convergence

Run `python -m examples.driven_three_level` for a rotating-frame, weakly
anharmonic three-level model with four noncommuting in-phase/quadrature pulses.
The model includes the sqrt(2) ladder coupling to level 2, detuning 0.15 and
anharmonicity -3, in inverse time units. A segment lasts 0.25. There is no
dissipation, calibrated device, hardware pulse emission or ion-trap model.

Existing interval parameters bind all eight control values. Named outputs
report ground/excited/leakage populations. Independent 3x3 unitary products
check the entire complex density at pulse boundaries; reversed pulse order must
give a different result. All eight leakage derivatives are compared against
four-point finite differences of that independent reference.

Refinement subdivides the same constant segments and repeats the same bindings.
It does not change or resample the waveform: Runge-Kutta errors decrease by
about 16 per doubling, while the existing Krylov path is checked at 1e-10.
The optimizer minimizes `(1-P1)**2 + 2*P2`; its improved population and leakage
are not a unitary-gate fidelity claim. Detached numeric plans replay exactly,
but JSON does not preserve the training graph. All solvers, bindings, outputs
and artifacts reuse the existing shared dynamics workflow.
