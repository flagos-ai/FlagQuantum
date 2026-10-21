# FlagQuantum Examples

Runnable, copy-ready workflows for building and training quantum AI programs
with the current FlagQuantum API.

Examples driven by the root-level `fq` alias use:

- `import flagquantum as fq` as the public entry point;
- `fq.Circuit(n_qubits=...)` for circuit construction;
- PyTorch for parameters, gradients, and optimizers;
- explicit runtime and distribution semantics when making performance claims.

These examples do not use that alias:

- [`boson_interference.py`](boson_interference.py) — it uses the existing
  `flagquantum.lindblad` namespace and boson algebra for two-photon interference
  and loss with explicit mode truncations. Run `python -m examples.boson_interference`.
  Unconditional output probabilities, survival, and plan serialization share the
  existing dynamics path; this is a bounded CPU model.
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
- [`qec/stim_user_migration.py`](qec/stim_user_migration.py) — it imports the QEC
  subpackage surface, where the detector error model reader and the decoder
  registry live, and `stim`, because the circuit layer stays Stim's.
- [`qec/css_code_from_matrices.py`](qec/css_code_from_matrices.py) — it imports
  the QEC subpackage surface, because the code record built from parity-check
  matrices lives there and has no root-level `fq.` name.
- [`qec/mixed_check_code.py`](qec/mixed_check_code.py) — it imports the QEC
  subpackage surface, because a mixed-check code record, its memory circuit, its
  detector error model and the matching decoder all live there and none has a
  root-level `fq.` name.
- [`dynamic_density_matrix.py`](dynamic_density_matrix.py) — it imports
  `flagquantum.dynamic` and `flagquantum.simulation.density_matrix`, because the
  exact route for a mid-circuit measurement is reached through
  `DynamicCircuit.density_matrix()` and `fq.run` still refuses a dynamic program
  by name.
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

Start with these small local workflows. The first three use only the stable
`fq.*` API; the dynamic example uses the candidate and experimental surfaces
named below:

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
python -m examples.local.dynamic_feedback
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
