# FlagQuantum Examples

Runnable, copy-ready workflows for building and training quantum AI programs
with the current FlagQuantum API.

Examples driven by the root-level `fq` alias use:

- `import flagquantum as fq` as the public entry point;
- `fq.Circuit(n_qubits=...)` for circuit construction;
- PyTorch for parameters, gradients, and optimizers;
- explicit runtime and distribution semantics when making performance claims.

These examples do not use that alias:

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

For exact support levels, consult the
[capability catalog](../docs/generated/CAPABILITIES.md).

## Three local basics

Start with three small programs that use only the stable `fq.*` API:

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
```

They cover local statevector simulation, exact and sampled measurements, and
PyTorch-native training without credentials, remote resources, or optional
backends. See the [annotated local guide](../docs/guides/LOCAL_WORKFLOWS.md)
before moving to configurable research examples.

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

To exercise the optional single-GPU Triton kernel for local one-qubit gates:

```bash
FQ_STATEVECTOR_TRITON_LOCAL_1Q=1 \
  python -m examples.triton_statevector_local_1q
```

The example uses the public `flagquantum.runtime.run_distributed` entry point.
It compares the rank-owned CUDA state with a CPU reference, requires the runtime
to select Triton, and prints the measured compiler distribution and integration
path. On one GPU, every circuit wire is local. In a sharded statevector, this
kernel is eligible only for a wire whose amplitude pairs remain on the same
rank. The ordinary single-GPU `fq.run(..., mode="statevector")` path currently
uses the local simulator instead of this distributed-statevector kernel.

Compile non-local gates for a concrete hardware topology with:

```bash
python -m examples.target_aware_compilation
```

The example targets a five-qubit line, checks every emitted two-qubit operation
against that connectivity, verifies logical-wire restoration, and executes the
compiled IR against the original result.

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
| Inspect sharded statevector ownership | [Distributed statevector](distributed_statevector_topologies/README.md) | One logical statevector across ranks |
| Inspect rank-owned MPS execution | [Distributed MPS](distributed_mps/README.md) | Development evidence |
| Train and package a circuit | [`train_parameterized_circuit_then_deploy.py`](train_parameterized_circuit_then_deploy.py) | Deployment bridge |
| Build an extension | [`extensions/reference_extensions.py`](extensions/reference_extensions.py) | Experimental API |
| Decode a Stim detector error model here | [`qec/stim_user_migration.py`](qec/stim_user_migration.py) | Experimental QEC surface |
| State a stabilizer code as matrices | [`qec/css_code_from_matrices.py`](qec/css_code_from_matrices.py) | Experimental QEC surface |

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
