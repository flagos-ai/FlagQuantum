# Jiuding execution (experimental)

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

Inside a Jiuding development workspace, the client reads the injected
`/etc/accesskey/user-ak` and `user-sk` automatically. Outside Jiuding, set both
`JIUDING_AK` and `JIUDING_SK`; these environment values override injected files
only as a complete pair. Partial credentials are rejected. Tokens stay in memory
and refresh according to the server expiry. Requests use HTTPS, reject redirects,
have a 20-second socket timeout, and are not retried blindly. No CLI installation
or manual project and queue IDs are required.

## First account check

A new account must first be added to a Jiuding project and an active compute
queue by a platform administrator. FlagQuantum cannot grant quota or project
membership. Once at least one development workspace is visible, verify access
without copying project or queue IDs:

```bash
python examples/remote/jiuding_workspace_bell.py --list-workspaces
```

The command reports only non-secret workspace, project, queue and accelerator
names. If exactly one workspace is visible, it is selected automatically and a
CPU Bell check needs no arguments:

```bash
python examples/remote/jiuding_workspace_bell.py
```

With multiple workspaces, select one explicitly. GPU use is always explicit:

```bash
python examples/remote/jiuding_workspace_bell.py \
  --workspace my-workspace \
  --target jiuding:gpu
```

The selected workspace must already be running and its image must contain a
compatible FlagQuantum installation. An account with no visible workspace gets
an actionable project-and-queue access error instead of a request for internal
IDs. Creating the first project, assigning quota and publishing a generally
available runtime image remain platform administration operations.

## Choose an execution path

Jiuding has two primary execution paths. They return the same FlagQuantum
result type but have different latency and lifecycle semantics.

| Need | Entry point | Required resource | Lifecycle |
| --- | --- | --- | --- |
| Interactive or repeated execution | [`fq.run(..., target="jiuding:...")`](#repeated-low-latency-workspace-execution) | A running development workspace | Reuses one resident process and SSH channel; no Job ID |
| Isolated, schedulable, recoverable work | [`JiudingClient.submit_program(...)`](#recoverable-batch-execution) | A running workspace, runtime image and queue capacity | Creates a batch Job; recoverable by Job ID |

Use `fq.run` when startup latency would dominate the calculation. Use
`submit_program` when the work should survive the caller process, wait in the
queue, or be recovered later. Neither path creates a development workspace
implicitly. `JiudingClient.submit()` is the advanced escape hatch for an
existing Python script and shared-storage layout; ordinary circuit users do
not need it.

## Development workspaces

FlagQuantum can also create and control a Jiuding development workspace. The
client uses an existing visible workspace only to identify the project, queue,
cluster and storage context; the new workspace is a separate platform resource.

```python
from flagquantum.remote.compute import JiudingClient

client = JiudingClient(workspace="fq-image-build-upload")
created = client.create_workspace(
    "example-resident-a100",
    target="jiuding:gpu/NVIDIA_A100-SXM4-40GB",
    image="flagquantum-runtime:v0.2.0-fcbaf8e6e96bcd24213b8eaa640139d8b0349759-cu128-a100",
    image_region="PRIVATE",
    accelerator_count=1,
    cpus=4,
    memory_gib=16,
)
print(client.workspace_state(created["id"]))
```

Creation starts the workspace but does not enable privileged mode, Jupyter
Cloud IDE or automatic snapshots. It is an explicit billable infrastructure
operation and is never triggered by `fq.run()`. Use
`client.stop_workspace("example-resident-a100")` to stop it without saving a
container snapshot and `client.start_workspace("example-resident-a100")` to
restart it. Source code and durable results must remain on mounted storage;
stopping a workspace does not make its container filesystem durable.
Set `accelerator_count=2` or higher only for an explicitly distributed
workspace. A larger allocation exposes devices to the workspace; it does not
turn replicated work into sharded execution. Launch a supported distributed
FlagQuantum engine under `torchrun` and retain its distribution evidence.

## Custom script jobs (advanced)

```python
from pathlib import Path
from flagquantum.remote.compute import JiudingClient

client = JiudingClient()  # matches current pod to one visible workspace
print(client.images())   # images observed in this queue's existing jobs

receipt = client.submit(
    "/shared/my_project/experiment.py",
    target="jiuding:cpu",
    image="<choose a compatible image from the list>",
    pythonpath="/shared/my_project",
    receipt="/shared/my_project/run-001.json",
)
value = client.result(receipt, timeout=180)
```

The `/shared/...` paths are placeholders: use the actual storage already mounted
in your workspace. Its existing storage mounts are reused by the job. The
selected image must contain compatible Python, PyTorch and either FlagQuantum
or the submitted source root. `python` defaults to the submitting interpreter's
absolute path and can be overridden. `pythonpath` replaces PYTHONPATH in the
submitted command. The adapter does not upload source, build images or infer
that local paths exist in a different container. It requires a pre-provisioned
shared project. Images listed from previous jobs are evidence of prior use,
not a certification of their current availability or dependencies.

`experiment.py` defines `main()` returning a JSON-serializable value:

```python
import flagquantum as fq

def main():
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(circuit, options=fq.ExecutionOptions(device="cpu"))
    return {"probabilities": result.to_statevector().abs().square().tolist()}
```

Keep ordinary function calls and imports inside the shared project. Run `main()`
locally for debugging. Remote submission adds process isolation and platform
scheduling; it does not change fq.run or PyTorch training semantics. This first
adapter allocates one instance (default 2 cores, 2 GiB, 0 GPUs).
Select GPU execution explicitly with `target="jiuding:gpu"`; this first adapter
allocates exactly one GPU. CPU is a first-class target, never an implicit
fallback from an unavailable accelerator.

```python
receipt = client.submit(
    "/shared/my_project/experiment_gpu.py",
    target="jiuding:gpu/NVIDIA_A100-SXM4-40GB",
    image="flagquantum-runtime:v0.2.0-36ee8a65ae509a0df0d25318015a6433194c7b53-cu128-a100",
    image_region="PRIVATE",
    pythonpath="/shared/my_project",
    receipt="/shared/my_project/gpu-001.json",
    cpus=4, memory_gib=8,
)
value = client.result(receipt, timeout=180)
```

The GPU script selects `fq.ExecutionOptions(device="cuda:0")`. Requesting a GPU
allocates a resource; numerical device selection remains in the user program.
The generic `jiuding:gpu` target resolves to the selected queue's GPU model. A
`jiuding:gpu/<model>` target requires an exact model match.

## Recoverable batch execution

Submit a circuit directly when it should run as an independent Jiuding Job:

```python
import flagquantum as fq
from flagquantum.remote.compute import JiudingClient

client = JiudingClient(workspace="my-workspace")
receipt = client.submit_program(
    fq.Circuit(2).h(0).cx(0, 1),
    target="jiuding:gpu",
    image="flagquantum-runtime:v1",
    outputs=fq.counts(),
    shots=1024,
)
print(receipt["jobId"])

result = client.result(receipt, timeout=600)
print(result.counts)
```

The default path stages the bounded program request, runner, source snapshot,
receipt and result under `/share/project/.flagquantum`. The image remains an
explicit choice. If the local process exits, restore the same Job without
submitting another one:

```python
client = JiudingClient(workspace="my-workspace")
receipt = client.restore_receipt("<jiuding-job-id>")
result = client.result(receipt, timeout=600)
```

Keep the Job ID until the result is collected. `result()` waits for the existing
Job; it never interprets a timeout as permission to resubmit. Use
`receipt="/shared/.../run.json"` only when you intentionally manage the shared
artifacts yourself. See
[`jiuding_submit_program.py`](../../examples/remote/jiuding_submit_program.py)
for the runnable command-line example. Add `--detach` to print the new Job ID
and return immediately instead of waiting through queue and execution time:

```bash
python examples/remote/jiuding_submit_program.py \
  --workspace my-workspace \
  --image flagquantum-runtime:v1 \
  --detach
```

The same command restores the existing Job without submitting another one:

```bash
python examples/remote/jiuding_submit_program.py \
  --workspace my-workspace \
  --restore-job <jiuding-job-id>
```

## Repeated low-latency workspace execution

Batch Jobs are appropriate for isolated, schedulable workloads, but their
container startup time dominates tiny circuits. For interactive or repeated
work, keep one development workspace running and reuse a resident executor.
Set the workspace once, then use the normal FlagQuantum entry point:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(
    circuit,
    target="jiuding:gpu",
    outputs=fq.expectation(fq.X(0) @ fq.X(1)),
)
print(result.expectation())
```

Inside the selected Jiuding workspace, no workspace name is required. Otherwise,
set `JIUDING_WORKSPACE` to the running workspace name before starting Python.
The first call verifies the workspace, establishes SSH, and starts the worker
when necessary. Later calls in the same process reuse both the resident Python
process and SSH channel. Use `JiudingClient(workspace=...)` directly when one
process must address workspaces explicitly. The worker listens only on workspace
loopback, executes through FlagQuantum Runtime, rejects target mismatches,
records the actual device and CPU-fallback status, and returns a normal
`ExecutionResult`.
Call `client.close()` or use the context manager to release the local channel.
If the workspace has restarted, the next readiness check discards its cached
SSH endpoint, discovers the replacement endpoint, and starts a new worker.
An interrupted execution request is never retried automatically because its
completion status may be ambiguous; the following explicit call performs
readiness recovery before submitting new work.

This path supports statevectors, probabilities, Pauli expectations, samples,
and counts. It is not a replacement for Jiuding batch scheduling, multi-node
launch, or a public multi-user service. Large full-state transfers remain
bounded by the protocol message limit; use batch artifacts for large outputs.

For parameter sweeps, parameter-shift gradients, or other groups of distinct
circuits with the same requested outputs, submit a bounded measurement batch
through one resident transport request:

```python
import flagquantum as fq
from flagquantum.remote.compute import JiudingClient

circuits = [fq.Circuit(1).ry(0, theta=value) for value in (0.1, 0.2, 0.3)]
with JiudingClient(workspace="example-resident-a100") as client:
    results = client.run_batch(
        circuits,
        target="jiuding:gpu",
        outputs=fq.expectation(fq.Z(0)),
    )

expectations = [result.expectation() for result in results]
```

Use the same batch operation for a parameter-shift gradient without adding a
gradient algorithm to the Jiuding adapter:

```python
from flagquantum.gradients import batched_parameter_shift_gradient

def build(parameters):
    return fq.Circuit(1).ry(0, theta=parameters[0])

def evaluate_batch(circuits):
    results = client.run_batch(
        circuits,
        target="jiuding:gpu",
        outputs=fq.expectation(fq.Z(0)),
    )
    return tuple(result.expectation().sum() for result in results)

gradient = batched_parameter_shift_gradient(build, parameters, evaluate_batch)
```

The current fail-closed profile supports H, X, RX, RY, RZ, and CX. Each input
parameter must directly control exactly one RX, RY, or RZ occurrence. See
`examples/remote/jiuding_parameter_shift.py` for one complete optimization
update. A live single-A100 run used one two-circuit batch call and decreased the
test energy after one update; see the
[parameter-shift evidence](../development/evidence/jiuding_parameter_shift_20260909.json).

The eight-parameter validation example sends 16 shifted circuits in one batch
per optimization step:

```bash
python examples/remote/jiuding_vqe.py \
  --workspace example-resident-a100 \
  --target jiuding:gpu \
  --steps 5 \
  --learning-rate 0.35
```

The example uses two RY layers separated by chained CX gates. The recorded
five-step A100 run used exactly one 16-circuit gradient batch per step,
decreased the energy at every update, and had a maximum gradient error below
`3.6e-7` against local autograd. See the
[entangling VQE evidence](../development/evidence/jiuding_entangling_vqe_20260909.json).

GPU runtime images must contain `gcc` and `libc6-dev`. Triton compiles its small
driver helper on first use; without that minimal toolchain, Torch-only gates can
appear healthy while the first fused gate fails. FlagQuantum never treats a
CPU fallback as a remedy for this image defect.

The batch is validated in full before its first circuit executes, accepts at
most 256 circuits, and returns ordinary `ExecutionResult` objects in input
order. Statevector batches are intentionally rejected. Per-result runtime
evidence includes the batch index, batch size, total batch time, device, result
transfer, and fallback status.
The target namespace recognizes `cpu`, `gpu`, `mlu`, `npu` and `xpu` so its
meaning remains stable as Jiuding adds adapters. Only CPU and single-GPU paths
are implemented today. MLU, NPU and XPU targets fail before any platform
mutation; they are reserved names, not capability claims.
It checks the saved resource configuration before launching. The worker requires
CUDA and exactly one visible GPU before calling `main()`. The provided
`examples/remote/jiuding_bell_gpu.py` also verifies that the result is on CUDA.
This path does not choose a sharding strategy or launch multiple nodes.

## Lifecycle and recovery

The platform requires two operations: POST experiment saves configuration; POST
job starts it. The adapter performs both and saves an exclusive receipt before
creation, then updates it after each stage. If creation or launch fails or times
out, inspect the saved experiment name/ID on the platform before any new submit.
An existing receipt is never silently overwritten to resubmit. Receipts are
experimental provider artifacts, not Stable Core serialization contracts.

For the default `submit_program()` path, retain the returned `jobId` and call
`restore_receipt(job_id)` after reconnecting. The managed receipt and result
remain in the workspace's shared project storage. Restoration only locates the
existing receipt; it does not create an experiment or Job.

For `submit()` or expert-managed `submit_program(receipt=...)`, load the local
receipt with `json.loads(Path(...).read_text())` after disconnecting and use
`client.status(receipt)` or `client.result(receipt)`. Result acceptance
requires platform `Succeed` and a shared artifact whose run_id matches the
receipt; platform success alone is insufficient. This matching prevents stale
result confusion, not adversarial tampering of shared storage. Large tensors
should be stored separately and referenced in the returned JSON.

A wait timeout leaves the job running; it never resubmits or automatically
cancels. `client.cancel(receipt)` requests that active jobs stop and does not
archive/delete experiments. Check status afterward to verify termination.
If launch is still uncertain and no jobs are visible, an empty status/cancel
result does not establish that no job exists. Queue quota availability and
scheduler delay remain platform concerns. Log streaming, automatic resume from
partial creation, multi-GPU tasks and result downloads are not implemented.

Queue detail and job-snapshot endpoints returned 403 for the test account.
The adapter uses its authorized workspace and job queries instead. Endpoint
defaults to `https://platform-multi.baai.ac.cn`; from outside a platform pod,
provide a unique `workspace=` name and credentials. Shared artifact access is
still required. No browser screenshot, access token pasted into chat, or manual
ID discovery is part of this journey.

## Live verification

On 2026-09-09 the adapter submitted a two-qubit CPU Bell-state calculation and
read its matching result from shared storage. Job
`33295f22-cce9-4c66-8782-c5fd23cea908` reached `Succeed` with 2 CPU cores, 2 GiB
memory and 0 GPUs. Probabilities were `[0.4999999701976776, 0, 0, 0.4999999701976776]`;
the complex64 state matched the expected Bell state with maximum absolute
error 0. See [the development record](../development/evidence/jiuding_bell_cpu_20260909.json).
The calculation used the example's normal `fq.Circuit` and `fq.run` path.
It validates this CPU forward task only, not training or GPU capacity.

The single-GPU request was accepted on the same date (job
`aede25bb-ef19-4083-83d3-fb4c06472c49`) but remained Pending for the 180-second
test window. It was cancelled and its terminal `Cancelled` state confirmed.
This does not establish available queue quota or successful GPU scheduling.
Separately, the final worker and GPU Bell example ran on one visible A100 40GB
in the existing development workspace: `cuda:0`, complex64, maximum state
error 0. See [GPU development evidence](../development/evidence/jiuding_bell_gpu_20260909.json).
No additional queued test was left running.

The private `flagquantum-runtime:v0.2.0-cu128-a100` image completed the same
GPU Bell task as an independent Job on 2026-09-09. Job
`f2d7cdfd-8382-44a0-b3cb-87967a0773f2` reached `Succeed` on one A100 40GB;
the run-bound result was retrieved from persistent storage and had maximum
state error 0. See [private-image GPU evidence](../development/evidence/jiuding_bell_private_gpu_20260909.json).

The private `flagquantum-runtime:v0.2.0-7925cc364c0cd7215c32eb6e3140e3f85092e570-cu128-a100` image also passed
a pure-image resident-executor check. A fresh one-A100 workspace started the
executor without any post-creation source upload, completed three Bell-state
runs with no CPU fallback, and reused one SSH channel. The two warm calls took
17.6--18.6 ms end to end; their recorded CUDA execution took 2.3--2.6 ms. The
temporary workspace was stopped after validation. See the
[resident-image evidence](../development/evidence/jiuding_warm_executor_image_20260909.json).

For repeated interactive execution, select one running workspace once and use
the stable root entry point. Calls in the same Python process reuse its resident
executor and SSH channel:

```bash
export JIUDING_WORKSPACE=example-resident-a100
```

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
state = fq.run(circuit, target="jiuding:gpu")
probabilities = fq.run(
    circuit,
    target="jiuding:gpu",
    outputs=fq.probabilities(),
)
correlation = fq.run(
    circuit,
    target="jiuding:gpu",
    outputs=fq.expectation(fq.X(0) @ fq.X(1)),
)
counts = fq.run(
    circuit,
    target="jiuding:gpu",
    outputs=fq.counts(),
    shots=1024,
)
```

Without `outputs`, this path returns an exact statevector. Probability and
expectation and sampled-output requests do not transfer the full statevector.
Numerical execution and sampling occur on the workspace GPU; requested results
are materialized in the caller process, and counts aggregation is reported as
host-side post-processing. Noise models, compiler selection and execution plans
remain unsupported and fail explicitly.
The production workspace and root entry point were validated on 2026-09-09;
see [the root-entry evidence](../development/evidence/jiuding_root_run_20260909.json).

GPU-side probability and Pauli-expectation reduction was validated with the
`v0.2.0-36ee8a65ae509a0df0d25318015a6433194c7b53-cu128-a100` image. See
[the remote-measurement evidence](../development/evidence/jiuding_remote_measurements_20260909.json).
