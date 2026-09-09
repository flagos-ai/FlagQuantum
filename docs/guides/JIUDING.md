# Jiuding CPU and single-GPU tasks (experimental)

Run from a Jiuding development workspace with injected AK/SK credentials.
The client reads `/etc/accesskey/user-ak` and `user-sk` automatically; explicit
`JIUDING_AK` / `JIUDING_SK` environment values override those files. Tokens stay
in memory and refresh according to the server expiry. Requests use HTTPS,
reject redirects, have a 20-second socket timeout, and are not retried blindly.
No CLI installation or manual project/queue IDs are required.

## Development workspaces

FlagQuantum can also create and control a Jiuding development workspace. The
client uses an existing visible workspace only to identify the project, queue,
cluster and storage context; the new workspace is a separate platform resource.

```python
from flagquantum.remote.compute.jiuding import JiudingClient

client = JiudingClient(workspace="fq-image-build-upload")
created = client.create_workspace(
    "flagquantum-dev",
    target="jiuding:gpu/NVIDIA_A100-SXM4-40GB",
    image="flagquantum-runtime:v0.2.0-cu128-a100",
    image_region="PRIVATE",
    cpus=4,
    memory_gib=16,
)
print(client.workspace_state(created["id"]))
```

Creation starts the workspace but does not enable privileged mode, Jupyter
Cloud IDE or automatic snapshots. It is an explicit billable infrastructure
operation and is never triggered by `fq.run()`. Use
`client.stop_workspace("flagquantum-dev")` to stop it without saving a
container snapshot and `client.start_workspace("flagquantum-dev")` to restart
it. Source code and durable results must remain on mounted storage; stopping a
workspace does not make its container filesystem durable.

```python
from pathlib import Path
from flagquantum.remote.compute.jiuding import JiudingClient

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
    image="flagquantum-runtime:v0.2.0-cu128-a100",
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
An existing receipt is never silently overwritten to resubmit. The receipt is
a local experimental JSON artifact, not a Stable Core serialization contract.

After disconnecting, load the receipt with `json.loads(Path(...).read_text())`
and use `client.status(receipt)` or `client.result(receipt)`. Result acceptance
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
