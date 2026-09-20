# Submit and resume remote jobs

This API requires the development version; it is not in PyPI 0.2.0.
`fq.run()` still waits for a result. Use `fq.submit()` when a Notebook should
remain available while a remote task is queued or running. Submission waits only
for preparation and the provider's acknowledgement, not for execution completion.

## Quafu

Configure `QUAFU_API_TOKEN` in the local environment, then submit once:

```python
import flagquantum as fq

job = fq.submit(fq.Circuit(2).x(0), target="quafu:Baihua", shots=1024)
job.save("quafu-job.json")
print(job.id)
```

Run a later cell whenever you want to check progress:

```python
state = job.status()
print(state, job.raw_status)
if state == "succeeded":
    result = job.result()
    print(result.counts)
```

`status()` queries once. Normalized states are `queued`, `running`, `succeeded`,
`failed`, `cancelled`, and `unknown`. Quafu's `Transpiled` means compilation is
finished and maps to `queued`; it does not mean hardware execution is finished.
`raw_status` preserves the last queried provider state. Unknown or missing states
never count as success. Calls still take time for their individual network I/O.

`result()` does not poll: it raises `ExecutionError` unless the job has succeeded.
To intentionally block the current cell, use `result = job.wait(timeout=600)`.
A timeout leaves the remote task running. Do not rerun the submission cell to
check progress. `job.cancel()` requests cancellation; query the state afterwards
to confirm it. Interrupting a cell or closing a Notebook is not cancellation.

The first detached Quafu journey supports full-register counts. Omit `compiler`
to use service compilation, or pass `compiler="qsteed"` for local compilation.
Grouped expectations remain available through `fq.run()`, but are not accepted
by `submit()` because they can produce several hardware tasks.

## Restore after restarting the Notebook

Keep the receipt private. It contains job identity and decoding context, not
credentials. It is JSON, not pickle, and saving never overwrites an existing file.
Configure credentials again after restarting, then run:

```python
import flagquantum as fq

job = fq.restore_job("quafu-job.json")
print(job.status())
```

Restoration never resubmits. A Quafu receipt preserves the submission artifact
identity, compiler mode, output name and logical bit ordering. Remote results
remain subject to provider retention and account permissions. Receipt metadata is
local context, not cryptographic proof of the circuit executed by hardware.
If submission itself loses its network response, reconcile with the provider
before retrying: the server may already have accepted the job.

## Jiuding native jobs

Choose one complete Jiuding credential source. Explicit credentials are the
appropriate path when each user supplies a different pair in a shared
JupyterLab. Local single-user processes may use environment variables, while a
Jiuding workspace can use the platform-injected files.

| Context | Configuration | Calls |
| --- | --- | --- |
| Per-user notebook or application session | Construct `JiudingCredentials(access_key=..., secret_key=...)` | Pass `credentials=` to `JiudingClient`, `fq.submit()` and `fq.restore_job()` |
| Local single-user shell | Set both `JIUDING_AK` and `JIUDING_SK` | Omit `credentials`; FlagQuantum reads the environment |
| Jiuding workspace | Let the platform inject `/etc/accesskey/user-ak` and `/etc/accesskey/user-sk` | Omit `credentials`; FlagQuantum reads both files |

`JiudingCredentials` accepts ordinary strings; it does not depend on
`getpass`. The placeholders below make that API explicit:

```python
import flagquantum as fq
from flagquantum.remote.compute import JiudingClient, JiudingCredentials

credentials = JiudingCredentials(
    access_key="YOUR_JIUDING_AK",
    secret_key="YOUR_JIUDING_SK",
)

# Copy project, queue and image values from one returned entry.
resources = JiudingClient(credentials=credentials).list_resources()
print(resources)

job = fq.submit(
    fq.Circuit(2).h(0).cx(0, 1),
    target="jiuding:gpu",
    project="YOUR_PROJECT_SET.YOUR_PROJECT",
    queue="YOUR_QUEUE",
    image="YOUR_FLAGQUANTUM_IMAGE",
    outputs=fq.counts(),
    shots=1024,
    credentials=credentials,
)
job.save("jiuding-job.json")
print(job.id)
```

A submit-ready entry has this shape:

```python
[
    {
        "project": "YOUR_PROJECT_SET.YOUR_PROJECT",
        "queue": "YOUR_A100_QUEUE",
        "accelerator_model": "NVIDIA_A100-SXM4-40GB",
        "images": [
            "flagquantum-runtime:YOUR_VERSION-cu128-a100",
        ],
        "submission_supported": True,
    }
]
```

Copy `project`, `queue`, and one compatible entry from `images` into
`fq.submit()`. The list is sorted by project and queue, and each image list is
sorted. Private-image visibility does not prove that an image contains
FlagQuantum; use the runtime image published or approved by the project
administrator.

An Active queue that the current adapter cannot submit to remains visible and
explains why:

```python
[
    {
        "project": "YOUR_PROJECT_SET.YOUR_PROJECT",
        "queue": "YOUR_QUEUE",
        "accelerator_model": None,
        "images": [],
        "submission_supported": False,
        "reason": "queue must expose exactly one high-priority resource configuration",
    }
]
```

An empty list means the account has no visible Active queue. It does not mean
authentication succeeded with newly provisioned resources; project membership
and quota still come from Jiuding.

Replace the AK/SK placeholders at runtime and never commit or save real values in
a notebook. `getpass()` remains an optional way to collect the same two strings
without displaying or storing them in a cell. The resulting
`JiudingCredentials` object behaves identically.

For a trusted local single-user shell, configure the fallback instead:

```bash
export JIUDING_AK="YOUR_JIUDING_AK"
export JIUDING_SK="YOUR_JIUDING_SK"
```

FlagQuantum reads the current process environment. It does not parse a `.env`
file automatically; a launcher or environment loader must export those values
before the Python process starts.

Then omit `credentials` from all calls:

```python
resources = JiudingClient().list_resources()
job = fq.submit(
    fq.Circuit(2).h(0).cx(0, 1),
    target="jiuding:gpu",
    project="YOUR_PROJECT_SET.YOUR_PROJECT",
    queue="YOUR_QUEUE",
    image="YOUR_FLAGQUANTUM_IMAGE",
    outputs=fq.counts(),
    shots=1024,
)
job.save("jiuding-job.json")
restored = fq.restore_job("jiuding-job.json")
```

Both environment variables are required; FlagQuantum rejects a partial pair.
Supplying `credentials=` takes precedence and bypasses environment and injected
file discovery completely. FlagQuantum never combines values from two sources.

Replace every `YOUR_...` value with an identifier from the signed-in user's
Jiuding project. These are Jiuding platform resource names, not names invented
or provisioned by FlagQuantum:

- `project` is the Jiuding project identifier in `project-set.project` form.
- `queue` is the exact compute-queue name assigned to that project.
- `image` is an image reference available from that queue's private image
  catalog and containing a compatible FlagQuantum program executor.
- `target="jiuding:gpu"` is the FlagQuantum target selector; the chosen Jiuding
  queue determines the concrete GPU model. Use `jiuding:gpu/<model>` only when
  an exact model match is required.

Users must obtain project membership, queue access and the image reference from
their Jiuding administrator or Jiuding console. One user's values generally do
not work for another account. `list_resources()` uses Jiuding's read-only
discovery APIs and returns only names needed for submission, the accelerator
model, private image names and whether the queue shape is supported. It does not
return platform IDs, tokens or image registry URLs. An empty list means the
account has no visible Active queue; it does not create access or quota. A listed
image must still contain a compatible FlagQuantum program executor.

This path uses native HTTP jobs, without SSH, workspace pods or source uploads.
The image must contain a compatible FlagQuantum program executor. You can set
`JIUDING_PROJECT` instead of passing `project`; omit `queue` only when exactly one
active queue is available. The current adapter requires a queue with one
high-priority resource configuration and uses its private image catalog.
Synchronous `fq.run()` retains its resident-workspace execution path.

The circuit is carried in a command bounded to 64 KiB. Results travel through
chunked authenticated job logs, bounded to 8 MiB and checked for completeness
and SHA-256 integrity. This is intended for small circuits and compact outputs;
prefer counts or expectations over large statevectors. Log retention and account
permissions determine how long results remain retrievable. The checksum detects
transport corruption; it does not attest hardware execution. Logs may become
readable shortly after completion; `wait()` retries incomplete results until its
deadline, while `result()` reports that no result is available yet.

Restore with `fq.restore_job("jiuding-job.json", credentials=credentials)` in a
later cell. After a kernel restart, create a new credential object and pass it
again. The receipt never contains AK, SK or an access token.
Restoration retrieves the existing job without resubmitting. If a
submission response is lost, the exception names a private local journal: inspect
that experiment before retrying to avoid duplicate jobs.

Do not configure long-lived Jiuding credentials in a Quafu JupyterLab shared by
multiple users. Session-level injection prevents ordinary environment, receipt
and logging leakage; it is not a hard tenant boundary when kernels share one Unix
account. A production multi-user deployment should use isolated kernels or a
broker that exchanges the signed-in user identity for a short-lived, scoped
Jiuding token, keeping long-lived AK/SK outside notebooks.

No background watcher is started automatically. Manual checks and explicit waits
are supported; a notebook notification widget is outside this API's scope.
