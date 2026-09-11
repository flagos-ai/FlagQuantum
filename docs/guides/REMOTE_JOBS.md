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

Configure `JIUDING_AK` and `JIUDING_SK`, then choose your project, queue and image:

```python
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
print(job.id)
```

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

Restore with `fq.restore_job("jiuding-job.json")` in a later process, using the same
account and endpoint. It retrieves the existing job without resubmitting. If a
submission response is lost, the exception names a private local journal: inspect
that experiment before retrying to avoid duplicate jobs.

No background watcher is started automatically. Manual checks and explicit waits
are supported; a notebook notification widget is outside this API's scope.
