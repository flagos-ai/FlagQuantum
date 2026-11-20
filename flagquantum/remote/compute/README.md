# Remote compute

Run classical workloads through external batch systems and reusable workspaces.
This package owns job lifecycle and workspace connections; Runtime owns circuit
execution, and directly controlled devices belong in Compute.

Use the categorized public entry point:

```python
from flagquantum.remote.compute import JiudingClient
```

## Choose the execution path

- **Program batch job:** use `JiudingClient.submit_program` and the
  [program example](../../../examples/remote/jiuding_submit_program.py). It
  manages source and result artifacts through the selected workspace.
- **Script batch job:** use `JiudingClient.submit` and the [submission example](../../../examples/remote/jiuding_submit.py).
  A shared user script defines `main()` and returns a JSON-serializable result.
- **Repeated circuit execution:** use `fq.run(..., target="jiuding:gpu")` with a
  configured running workspace. The resident executor reuses its connection and
  supports bounded measurement requests.

Follow the [Jiuding guide](../../../docs/guides/JIUDING.md) for credentials,
shared code, compatible images, resource selection, and supported outputs.
The adapter interfaces are experimental; reserved chip names do not imply
implemented hardware support.

Native project jobs submitted by `fq.submit(..., project=...)` recover result
chunks through `POST /api/v1/telemetry/logs/query`. The adapter queries application
logs for the submitted job and cluster, follows forward cursors, and verifies
chunk completeness and SHA-256 before decoding the result. It uses integer
telemetry enums (`workload_type=1`, `log_source=1`, `direction=2`). Grafana
datasource discovery is no longer required. Both gateway camelCase pagination
fields and the documented snake_case fields are accepted. Existing receipts
resolve their cluster from the authenticated project queue when it was not recorded.
Queries cover at most the last 30 days; results outside the platform's log
retention period cannot be recovered from logs. Missing chunks remain a pending
result, so waiting never submits a replacement job.

## Change and verify

Start in [jiuding.py](jiuding.py) for the client,
[_worker.py](_worker.py) for batch results, or
[_workspace_executor.py](_workspace_executor.py) for resident execution.
The `execution.py` module owns the `fq.run` boundary validation and resident
client reuse; `JiudingClient.run` owns only workspace execution.
Run offline checks from the repository root:

```bash
python -m pytest tests/team/remote/test_jiuding.py \
  tests/team/remote/test_workspace_executor.py -q
```

Verify task-bound results, workspace restart, timeout behavior, cancellation,
and connection cleanup. Keep QPU counts contracts, distributed launch policy,
and numerical algorithms out of the adapter.
