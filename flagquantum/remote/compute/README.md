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
