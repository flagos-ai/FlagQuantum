# Remote compute

This directory owns experimental adapters to external CPU/GPU job systems.

Use the categorized public entry point:

```python
from flagquantum.remote.compute import JiudingClient
```

Jiuding currently supports explicit development-workspace creation and
start/stop control, one CPU or single-GPU task instance, automatic workspace
context discovery, direct program submission with workspace-managed artifacts,
status, waiting, shared JSON results, stopping active jobs, and a loopback-only resident statevector
executor for low-latency work in an already-running workspace. The resident
path supports exact measurements, sampled outputs, and bounded measurement
batches in one transport request. It uses the standard library and direct
HTTPS/SSH calls.
Compute selection uses `jiuding:<chip-type>` or
`jiuding:<chip-type>/<model>`. CPU and GPU are implemented; MLU, NPU and XPU
are recognized reserved names that fail closed until real adapters exist.
It does not own circuit execution, numerical backend selection, gradients,
distributed launch, or the QPU shots/counts contracts. No Stable Core exports
are added. These adapter-specific interfaces are not frozen.

Start with `JiudingClient` from `flagquantum.remote.compute` and
`examples/remote/jiuding_submit_program.py`. `submit_program()` accepts a
Circuit directly and manages its source and result artifacts through the
selected workspace. The lower-level `submit()` accepts a shared user script
whose `main()` returns a JSON-serializable value.

Read `docs/guides/JIUDING.md` for the supported journey and limits. Run
`python -m pytest tests/team/remote/test_jiuding.py
tests/team/remote/test_workspace_executor.py -q` for offline behavior tests.
No test in those files creates real tasks. The Bell example provides a
small CPU numerical check for live acceptance; `jiuding_bell_gpu.py` checks
CUDA execution with one visible GPU; `jiuding_workspace_bell.py` exercises the
resident path. A new provider-wide contract, `flagquantum.remote` root export,
or distributed claim
requires a separate reviewed change.
