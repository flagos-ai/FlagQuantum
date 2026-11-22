# CI worker exit diagnostics

Several compiler reference tests lost xdist workers shortly after a traceback
dump. A dump is diagnostic rather than a process timeout. An isolated complete
Rainbow reference passed on the affected Linux host and locked image after the
same 300-second dump. The original CI exits remain unexplained.

The existing `tools.pytest_shard` plugin now emits `CI worker exit:` followed
by a JSON object when xdist reports a worker error. It records the worker ID,
transport error, available local process exit status, and negative-status POSIX
signal number. It polls without waiting, killing, retrying, or restarting the
worker. Missing local process handles and processes not yet reaped are reported
as null; null does not exclude a signal or establish successful termination.

The implementation inspects execnet's existing local Popen transport within
the CI tooling boundary. No new dependency is introduced. Other transports
retain explicit unknown status. A future execnet transport change may require
updating this inspection; the worker error and pytest failure remain intact.
The hook is optional so serial pytest still loads the plugin without xdist.

Assertions, collection, coverage, resource budgets, timeouts and admission gates
are unchanged. This is evidence collection, not a fix for the original crash.

Run the process-backed scenarios and existing shard tests with:

```bash
python -m pytest tests/unit/test_pytest_worker_exit_diagnostics.py tests/unit/test_pytest_shard.py -q -p tools.pytest_shard
```

Expected: ordinary subprocess exit 17 and POSIX SIGTERM are distinguished, a
missing handle remains unknown, clean workers emit no failure, and existing
shard behavior passes. This supports the unchanged user compilation workflow:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

The [locked Linux image check](evidence/ci-worker-exit-diagnostics.json) also
injected SIGTERM into a real xdist worker. Its inner pytest deliberately failed,
while the outer conformance check passed: the new record retained exit status
`-15` and signal `15`. This proves signal capture for that scenario, not the
cause of earlier compiler worker exits. Complete output and launcher are retained.

The change affects CI failure evidence only. No public API, capability maturity,
SDK comparison conclusion, or host configuration changes.
