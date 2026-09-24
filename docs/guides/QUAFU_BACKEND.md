# Quafu backend

FlagQuantum prefers Quafu's current task API and falls back to the legacy SQC
HTTP contract when the new platform is unavailable before submission.
Installing `quafusqc` is not required.

## Configure credentials

Keep credentials outside source control. `QUAFU_API_KEY` authenticates the
current task API. `QUAFU_API_TOKEN` is optional, but is required for an
authenticated fallback to the legacy SQC platform.

```bash
export QUAFU_API_KEY=qf_replace_with_your_api_key
export QUAFU_TASK_SERVER_URL=https://quafu.com.cn/api/v1
export QUAFU_API_TOKEN=replace_with_your_legacy_token
```

`QUAFU_TASK_SERVER_URL` is optional and defaults to the production URL above.
The provider requires HTTPS and never includes either credential in task
receipts, result metadata, or exception text.

To use a local `.env` file:

```bash
cp .env.example .env
# Add QUAFU_API_KEY and QUAFU_TASK_SERVER_URL, and replace the legacy token.
set -a
. ./.env
set +a
```

`.env` is ignored by Git. Keep `.env.example` public and never put a real
credential in it. Loading is explicit so importing FlagQuantum does not modify
process-wide environment variables.

## Run on the ideal simulator

Use `quafu:sim` to exercise the current HTTP path without consuming real-device
capacity:

```python
import flagquantum as fq

result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    target="quafu:sim",
    shots=1024,
)
print(result.counts)
print(result.provenance["deployment"]["quafu_protocol"])
```

`quafu_protocol == "task_api_v1"` confirms that the preferred platform handled
the task. A successful fallback reports `sqc_legacy` instead.

## Run on a real device

Inspect the current device list before choosing a target. Python discovery uses
the new `/devices` endpoint first and falls back to legacy SQC discovery:

```python
from flagquantum.remote import QuafuProvider

provider = QuafuProvider()
for backend in provider.discover_backends():
    print(
        backend.name,
        backend.n_wires,
        backend.metadata.get("status"),
        backend.metadata.get("queue"),
    )
```

Choose a device whose status is `online`, then use its case-sensitive name after
`quafu:`. Real-device execution consumes quota and may wait in a queue.

```python
import flagquantum as fq

result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    target="quafu:Shenglian",
    shots=1024,
)

deployment = result.provenance["deployment"]
print(result.counts)
print(deployment["job_id"])
print(deployment["quafu_protocol"])
```

This direct path sends logical OpenQASM 2.0 and does not require local QSteed or
QuarkCircuit installation. The platform performs compilation and physical
routing. Counts are returned in logical-wire order (wire 0 on the left), while
raw provider metadata retains the original response. Hardware shots must be a
positive multiple of 1024.

`outputs=fq.expectation(...)` is also supported through grouped basis
measurements:

```python
result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    target="quafu:Shenglian",
    shots=1024,
    outputs=fq.expectation(fq.Z(0) @ fq.Z(1)),
)
print(result.expectation())
```

## Submit without blocking and restore later

Use `fq.submit` when a notebook or process must detach while a real-device job
is queued. Saving a receipt never saves credentials.

```python
import flagquantum as fq

job = fq.submit(
    fq.Circuit(2).h(0).cx(0, 1),
    target="quafu:Shenglian",
    shots=1024,
    name="bell-hardware-check",
)
print(job.id)
job.save("quafu-job.json")

# This can run in a later process after credentials are configured again.
restored = fq.restore_job("quafu-job.json")
print(restored.status())
result = restored.wait(timeout=1800, poll_interval=3)
print(result.counts)
```

Call `job.cancel()` to request cancellation. A task already sent to hardware may
stop being observed without refunding consumed capacity.

The published 0.2.0 release predates this direct-submission feature. Use a source
checkout containing this change until the next release is published.

For local compilation before submission, explicitly pass `compiler="qsteed"`
and install the plugin below. The service compiler option is distinct from the
root API's local compiler selection. A submitted circuit digest is not evidence
of the final circuit executed on hardware; inspect returned `transpiled` data.

## Optional local compiler plugin

The Quafu examples use the independently maintained
[FlagQuantum Compiler QSteed](https://github.com/FlagQuantum/FlagQuantum-Compiler-QSteed)
plugin. Installing FlagQuantum alone, or its `quafu` extra, does not install
this compiler plugin. The `quafu` extra supplies the optional calibration reader.

Use Python 3.12 for the verified setup below. From a FlagQuantum 0.2 checkout:

```bash
python -m pip install -e .
python -m pip install "qsteed @ git+https://github.com/BAQIS-Quantum/qsteed.git@46584efde731aea9eec27b5466919b76fe5f3184"
python -m pip install flagquantum-compiler-qsteed==0.1.0
```

The plugin requires FlagQuantum `>=0.2,<0.3` and QSteed
`0.2.3+quafu.sqc`. Install the pinned upstream build before the plugin;
PyPI QSteed `0.2.2` is not a supported substitute. The plugin itself is
[published on PyPI](https://pypi.org/project/flagquantum-compiler-qsteed/0.1.0/).

FlagQuantum discovers the installed `compiler.qsteed` entry point automatically.
Verify installation without provider credentials or a hardware submission:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
compiled = fq.compile(circuit, compiler="qsteed")
print(compiled.instructions)
```

This checks offline compilation only. Selecting `target="quafu:Baihua"` also
requires access to the selected chip's calibration; executing `fq.run` with
that target submits a real task.

## Protocol selection and fallback

The preferred task API uses `QUAFU_TASK_SERVER_URL` (default
`https://quafu.com.cn/api/v1`) and these endpoints:

- `GET /healthz` before submission;
- `POST /jobs` for idempotent OpenQASM 2.0 submission;
- `GET /jobs/{job_id}/status` and `GET /jobs/{job_id}/results`;
- `POST /jobs/{job_id}/cancel`.

Each authenticated request uses `Authorization: Bearer $QUAFU_API_KEY`. A UUID4
`client_ref` is generated for every submission. If the health check fails, or
the submission receives a definite 4xx rejection, FlagQuantum falls back
to these legacy Quafu SQC endpoints:

- `GET /task/verify` for token verification;
- `GET /task/status/0` for chip availability and queue depth;
- `POST /task/run/` for OpenQASM 2.0 submission;
- `GET /task/status/{tid}` and `GET /task/result/{tid}` for asynchronous jobs;
- `GET /task/cancel/{tid}` for cancellation.

Submissions with an explicit `target_qubits` mapping continue to use the legacy
contract because the new logical-circuit task API has no equivalent mapping
field. A timeout after `POST /jobs` is not retried against the legacy endpoint:
the server may already have accepted the idempotent task, so a cross-platform
retry could create a second hardware job.

| Submission | Selected protocol |
|---|---|
| Logical circuit, no explicit physical mapping, healthy task API | `task_api_v1` |
| Task API health check fails | `sqc_legacy` |
| Task API returns a definite 4xx rejection | `sqc_legacy` |
| `POST /jobs` has a 5xx response or ambiguous transport failure | Raise without resubmitting |
| Explicit `target_qubits` or locally precompiled mapped circuit | `sqc_legacy` |

These endpoints do not expose a compile-only request that returns the
authoritative final circuit before submission. QuarkCircuit and QSteed can
transpile locally against a calibration snapshot, but that output is only a
candidate circuit: it does not prove which circuit the Quafu control plane will
ultimately execute. The current Quafu digital-twin path therefore reports
hardware comparisons as post-execution diagnostics. A returned `transpiled`
circuit is retained as execution evidence, never presented as a circuit known
before submission.

Precompiled submission uses one explicit contract: the OpenQASM program keeps
logical indices `q[0]` through `q[N-1]`; `options.target_qubits[i]` names the
physical qubit for logical wire `i`; and `options.compiler` is `None`. The
obsolete top-level `compile` flag is neither sent nor accepted by the adapter.

For explicit backend selection:

```python
import flagquantum as fq

circuit = fq.Circuit(2)
circuit.h(0).cx(0, 1)
result = fq.run(
    circuit,
    compiler="qsteed",
    target="quafu:Baihua",
    # Optional: lock logical q[0], q[1] to these physical qubits.
    # target_qubits=(17, 18),
    shots=1024,
    name="bell calibration",
)
```

`fq.run` compiles, packages, submits, and waits for the Quafu result. It returns
the same `fq.ExecutionResult` type as local execution; counts are available as
`result.measurement("counts")`. Use `fq.compile` or
`create_deployment_package` separately only when the compiled IR or sealed
deployment artifact must be inspected, stored, or submitted later.
The name is optional, but an explicitly provided name must not be empty; Quafu
still assigns the immutable task ID.

Compilation is the pre-submission check: QSteed resolves the selected chip's
current calibration and topology, validates the requested physical-qubit
mapping, and emits a precompiled circuit. If that step fails, `fq.run` does not
create a Quafu task. After submission it waits for a terminal state and returns
the task ID, counts, target information, and deployment evidence through the
standard result object.

If `target_qubits` is omitted, QSteed selects a connected physical subgraph
from the current calibration snapshot. If supplied, its order defines the
logical-to-physical mapping and the compiler fails before submission unless the
selection exists, is unique, and is connected.

The checked-in [live execution evidence](../reference/QUAFU_LIVE_EXECUTION_EVIDENCE.md)
shows this mapping carried from the public API through the returned physical
Quafu circuit and task result.

Quafu reports queue state but not qubit capacity from the status endpoint.
FlagQuantum therefore retains the requested width in discovered profiles; the
platform compiler remains authoritative for physical topology validation.
Quafu hardware shots should be an integer multiple of 1024.

## Calibration-derived noise model

With Python 3.12+, install the optional QuarkCircuit calibration reader:

```bash
pip install -e '.[quafu]'
```

Convert a timestamped `Backend(...).chip_info` payload into a FlagQuantum
noise model for the physical qubits used by the transpiled circuit:

```python
from quark.circuit import Backend
import flagquantum.deployment as fqd

chip_info = Backend("Baihua").chip_info
noise_model = fqd.quafu_noise_model_from_chip_info(
    chip_info,
    physical_qubits=(123, 124),
    readout_confusion_matrices=(
        ((0.978, 0.022), (0.087, 0.913)),
        ((0.956, 0.044), (0.116, 0.884)),
    ),
)
```

The converter maps per-qubit T1/T2 and gate duration exactly. Because the
current `NoiseRule` contract does not distinguish gate-match wires from
channel-target wires, selected one-qubit fidelities are averaged into one
depolarizing rate that follows the acted-on logical wire. Two-qubit fidelity
conversion is deliberately unsupported and fails closed rather than applying
an incorrect one-qubit approximation. Current Baihua payloads may report zero
readout fidelities; in that case supply independently measured assignment
matrices, with task IDs and calibration time retained alongside the result.

For nonblocking Notebook submission and restart recovery, see [remote jobs](REMOTE_JOBS.md).
