# Quafu SQC backend

FlagQuantum talks directly to the HTTP contract implemented by
`quafusqc.Task`; installing `quafusqc` is not required.

Create a local `.env` file without putting the token in source code:

```bash
cp .env.example .env
# Edit .env and replace the placeholder, then load it into this shell.
set -a
. ./.env
set +a
python examples/quafu_backend.py
```

`.env` is ignored by Git. Keep `.env.example` as the public template and never
put a real token in that file. Loading is explicit so importing FlagQuantum
does not unexpectedly modify process-wide environment variables.

The provider uses these Quafu SQC endpoints:

- `GET /task/verify` for token verification;
- `GET /task/status/0` for chip availability and queue depth;
- `POST /task/run/` for OpenQASM 2.0 submission;
- `GET /task/status/{tid}` and `GET /task/result/{tid}` for asynchronous jobs;
- `GET /task/cancel/{tid}` for cancellation.

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
    target="quafu:ScQ-P10",
    shots=1024,
)
```

`fq.run` compiles, packages, submits, and waits for the Quafu result. It returns
the same `fq.ExecutionResult` type as local execution; counts are available as
`result.measurement("counts")`. Use `fq.compile` or
`create_deployment_package` separately only when the compiled IR or sealed
deployment artifact must be inspected, stored, or submitted later.

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
