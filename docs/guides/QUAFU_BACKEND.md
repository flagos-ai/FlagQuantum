# Quafu SQC backend

FlagQuantum talks directly to the HTTP contract implemented by
`quafusqc.Task`; installing `quafusqc` is not required.

## Install the compiler plugin

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

## Configure and run

Create a local `.env` file without putting the token in source code:

```bash
cp .env.example .env
# Edit .env and replace the placeholder, then load it into this shell.
set -a
. ./.env
set +a
python examples/remote/quafu_bell.py
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
