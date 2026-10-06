# Stabilizer runtime

Serve measurement outcomes and exact Pauli expectations for a Clifford circuit
whose state is a Pauli tableau rather than an amplitude store. The tableau grows
with the square of the wire count, so this is the only representation that reaches
wire counts a dense state cannot hold.

A program that carries a noise channel is served here too. A scene-level
`NoiseModel` reaches this route already lowered into positioned channel
instructions the way the density-matrix route receives one, because the lowering
is performed by the dispatcher in [`execution.py`](../../execution.py) -- the
Runtime module that is permitted to hold a Compiler dependency -- before the
plan-aware lifecycle here is called. An inline channel and an equivalent model
therefore reach the engine as the same program, and the engine draws the Pauli
frames the channel defines. Which of the engine's two sampling entry points
applies is read from the program rather than passed alongside it, so this route
cannot hold a program and a claim about it that disagree. A noisy program is
sampling-only: the planner refuses an exact-value request on one by name, and
refuses a channel the engine cannot frame while planning, before a plan is
published.

This package owns the plan-aware lifecycle of that route: binding a plan's
measurement requests to the engine, adapting the engine to the Runtime
measurement contract, and keeping the tableau's cost visible against a declared
memory limit. The numerical work belongs in
[`flagquantum/simulation/stabilizer`](../../../simulation/stabilizer/README.md);
gate classification, tableau algebra, sampling, and the Pauli conjugation the
readout performs are not owned here. Lowering a model into positioned
instructions belongs to
[`flagquantum/compiler/noise.py`](../../../compiler/noise.py); it is called from
the dispatcher named above rather than reimplemented, and not from this package,
whose subdirectory is deliberately absent from the Runtime-to-Compiler allow list
in `architecture.toml`. The engine is selected explicitly with
`fq.ExecutionOptions(mode="stabilizer")`; automatic selection never chooses it.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Measurement request binding and refusal | [route.py](route.py) `run_stabilizer_mode` |
| Adapter to the measurement contract | [route.py](route.py) `StabilizerTarget` |
| Mode vocabulary and the tableau estimate | [planner/execution_policy.py](../../planner/execution_policy.py) |
| Planning-time refusals, including the noisy-program census | [planner/__init__.py](../../planner/__init__.py) `_require_stabilizer_request` |
| Execution dispatch, the model lowering, and the result stamp | [execution.py](../../execution.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/unit/test_stabilizer_execution_mode.py \
  tests/integration/test_stabilizer_execution_mode.py -q
```

Any change to the mode vocabulary is a Stable Core contract change: the candidate
contract `contracts/execution-options-v1-candidate.json` and
`flagquantum/runtime/options.py::_MODES` must move together, and
`tests/unit/test_execution_options_candidate.py` pins the pair.
