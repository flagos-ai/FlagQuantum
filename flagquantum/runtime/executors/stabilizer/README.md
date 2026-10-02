# Stabilizer runtime

Serve computational-basis sampling for a Clifford circuit whose state is a Pauli
tableau rather than an amplitude store. The tableau grows with the square of the
wire count, so this is the only representation that reaches wire counts a dense
state cannot hold.

This package owns the plan-aware lifecycle of that route: binding a plan's
measurement request to the engine, adapting the engine to the Runtime
measurement contract, and keeping the tableau's cost visible against a declared
memory limit. The numerical work belongs in
[`flagquantum/simulation/stabilizer`](../../../simulation/stabilizer/README.md);
gate classification, tableau algebra, and sampling are not owned here. The engine
is selected explicitly with `fq.ExecutionOptions(mode="stabilizer")`; automatic
selection never chooses it.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Sampling request binding and refusal | [sampling.py](sampling.py) `run_stabilizer_mode` |
| Adapter to the measurement contract | [sampling.py](sampling.py) `StabilizerSamplingTarget` |
| Mode vocabulary and the tableau estimate | [planner/execution_policy.py](../../planner/execution_policy.py) |
| Planning-time refusals | [planner/__init__.py](../../planner/__init__.py) `_require_stabilizer_request` |
| Execution dispatch and the result stamp | [execution.py](../../execution.py) |

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
