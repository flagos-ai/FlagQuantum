# CPU vertical slice

Status: implemented on the integration branch.

## Golden path

```text
fq.run(program, options=options)
  -> Compiler builds an immutable ExecutionPlan
  -> Runtime validates and executes that exact plan
  -> CPU Platform Provider resolves the requested device
  -> Simulation executes the compiled CircuitIR
  -> Runtime returns ExecutionResult with actual-path diagnostics
```

The path is deliberately limited to local PyTorch statevector execution with
`world_size=1`. It preserves the stable `fq.run`, `fq.plan`, `ExecutionOptions`,
`ExecutionPlan`, and `ExecutionResult` interfaces.

Runtime owns plan validation, device selection, dispatch, and result assembly.
Simulation owns the numerical statevector call. The CPU Platform Provider owns
device availability and identity. Compiler remains the only stage that changes
the program.

`simulation.statevector` now owns the local numerical loop. `Circuit.state()`
is a thin public facade, so existing users and Runtime callers do not change.
For this first physical migration, `Circuit` still owns the initial-state and
lifecycle cache containers; moving those containers is separate work and must
not create a second public execution contract.

The result reports the selected device, platform provider, simulation engine,
and `single_device_fast_path` semantics. An explicit CPU request reports
`cpu_fallback_used=False`; this is a local correctness path, not distributed or
accelerator evidence.

## Run and modify

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py -q
```

Start with `flagquantum/runtime/execution.py` for dispatch or result assembly,
`flagquantum/simulation/statevector.py` for the numerical entry, and
`flagquantum/runtime/platforms/pytorch.py` for CPU lifecycle behavior. Changes
to one concern should normally remain in its owning domain.

GPU, distributed execution, noise, MPS, tensor networks, training, QPU, and new
public contracts are outside this slice.
