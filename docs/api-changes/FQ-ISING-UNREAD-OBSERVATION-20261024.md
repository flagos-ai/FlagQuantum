# Ising unread pair-charge observation

Proposal: [089](../development/API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md).
The five decisions were explicitly approved on 2026-10-24. Each PR still requires
its own merge authorization and complete applicable CI/GPU.

## Shared baseline and replacement

The [existing execution-plan authority](../../contracts/execution-plan-v1-candidate.json)
records the approved narrow extension. Braid-only source/plan 1.1 is unchanged;
exactly one ordered `dephase_pair()` uses native source/plan 1.2. Root signatures,
options and result fields remain unchanged.

The baseline landed before implementation in #798. Its test-only source and
recipe composed existing exchange/Kraus actions without claiming production 1.2
support. Core #804 added the bounded source operation and strict encoding;
Simulation #807 added the channel to the existing ordered action kernel.
Runtime now admits matching native plan/source 1.2 through the existing codec,
identity, preflight, replay and result authorities. It implements no numerical
channel, solver or controller.

The four [consumer factories](../../tests/unit/test_ising_unread_contract.py)
now use real source construction, planning, execution and JSON reading. The
reference source/recipe/execution implementations have been deleted. Independent
consumer assertions and fixed F/R oracles remain unchanged at atol=1e-12, rtol=0.
They cover both initial channels, zero-weight branches, noncommuting continuation,
full complex density, trace, mixed-state echo, named outputs, snapshots, exact
replay and unchanged Torch/Python RNG.

[Compatibility consumers](../../tests/unit/test_ising_unread_compatibility.py)
retain six exact braid-only source/plan captures made before implementation at
`4be2375eff47c112ce8d8e24f8206ed5592130d8`. JSON strings and hashes were not
regenerated. Existing circuit, Rydberg and Lindblad captures remain unchanged.

## Runnable native workflow

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

program = IsingBraidProgram().exchange(1).dephase_pair().exchange(1, clockwise=True)
output = fq.expectation(program.projector("psi"), name="psi")
plan = fq.plan(program, outputs=output)
replay = fq.ExecutionPlan.from_json(plan.to_json())
result = fq.run(replay)
print(plan.schema_version)
print(round(float(result.expectation("psi")[0]), 12))
print(torch.equal(result.state, fq.run(plan).state))
```

Executed output: `1.2`, `0.5`, `True`. The result state has shape `(1,2,2)`
and is the mixed density I/2; the channel removes the coherence required by the
inverse-braid echo. No outcome was selected or sampled.

Run the replacement, old-byte and admission tests with:

```bash
python -m pytest tests/unit/test_ising_unread_contract.py tests/unit/test_ising_unread_compatibility.py tests/unit/test_ising_unread_runtime.py -q
```

## Preserved limits

CPU complex128, four sigma anyons with total vacuum, zero or one ideal unread
observation of current positions 0/1. No gradients, shots on `fq.run`, noise,
alternate backend, sampled intermediate branches, feedback, classical bits,
timestamps, Remote, hardware or other anyon sectors. Existing explicit terminal
sampling remains separate. Source/plan mismatch and relabelled braid-only 1.2
sources fail even after rehashing. Memory and environment refusals remain in the
existing preflight path. No qubit analysis or physical duration is fabricated.
