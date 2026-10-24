# One sampled Ising pair-charge trajectory

The user explicitly approved the five decisions in
[proposal 091](../development/API_CHANGE_PROPOSAL_091_ISING_SAMPLED_OBSERVATION.md)
on 2026-10-24 and authorized its document PR #818 to merge after complete
applicable CI/GPU. Each later PR requires separate merge authorization.

## Replaceable baseline

The [existing execution-plan candidate](../../contracts/execution-plan-v1-candidate.json)
records the approved extension as `core_source`, with
`production_implemented=false`. No production source 1.3, environment 1.1,
new IR, result class or execution path was activated by the baseline.
The exact candidate guard continues to reject unapproved changes.

Four factories in the
[consumer tests](../../tests/unit/test_ising_sampled_contract.py) own source,
planning, execution and JSON reading. Their disposable
[reference](../../tests/unit/ising_sampled_reference.py) composes existing
Core exchange validation, native option/output admission, tensor encoding,
canonical hashing, Simulation exchange/Kraus actions, terminal sampling and
ExecutionResult/MeasurementResult. It uses a `test.ising_sampled_recipe`, never
an ExecutionPlan or a production source schema, and attaches no fake plan.

The consumers use independent fixed F/R matrices, both initial channels,
noncommuting continuation and explicit seeds including both integer limits.
They compare full complex conditional densities, exact conditional outputs,
trace, purity and weighted unread continuation at atol=1e-12, rtol=0. They check
current anyon IDs, original probabilities, result ordering/name selection,
frozen source/output snapshots, private RNG, exact direct/planned/recipe replay,
mutually exclusive observations, the 215/216-byte boundary and existing profile
refusals. The 4,096 private single-index draws retain the predeclared
1,856..2,240 psi-count range; no build-specific random sequence is encoded.

[Compatibility tests](../../tests/unit/test_ising_sampled_compatibility.py)
consume eight exact unread source/plan 1.2 captures made before implementation
at `e873723bd359832de3e718235373affd19544bab`. The capture is copied from the
immutable pre-implementation evidence, never regenerated. Existing 1.1,
circuit, Rydberg and Lindblad fixture authorities remain unchanged.

After this baseline is accepted, deliver Core, Simulation and Runtime separately.
Replace only the four factories and delete the obsolete reference; retain the
independent assertions. Real native source/plan/environment version and rehashed
tamper tests join the existing codec/admission suites when those domains are
implemented. The recipe does not pretend to validate a production 1.3 payload.

## Core source delivery

Baseline #821 merged after independent authorization and complete CI/GPU.
Core now provides the fluent zero-argument `measure_pair()` builder and strict
native source codec 1.3. It reuses ProgramRecord/Operation with literal null
physical data, the existing physical verifier and canonical codec. A shared
private append guard rejects any second sampled or unread observation before
mutation. No new IR class, mutable identity cache, RNG or numerical algorithm
is introduced. Existing braid-only 1.1 and unread 1.2 bytes remain unchanged.

The candidate stage means source construction only: `production_implemented`
remains false until native numerical execution is delivered. The existing
Runtime version boundary still refuses 1.3 before numerical execution, including
rehashed attempts to place the real new source in old plans. Root signatures,
plan/environment protocols, independent baseline factories and physics
assertions are unchanged. The obsolete baseline assertion that the method was
absent is removed; real closed-admission tests replace that stage assertion.

```python
import json
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict, native_program_to_dict,
)
from flagquantum.errors import CapabilityError

source = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
payload = native_program_to_dict(source.to_ir())
print(payload["version"])
print(payload["functions"][0]["blocks"][0]["operations"][1]["opcode"])
print(native_program_from_dict(json.loads(json.dumps(payload))) == source.to_ir())
try:
    fq.plan(source, options=fq.ExecutionOptions(shots=1, seed=17))
except CapabilityError:
    print("native execution not yet admitted")
```

Output: `1.3`, `ising.measure_pair`, `True`, `native execution not yet admitted`.
The codec functions above are the existing internal source boundary, not new
root exports. This demonstrates source inspection and the honest execution
boundary; the proposed conditional-state workflow is not available yet.

## Runnable current user workflow

The existing terminal sampler provides an explicit draw today; it returns no
conditional state and is not the proposed intermediate operation:

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.runtime import sample_terminal

source = IsingBraidProgram().exchange(1)
state = fq.run(source).state
instrument = {c: [source.projector(c)] for c in ("1", "psi")}
first = sample_terminal(state, instrument, shots=1, seed=17)
repeat = sample_terminal(state, instrument, shots=1, seed=17)
print(round(float(state[0, 1, 1].real), 12))
print(torch.equal(first.value, repeat.value))
```

Expected output: `0.5`, `True`. The full current-API conditional mathematics
reference is in the approved proposal; the shorter example above makes no claim
that `fq.run` now accepts intermediate sampling.

Run the baseline with:

```bash
python -m pytest tests/unit/test_ising_sampled_contract.py tests/unit/test_ising_sampled_compatibility.py -q
```

The accepted scope remains one CPU trajectory, four sigma anyons and total
vacuum. No multiple observations, feedback, classical register, atom-loss
inference, gradient, alternate backend, Remote or hardware support is claimed.
