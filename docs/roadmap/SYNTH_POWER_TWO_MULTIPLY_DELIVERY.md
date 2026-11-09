# Exact multiplication by powers of two

Follow #999, merged at 57883bcda1aa6fe54b3e3989f65c5de44c58c9a0 after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-POWER-TWO-MULTIPLY-20261101.md).

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth


def scale(x: int) -> int:
    return x * 256


program = synth.synthesize(
    scale,
    inputs={"x": synth.UInt(1)},
    input_state=synth.InputState.uniform(),
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
print(program.report["resources"])
print(program.execution.state_bytes)
```

Run `python -m examples.synthesis.tutorials.arithmetic.power_two_multiply.main`. Pairs (0,0) and
(1,256) each have probability 1/2. Users write ordinary multiplication and
supply quantum domains, preparation and budgets. Compiler selects the bit
mapping, gates and cleanup; users do not supply shifts, scratch or line numbers.

## Actual capacity and implementation boundary

CPU complex128 / Torch 2.13.0, complete uniformly prepared workload:

| Expression/domain | Schedule | Allocated/used | Hidden workspace | Operations | Depth | Planned state |
| --- | --- | --- | --- | --- | --- | --- |
| x*4, two-bit x | Preceding generic | 14/12 | 8 | 24 (2 H, 18 CX, 4 CCX) | 18 | 256 KiB |
| x*4, two-bit x | Shifted copy | 10/10 | 4 | 10 (2 H, 8 CX) | 4 | 16 KiB |
| x*256, one-bit x | Preceding generic | 28/20 | 18 | 14 (1 H, 13 CX) | 6 | 4 GiB |
| x*256, one-bit x | Shifted copy | 19/19 | 9 | 12 (1 H, 11 CX) | 4 | 8 MiB |

The preceding results are measured from checked plans at the merged baseline,
without executing a 4 GiB state. Both new rows execute and verify their full
states under 64 MiB. The two-bit x*4 probe gives (0,0), (1,4), (2,8) and (3,12)
each with probability 1/4. Allocated wires, including unused scratch in the preceding
plan, determine dense capacity. These are exact workload resource costs, not
runtime speedup, optimality, hardware or process peak-memory claims.

Only Compiler `_construction/lowering.py` changes in production. A classical
constant 2**k with k>=1 and a positive complete unsigned operand bound admits
a shifted copy into existing result storage. Low k bits remain zero. Both
operand orders use current physical bindings; generic multiplication scratch
is skipped only for the proven case. Result allocation, arbitrary XOR copy-out,
exact inverse, retain/recompute, zero-pool ownership, event/resource accounting,
checked Program and execution preflight remain shared. No new API, opcode, IR,
arithmetic kernel, allocator, scheduler, executor or dependency; no direct
output fusion.

Zero, one, non-power constants and quantum-quantum operands retain the generic
kernel. Derived zero-bound operands also retain it, avoiding a truncated or
empty shifted destination for their one-bit zero result. No proof uses basis
preparation, samples or user assertions. Complete-domain width and physical,
operation and memory budgets remain mandatory.

## Verification boundaries

The 26 new cases pass in 1.22 seconds: both operand orders and schedules, every
two-bit input and all sixteen arbitrary XOR targets on noncontiguous bindings;
complex reference-entangled targets, shared dependencies and rematerialization;
Python/native producers under both objectives, nested phase cancellation and
hidden work zero; actual scratch reservation for nine constants; derived-zero
fallback, quantum/non-power refusal to specialize from basis zero; and memory
and 256-bit guards. A 1 MiB budget refuses the new wide example before state
allocation, while 64 MiB allows real execution.

Initial collection failed because the fixture imported PlanningError from
Runtime instead of its authoritative errors module. The subsequent run had
25 passes and one fixture failure because a width guard was expected to raise
SynthesisError; its existing contract raises HybridVerificationError. Both
fixtures were corrected without changing production exceptions or exports.
The superseded regression process was stopped before rerunning the corrected
snapshot; its partial run is not recorded as passing.

Strict mypy 2.4 passes 879 production/example sources. Black 26.10.1, Ruff,
architecture, unchanged public API snapshot, dependency policy, generated docs,
maturity, language, links and whitespace pass. Documentation/ownership unit
cases: 31 passed. Original source/license digests: seven passed.

The required default smoke/unit run records 5030 passed, 121 skipped and the
same three existing local Simulation QFT fusion assertion failures (44 state
applies instead of 28) in 206.46 seconds. It stopped at its configured
three-failure limit. Simulation and its assertions are unchanged; the complete
default tier is not green. Log: /private/tmp/fq-power-two-default.log.

The complete selected Compiler construction, synthesis workflow, Services plan
and related API regression passes all 2267 cases in 510.35 seconds, including
the 26 new cases and the preceding Python powers/carry-free addition suites.
The two source-overload mypy-cache fixtures are excluded; whole-package strict
typing is recorded above. Log: /private/tmp/fq-power-two-regression.log.
Original source/license bytes, four-original cohort, trials 0/2 and
release_authorized=false remain unchanged. No corpus-wide, release, notebook
trial or hardware claim is added.
