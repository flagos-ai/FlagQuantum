# Range-proven positive constant addition

Follow #998, merged at 4d5771af4585144d5247a243890235ad4318c298 after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-CARRY-FREE-ADD-20261101.md).

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth


def offset(x: int) -> int:
    return x + 256


program = synth.synthesize(
    offset,
    inputs={"x": synth.UInt(1)},
    input_state=synth.InputState.uniform(),
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
print(program.report["resources"])
print(program.execution.state_bytes)
```

Run `python -m examples.synthesis_carry_free_add`. The pairs (0,256) and
(1,257) each have probability 1/2. Users supply the computation, quantum domain,
preparation and memory budget; Compiler selects gates and cleans its work bits.
No additional annotation, flag, line number, carry register or bit decoding is
required.

On CPU complex128 / Torch 2.13.0 the complete prepared example uses 19 allocated
and used qubits, nine hidden workspace bits, 14 operations (one H, two X, eleven
CX), depth four and an 8 MiB planned state buffer. The preceding ripple schedule
required 29 qubits and an 8 GiB state buffer for this domain and expression.
That earlier buffer requirement was measured by preflight, without executing an
8 GiB state. This is a capacity comparison for one exact workload, not a runtime
speedup, optimality, hardware, or process peak-memory claim.

## Architecture and limits

Only Compiler `_construction/lowering.py` changes in production. The scheduler
checks a positive classical constant against the conservative low-bit mask of
the other operand's complete inferred unsigned bound. Disjoint bits prove that
addition equals XOR. A proven operation allocates only its result and reuses
existing X/CX copy logic. Current operand bindings are read when rematerializing.
Both operand orders work. Zero constants, overlapping masks and two quantum
operands keep the existing ripple implementation. Basis preparation, samples or
user assertions never establish the proof.

The same result copy-out, exact inverse, zero-pool ownership, retain/recompute
schedules, event/resource accounting, checked Program and execution memory
preflight remain authoritative. This is not direct output fusion. No new public
API, opcode, kernel, representation, scheduler, executor or dependency is added.

## Verification boundaries

The 18 new cases cover both operand orders and schedules, all four two-bit
inputs and all eight initially arbitrary XOR targets on noncontiguous bindings;
complex reference-entangled targets, shared dependencies and actual
rematerialization; Python/native producer replacement under both objectives;
nested scope phase cancellation and cleanup; full-domain proof boundaries.
Full-state comparisons check hidden workspace zero and exact relative phases.
The original 65-case Python powers suite also passes: 83 cases in 31.39 seconds.

The old wide-constant memory refusal now uses a 1 MiB budget, below the new
8 MiB requirement. A separate public 64 MiB positive execution proves the
capacity improvement rather than simply removing the old refusal. The initial
new-suite run had 12 passes and six test errors because a fixture inspected
CandidateRow.circuit; correcting it to the existing public Program.circuit
resolved them without a production API change.

Strict mypy 2.4 passes all 879 production/example sources. Black 26.10.1,
Ruff, architecture boundaries, unchanged public API snapshot, dependency policy,
documentation generation, maturity, language and local-link checks pass. The
31 documentation/ownership unit cases and seven original source/license digest
checks pass.

The required default smoke/unit run records 5030 passed, 121 skipped and the
same three local QFT fusion assertion failures in 204.76 seconds. It stopped at
the configured three-failure limit; the entire default tier is not green. These
failures still expect 28 state applies and observe 44 in Simulation, which this
change does not modify. Log: /private/tmp/fq-carry-free-default.log.

The broader Compiler construction, synthesis workflow, Services plan and related
API regression passes all 2158 cases in 476.98 seconds. This run excludes the
separately passed 83 cases and the two source-overload mypy-cache fixtures;
strict whole-package typing is recorded above. Log:
/private/tmp/fq-carry-free-integration.log.

Original source/license bytes, four-original cohort, trials 0/2 and
release_authorized=false remain unchanged. No general Classiq compatibility,
release, notebook-trial or hardware claim is added.
