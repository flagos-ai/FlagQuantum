# Positive literal powers in ordinary Python

Follow #997, merged at d78eb7f2a2ca890ecd9263a368c7fa35bf8fb3ed after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-PYTHON-POWERS-20261101.md).

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth


def cube(x: int) -> int:
    return x**3


program = synth.synthesize(
    cube,
    inputs={"x": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
    memory_limit_bytes=256 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
```

Run `python -m examples.synthesis.tutorials.arithmetic.python_powers.main`. The pairs (0,0), (1,1),
(2,8) and (3,27) each have probability 1/4. Users keep ordinary Python, declaring
quantum domains, preparation and budgets without a decorator, native source,
temporary registers, multiplication chains, gates or bit decoding.

## Architecture and limits

Only frontend_python.py changes in production. A small private arithmetic
routine reuses existing constant folding, full-domain checks and emission for
+/* and power expansion. The base is captured once, including inlined helpers;
exponent one returns it and larger exponents emit at most seven multiplications.
Existing **2 graphs retain their shape. Existing qint records, multiplication
kernels, scheduling, cleanup, resource selection, checked Program and Runtime
remain authoritative. No new public API, IR, power kernel, executor or dependency.

Only positive literal AST integers 1..8 are admitted. Bool, zero, negative,
fractional, symbolic/configured/computed and larger exponents are refused. The
256-bit, model/visit, target and resource guards remain. Source function bodies
are not executed. Classical power subexpressions fold, but the entry still must
return a quantum-dependent expression; pure classical entries remain refused.
Phase mode still requires a one-bit result. Notebook acceptance and independent
developer trials are not inferred from saved-file or cached-source tests.

On CPU complex128 / Torch 2.13.0, both objectives select 21 allocated/used qubits,
14 hidden work bits, 269 operations (2 H, 133 CX, 134 CCX), depth 206 for the
cube example, matching the native and explicit-product producers. The result
width is five bits, from the complete two-bit input domain. These are measured
implementation costs, not optimality, performance or hardware claims.

## Verification boundaries

The final new suite has 65 cases: all eight exponents on basis inputs under both
objectives, coherent two-bit cubes and native/product replacement, value/phase
complex reference entanglement, quantum-dependent constants/global signs,
classical folding, helper/base capture once, unsupported forms/configured
exponents, 256-bit/workspace and explicit memory refusal. Final run results are
recorded in the PR. The final targeted Compiler/Python regression passes all
996 cases in 193.25 seconds, including the final 65 new cases. The capture file
has 95% statement/branch coverage (333 statements, 180 branch exits); no claim
of complete file coverage is made. Strict mypy 2.x passes 879 production/example
sources; API snapshot and architecture checks pass unchanged.

Initial constant-folding fixtures incorrectly requested a pure classical return;
they were corrected to quantum-dependent returns without broadening production
semantics. A helper probe incorrectly expected a qint.call; existing capture
inlines it, so the probe now verifies one addition and two multiplications.
Those failed fixture runs remain recorded, rather than reported as green.

An unbudgeted high-constant execution probe exposed a real capacity issue:
x+2**8 can require an 8 GiB state. Owned duplicate test processes were stopped;
all eight folds and widths are now verified on the construction representation,
only small cases execute under explicit budgets, and a 64 MiB memory limit must
reject the wide case before state allocation. This is test-capacity discipline,
not removal of the width guard or a change to public memory defaults.

The superseded 3096-case broad snapshot was terminated after fixture correction
and resource contention; it is not a passing run. Final targeted Compiler and
Python workflow regression is recorded separately. The default suite again has
the same three local QFT fusion assertion failures; no Simulation implementation
or assertion is changed. The default run records 5030 passed, 121 skipped and
three QFT fusion assertion failures in 230.49 seconds; it is not recorded green.

Original source/license bytes, four-original cohort, trials 0/2 and
release_authorized=false remain unchanged. No arbitrary Python, general exponent
language, signed/fixed-point arithmetic, consuming roles or hardware admission.

## Subsequent capacity improvement

The [carry-free addition follow-up](SYNTH_CARRY_FREE_ADD_DELIVERY.md) proves
that the one-bit x+2**8 example needs no ripple carry workspace. Its planned
state is now 8 MiB, so the executable negative memory test uses a 1 MiB budget
and a separate positive case executes under 64 MiB. The 8 GiB / 64 MiB refusal
above records the earlier implementation, not the current capacity boundary.
