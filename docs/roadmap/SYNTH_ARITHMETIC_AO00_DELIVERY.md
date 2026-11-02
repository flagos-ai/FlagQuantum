# AO-00: reproducible arithmetic synthesis baseline

AO-00 fixes the expression corpus, independent oracles and cost interpretation
before optimizing lowering. Production code and public contracts are unchanged.
The eighth planning revision's AO-01 through AO-05 remain future work.

## User workflow

Save this in a source-retaining Python file, or run
`python -m examples.synthesis_arithmetic_baseline`:

```python
import flagquantum as fq
from flagquantum import synth


def energy(x: int, y: int) -> int:
    return x*x+y


program = synth.synthesize(
    energy,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

Expected: all sixteen `(x,y,x*x+y)` outcomes have probability approximately
1/16; the complete-domain result has four bits, with all auxiliary wires zero.
The user declares unsigned domains and preparation, with no gate design,
physical wire assignment, carry management or bitstring decoding.

The equivalent existing Qmod path is:

```qmod
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);
    hadamard_transform(x);hadamard_transform(y);
    result=x*x+y;
}
```

Pass that string to `synth.synthesize(source, source_format="qmod")` and use the
same run/read workflow. Python uses the existing pure-function convenience form;
it does not imply that the still-separate UP-01 PR has been merged. AO-00 is
based on main commit `ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62` and does not merge
or require the new Python program frontend. UP and AO share existing Compiler
semantics and execution; this baseline does not introduce an alternate engine.

The independent two-bit Circuit reference is:

```python
import torch

reference = fq.Circuit(9, dtype=torch.complex128)
for wire in range(4):
    reference.h(wire)
reference.cx(1, 7).cx(3, 7)
reference.cx(2, 6).ccx(1, 3, 6)
reference.x(1).ccx(0, 1, 5).x(1)
reference.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8)
reference.ccx(0, 1, 4)
```

It is derived independently for this finite domain, not called by production
synthesis. For x=2a+b and y=2c+d, low-to-high output bits are b XOR d, c XOR bd,
a(1-b) XOR bcd and ab; scratch bc is uncomputed. The runner checks the full
uniform state against mathematics; tests check all 256 input/target combinations.

## Resource meaning and observed sentinel

| Implementation/profile | Allocated/used | Workspace | Operations | Depth | Two-qubit operations |
| --- | --- | --- | --- | --- | --- |
| Python/Qmod logical | 18/18 | 10 | 166 | 124 | 106 |
| Manual logical | 9/9 | 1 | 15 | 11 | 3 |
| Python/Qmod exact named basis | 18/18 | 10 | 860 | 567 | 410 |
| Manual exact named basis | 9/9 | 1 | 92 | 60 | 39 |
| Python/Qmod named basis + line | 18/18 | 10 | 2162 | 1088 | 1712 |
| Manual named basis + line | 9/9 | 1 | 196 | 112 | 143 |

These are static counts of generated circuits, not CPU runtime measurements.
Logical CCX is one three-qubit operation, so its cost becomes visible in the
decomposed target rows. Every target row uses the same existing exact
phase-capable basis and normal target compilation. Converting fixed logical
circuits is distinct from rerunning target-aware candidate selection. The
matrix's other expressions only claim logical resource evidence.

The 18-qubit planned complex128 state buffer is 4 MiB, versus 8 KiB at nine
qubits. Neither is process peak memory. The manual result is a reference, not a
lower bound or optimality proof. No AO optimization is implemented here.

The equivalent helper-expanded `square(x)+y` retains 18 qubits/166 operations/
depth 124. Exchanging addition operands to `y+x*x` currently produces 20 qubits,
12 workspace, 170 operations and depth 126, with a 16 MiB planned buffer. AO-00
records this real normalization gap; it does not modify lowering or assert that
all equivalent spellings already have equal cost. This is a regression target
for AO-01 and later candidate selection.

## Reproduction and scope

The [benchmark guide](../../benchmarks/synthesis/README.md) specifies all ninety
cases, schema, oracles, budgets, timing boundaries and commands. The checked
[baseline JSON](../../benchmarks/synthesis/baselines/ao00.json) records each case,
including completion status and verification scope. The runner source hashes
and measured code revision distinguish source identity from the later evidence
commit. Host-dependent durations are not exact-equality assertions.

Small cases have exact complete complex-amplitude checks, not just marginal
probabilities. Larger cases have independent exact gate interpretation on an
explicit finite sample, not a whole-domain quantum or formal equivalence claim.
A timed-out case is incomplete and cannot claim a cost or successful verification.
Default source/function naming and spelling differences are not normalized away
from this initial record.

This is local CPU static-resource/correctness evidence. It makes no performance,
optimality, QPU, distributed, independent-user-trial, release or increased
original-Qmod-corpus-coverage claim. No original source/license pin is changed.

## Validation receipts

Final measured corpus counts, repeat comparison and test/check receipts are
recorded with the completed baseline below; intermediate exploratory runs are
not substituted for the checked evidence.
