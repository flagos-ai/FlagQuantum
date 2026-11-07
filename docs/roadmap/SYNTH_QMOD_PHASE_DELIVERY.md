# Native one-bit phase statement delivery

Date: 2026-11-01. Unreleased logical CPU development evidence after #984
(merge aeb818233f8bc2a54d17f34af35e098deabc2c1d, 57 successful checks and
three configured skips). See the
[semantic admission](../api-changes/FQ-SYNTH-QMOD-PHASE-20261101.md).
Independent developer trials remain 0/2; release_authorized=false.

## User workflow

Run `python -m examples.synthesis.tutorials.qmod.qmod_phase.main`, or:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<1>, output y:qnum<1>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    phase(x*y, pi);
    hadamard_transform(y);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
print(program.report["resources"])
```

The named tuples (0,0) and (1,1) each have probability approximately 0.5.
A Path containing that source uses the same existing entry and result flow.
Users declare widths, known preparation, the sign expression and budgets.
They do not design a CZ, allocate scratch, choose physical wires, implement
inverse cleanup or decode bitstrings. Before the final Hadamard, the complete
state has amplitudes (1,1,1,-1)/2 in x/y order. Probabilities alone cannot
certify this phase; tests compare complete complex states and interference.

## Ownership and reuse

| Responsibility | Implementation |
| --- | --- |
| Native phase admission and pi/name/width diagnostics | New branch in existing Ecosystem adapter |
| Expression parsing, bounds and owned arithmetic records | Reuse existing native parser and _Expression |
| Initialized immutable borrows and linear ordering | Reuse Compiler qreg.phase contract from #984 |
| Exact marker, inverse cleanup, alias and workspace reuse | Reuse Compiler arithmetic/phase schedules |
| Constraints, selection and exact-phase target guard | Reuse existing Compiler checks |
| Text/Path acquisition and located public errors | Reuse existing Ecosystem source boundary |
| Program, memory planning, execution and named results | Reuse Services and Runtime unchanged |

All production changes are in one Ecosystem adapter. No production dependency,
public signature/export, snapshot, report schema or executor is added. The
adapter emits the same owned callee/phase action previously built by private
producers. The public Python phase producer and native source producer pass the
same unchanged execution/full-state consumer.

## Semantics and boundary

Only `phase(expression, pi)` with a quantum-dependent, exactly-one-bit unsigned
expression is admitted. The coefficient is symbolic, never evaluated. Register
basis values and initialization survive the call; relative/global phase may
change. `x*x` aliases are preserved, and `x*0+1` retains the overall minus sign.
Later source actions can mutate inputs or compute fresh arithmetic results
after the phase call's workspace is clean.

Other arities/coefficients, shadowed phase/pi names, pure classical phases,
multibit results, comparisons, arrays, helpers, Input/Const, drop/free and
within/apply remain explicitly unsupported. A multibit register initialized to
zero still has its declared quantum domain and is not truncated to one bit.
Default value mode returns the live named outputs; output_mode="phase" is still
a Python-only expression mode and remains refused for Qmod. Default logical
CPU targets are supported; public explicit basis/line conversions are refused
by the existing exact-phase guard before planning. The private exact-basis
target profile from #984 is not newly exposed by this source admission.
Memory limits retain the existing planned-buffer check before execution.

The pinned native library and official reference use expression/coefficient
argument order. Its QAOA Max-Cut original also needs arrays, real coefficients
and helpers and is not supported here. All new source examples are synthetic;
original source/license digests and the 3/3 admitted-original count are unchanged.
This slice makes no all-216, hardware, scalability or release claim.

## Acceptance and measured costs

On CPython 3.12.14, PyTorch 2.14.1, native CPU complex128:

| Synthetic workflow, either objective | Total qubits | Workspace | Operations | Depth |
| --- | ---: | ---: | ---: | ---: |
| Gate-free public example including preparation/interference | 2 | 0 | 4 (3 H, 1 CZ) | 3 |
| Complex phase, mutation, fresh product, second phase, interference | 6 | 3 | 21 | 20 |

Costs include complete preparation and final transformations, not only the
arithmetic marker. Width and depth objectives happen to select identical costs
for these fixtures; no scheduling improvement is claimed for them.

The 48 new tests cover all four basis assignments, complex superpositions,
text/Path entry, arithmetic aliases, exact global signs, repeated phases,
mutation, arithmetic after cleanup, interference, zero workspace probability,
named outputs, plan identity, located refusals, resource/memory boundaries and
replacement of the source producer behind the same execution consumer.
Together with the 98 existing parser/source unit cases, all 146 tests pass;
the entire adapter has 188/188 statements and 72/72 branch exits covered.
Run the focused pair and existing source cases with:

```bash
python -m pytest tests/team/ecosystem/test_qmod_phase.py tests/integration/test_synthesis_qmod_phase.py tests/team/ecosystem/test_qmod_parser.py tests/team/ecosystem/test_qmod_source.py -q
```

Existing arithmetic/source regressions, default-tier results and static/API
checks are recorded in the pull request after completion. They must distinguish
the known local native-QFT failures from this source slice. General quantum
function roles, source scopes and further original-corpus admission remain
separate dependency stages.
