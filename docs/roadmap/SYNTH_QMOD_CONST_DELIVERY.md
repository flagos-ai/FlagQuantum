# Native Const/Output admission and call-boundary replacement

Date: 2026-11-01. Sixth-plan source admission after #988, merged at
9ed21b991da11ede194cea9e2eadc0005c6cc891 with 57 successful checks and
three configured skips. See the
[contract](../api-changes/FQ-SYNTH-QMOD-CONST-20261101.md). Independent trials
remain 0/2 and release_authorized=false.

## Complete user flow

Run `python -m examples.synthesis_qmod_const`, or:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc compute_energy(const x:qnum<2>, const y:qnum<2>, output energy:qnum) {
    energy=x*x+y;
}
qfunc main(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    compute_energy(x,y,energy);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
print(program.report["resources"])
```

Each of the sixteen `(x,y,x*x+y)` tuples has probability approximately 1/16.
The same source in a Path uses the existing entry and named reader. Users
declare types, preparation, effects/roles and budgets; they do not design gates,
work storage, inverse cleanup, wire mappings or bitstring decoding. The helper
receives live values and produces a fresh result, so reusable computation is no
longer restricted to preparing every argument from fresh Output storage.
This native compatibility feature does not change the Python-first direction
or add Python Const classes, a new runner or a required decorator.

## Architecture and replacement

| Responsibility | Implementation |
| --- | --- |
| const scalar syntax and explicit widths | Extend existing private Ecosystem parser |
| Source positional kinds/order to Const-first ABI | Existing adapter emits qreg.call |
| Definition/initialization/freshness/alias/effect checks | Reuse #988 Compiler verifier |
| Call substitution, visit/node bounds and work cleanup | Reuse #988 Compiler expansion/lowering |
| Phase target guards, resource selection and identity | Reuse existing Compiler boundary |
| Program, memory, execution and named reading | Reuse Services/Runtime unchanged |

Remove the former Ecosystem Output-only effect substituter and alias checker.
Both Output-only and mixed helpers stay owned call records until Compiler
lowering. All definitions, including unused ones, are verified at the adapter
handoff. Qualified arithmetic callees and register definitions share the
existing handoff with no new type, dispatcher, canonical IR, serializer or
dependency. Only two existing Ecosystem production modules change; Compiler
production code is unchanged. Public signatures, exports, snapshots, report
fields and execution interfaces remain unchanged.

## Semantic boundary

Helpers admit const qbit and const qnum with an explicit unsigned width, plus
existing Output scalars. Parameters can be interleaved in source order; the
adapter matches positionally before emitting Const actuals then Outputs.
Const actuals must have known exact kinds/widths and be initialized. They can
alias other Const actuals. Outputs must remain fresh and disjoint from all
arguments. Output width inference still uses complete quantum domains, not a
particular prepared input value. A malformed helper cannot borrow caller state
to pass verification, and empty calls still require initialized borrows.

Const preserves basis values while allowing phase and entanglement. Arithmetic
and phase inputs and nested Const calls are allowed. Gate targets (including
S/Z and cancelling X/X), reinitialization, compute destinations and nested
Output roles are conservatively refused by the existing closed proof. Callers
can mutate values after return. Main stays Output-only with source-owned
preparation. Bare-width Const qnum, Input, input const, output const, unqualified
mutable parameters, arrays/views, locals, free/drop and within/apply remain
unsupported. Existing one-bit/pi phase and exact-target boundaries remain.

Error classes and source coordinates are preserved; role/alias/expansion errors
now carry Compiler categories. Existing lexical/expression/source/declaration
limits remain. Compiler owns graph, 4096-node/visit and 32-call-level bounds,
including empty call trees. The three retained original Qmod files, licenses
and source digests stay unchanged; new probes are synthetic and do not increase
the 3/3 admitted-original count or imply all-216 compatibility.

## Evidence

68 new cases pass: 23 source semantics and 45 real workflow cases. Together
with existing helper/parser/source/phase cases, 250 distinct cases are verified.
The final coverage invocation had 248 passes and two file-fixture setup errors
from transient local ENOSPC; both file cases passed separately on rerun. The
first invocation is not described as green. All native frontend modules cover
494/494 statements and 198/198 branch
exits. The old helper tests now assert owned calls and Compiler refusal codes;
the existing execution consumer is unchanged.

Acceptance includes every two-bit x/y input, all source role orders, inferred
Output widths and fresh-helper outputs used as Const actuals, full complex states, external correlations, repeated Const
aliases, global signs/interference, caller mutation, repeated-call work reuse,
text/Path, direct-producer replacement, invalid unused definitions, empty
borrows, forbidden writes, budget/memory/target refusal, a value-only line target beside an unused phase helper, and no execution during
synthesis. State residual and dirty-work probability use 1e-12 tolerance.

On CPU complex128, CPython 3.12.14 and PyTorch 2.14.1, the complete public example
uses 18 total/used qubits, 10 work qubits, 166 operations (4 H, 106 CX, 56 CCX)
and depth 124. Both objectives select the same costs here; no optimality or
scheduling benefit is claimed. Broad/default/static results are recorded in the
PR on completion, with the existing local QFT failures reported separately.

```bash
python -m pytest tests/team/ecosystem/test_qmod_const.py tests/integration/test_synthesis_qmod_const.py -q
```
