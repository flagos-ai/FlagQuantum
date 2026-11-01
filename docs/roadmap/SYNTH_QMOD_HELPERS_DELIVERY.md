# Fresh-output native quantum helper delivery

Date: 2026-11-01. Unreleased logical CPU development evidence. #985 passed 57
checks with three configured skips and merged at
03ddcc63fcd24d0fee04efe576bfdafd3c8fb172. This sixth-plan slice establishes
the existing fresh Output obligation across named helper boundaries. See the
[semantic contract](../api-changes/FQ-SYNTH-QMOD-OUTPUT-HELPERS-20261101.md).
Independent trials remain 0/2 and release_authorized=false.

## User workflow

Run `python -m examples.synthesis_qmod_helpers`, or:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc prepare_grid(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
}
qfunc prepare_source(output x:qnum<2>, output y:qnum<2>, output energy:qnum) {
    prepare_grid(x,y);
    energy=x*x+y;
}
qfunc main(output x:qnum, output y:qnum, output energy:qnum) {
    prepare_source(x,y,energy);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "energy"))))
print(program.report["resources"])
```

The sixteen `(x,y,x*x+y)` tuples each have probability approximately 1/16.
Output widths infer to 2, 2 and 4. A Path containing this source uses the same
entry. Users organize reusable domain preparations and high-level arithmetic;
they do not design gates, temporary storage, inverse cleanup, wire mappings or
bitstring decoding. Explicit widths, quantum preparation, fresh-output roles
and resource ceilings remain user obligations. This compatibility entry does
not change the Python-first product direction.

## Architecture and reuse

| Responsibility | Implementation |
| --- | --- |
| Bounded definitions and exactly one main | Extend existing Ecosystem parser |
| Acyclic resolution, output binding, source coordinates | New logic in existing Ecosystem adapter |
| Definition verification and final main verification | Reuse Compiler register model/verifier |
| Output initialization and no double initialization | Reuse existing fresh-register rules |
| Arithmetic, phase and exact cleanup | Reuse existing owned callees and Compiler schedules |
| Selection, targets, budgets, memory, Program, run and read | Reuse Compiler/Services/Runtime unchanged |

Every helper, including unused definitions, is independently verified before
it can supply output widths. Verified effects are substituted by reference and
threaded through the existing linear effect token. Arithmetic callees carry
definition-qualified names and are deduplicated across repeated calls. There
is no new canonical call IR, public type, executor, dependency, signature,
report field, serializer or runtime dispatch table. Production changes are
confined to two existing Ecosystem modules. The source parser remains private.

Output formals mean fresh storage on entry and fully initialized storage on
return. Actuals must be distinct scalar output variables of the same kind and
resolved width. Bare qnum actuals can infer width; declared widths cannot be
silently widened or narrowed. Already initialized actuals fail the existing
Compiler verifier. Helpers cannot capture caller variables or use caller state
to validate an otherwise invalid body. Definition order is independent of call
order; cycles and calls to main are rejected. Source budgets remain bounded at
32 definitions, 4096 tokens, 4096 expanded effects and existing Compiler
node/width limits. Expanded operations locate call sites, while arithmetic
definitions retain their source locations.

Input/Const, classical parameters, higher-order functions, arrays/views, locals,
drop/free and within/apply remain unsupported. Existing one-bit/pi phase and
exact target restrictions apply inside helpers. This does not claim general
mutable borrowing, arbitrary unitary composition or full Classiq compatibility.
The [official quantum-types examples](https://docs.classiq.io/qmod-reference/language-reference/quantum-types/)
show output preparation helpers; this implementation admits only the scalar
unsigned subset. All new examples are synthetic. Original source/license hashes
and admitted-original count 3/3 remain unchanged; no all-216 or release claim.

## Acceptance and costs

On native CPU complex128, CPython 3.12.14 and PyTorch 2.14.1, the complete
example uses 18 total/used qubits, 10 workspace, 166 operations (4 H, 106 CX,
56 CCX), and depth 124 under either objective. These include preparation and
cleanup, and match the equivalent flat-source workflow. The two objectives
happen to select identical costs here; no scheduling benefit is claimed.

The 62 new tests pass: 31 source cases and 31 complete workflow cases. Coverage
with 120 existing parser/source/phase unit cases totals 182 passes; all four
native frontend modules cover 478/478 statements and 192/192 branch exits.
Tests cover the maximum 32-definition chain, expansion bounds, forward/nested/
repeated calls, lexical isolation, invalid unused definitions, exact kinds and
widths, alias/freshness refusals, all sixteen two-bit input pairs, uniform full
states and zero workspace, complex phase/interference, text/Path, producer
replacement, named reading, plan identity and no execution during synthesis.

```bash
python -m pytest tests/team/ecosystem/test_qmod_helpers.py tests/integration/test_synthesis_qmod_helpers.py -q
```

The PR records completed broad regressions, default-tier results and static
checks, including the known local QFT count failures if reproduced. General
function-role and scope contracts remain the next dependency; this Output slice
does not establish consumed Input or basis-constant Const effects.
