# Native unsigned comparison delivery

Date: 2026-11-01. Continue sixth-plan Stage 2 after #989, merged at
1283555b482ca2e6ac0e727e66531fa00bf00615 with 57 successful checks and three
configured skips. See the [semantic contract](../api-changes/FQ-SYNTH-QMOD-COMPARISONS-20261101.md).
Independent developer trials remain 0/2 and release_authorized=false.

## User workflow and architecture

Run `python -m examples.synthesis.tutorials.qmod.qmod_comparisons.main`, or copy the complete source
and run example from the [synth README](../../flagquantum/synth/README.md).
Users describe live inputs, fresh outputs, preparation, unsigned widths and
resource ceilings. The helper computes x<y on all sixteen two-bit input pairs
coherently, without gate design, wire numbers, work scheduling or bit decoding.
The same Program/run/named-reader consumer accepts native and Python producers.
Python remains the primary user entry; native syntax is source compatibility.

Only two existing Ecosystem production modules change. The private parser
recognizes ==, !=, <, <=, > and >= below arithmetic precedence; the adapter emits
existing qint.equal/qint.less operations, reversed operands and equality-to-zero
complements. Operand expressions are emitted once in original source order.
Compiler owns complete unsigned zero extension, verification, call expansion,
comparators, cleanup, schedules and costs. Services and Runtime keep Program,
memory/target guards, execution and named reading unchanged. The small common
SSA emitter replaces duplicate constant/arithmetic construction; no new kernel,
operation kind, public type, API, report field, dependency or executor is added.

## Semantic boundaries

Quantum comparisons return UInt(1), independent of prepared values; classical
comparisons fold to exact 0/1. Literal operands, unequal widths, aliased Const
actuals and composition with arithmetic are supported. Unparenthesized chains
fail rather than acquiring Python chain semantics. Explicit binary nesting is
allowed. Logical operators, signed/fractional values, mutable/consuming Input,
locals, free/drop, within/apply and statement-level control remain unsupported.
Existing lexical, width, source/declaration and Compiler expansion budgets apply.

The official [native phase documentation](https://docs.classiq.io/qmod-reference/language-reference/statements/phase/)
requires a polynomial expression. A comparison anywhere in its argument is
refused before planning. Compute an exported flag and phase(flag,pi) instead.
The flag remains a live named output with its correlations; only internal work
is zeroed. This is not implicit uncomputation of an exported result. Python
predicate phase mode and exact CPU target restrictions retain their contracts.
An existing negative phase test now expects qmod.phase instead of qmod.operator
because comparison parsing is admitted but this statement remains restricted.
The earlier Python-comparison delivery's native refusal case now checks a
positive full-state binary comparison and a refused chain, matching this
contract instead of retaining an obsolete prohibition on all comparisons.

## Original source and evidence

The unchanged comparator_integer_example.qmod at Classiq commit
2bfd835e73b6882fcc515a725098b04766c238e7 prepares a uniform three-bit a and
computes res=a<=2. The fixture retains all 108 upstream bytes; Git blob SHA-1
76c6b3ce28f48901a4982fc3d4251deeb1777d88 and SHA-256
1290eacc5d2d6ab66e7ea8f89e0110d588c949666dfba54cddc589acf7257528 are recorded in
tests/fixtures/classiq_qmod/comparison_sources.json. Its pinned repository/path
and shared original MIT license are checked, including corruption refusals.
The historical manifest, three fixtures, license, verifier and API contract stay
byte-identical; the added cohort has its own provenance test.

Both Path and text execute at the original three-bit input and inferred one-bit
result sizes. Complete amplitudes match 1/sqrt(8) at (a,int(a<=2)), with zero
internal work and named probabilities 1/8. This adds one actually parsed,
lowered, executed and full-state-verified original to the baseline three:
4/4 for the retained execution cohort, not all 216 upstream files. The separately
inspected comparator_2vars_example.qmod needs signed multi-argument allocation;
it is not rewritten, admitted or counted here.

166 new tests pass: 34 unit/provenance and 132 real workflow cases. They include
all six relations over all sixteen two-bit pairs, unequal domains, literals on
both sides, aliases, precedence and chains, classical folding, complex phases
on exported flags, external entanglement and zero workspace under both
objectives. They also prove native/Python producer replacement behind the
unchanged consumer, original file/text execution, refusal before planning,
no execution during synthesis, and existing qubit/memory/target guards.
Full-state and dirty-work tolerances are 1e-12. Broader results are recorded in
the PR; known local QFT baseline failures are reported separately.
The 340-case frontend coverage run passes with all 517 statements and all 210
branch exits covered across the existing four native frontend modules.
The broad run records 2173 passes, 21 skips and one obsolete all-comparison
refusal. After updating that case to verify binary admission and chain refusal,
its targeted rerun passes, giving 2174 distinct existing cases verified. The
initial invocation is not described as green. The default tier records 4911
passes, 121 skips and the same three local native-QFT 44-versus-28 failures.

CPU complex128 measurements on CPython 3.12.14 / PyTorch 2.14.1 include all
preparation and exported flags. Both objectives choose identical costs here:

| Complete workload | Allocated / used qubits | Work qubits | Operations | Depth |
| --- | --- | --- | --- | --- |
| Two-bit x<y helper example | 10 / 10 | 5 | 49 (4 H, 14 X, 17 CX, 14 CCX) | 40 |
| Original three-bit a<=2 | 10 / 9 | 6 | 42 (2 U3, 1 H, 18 X, 3 CX, 18 CCX) | 27 |

Allocated work includes a reserved unused line in the original; do not replace
allocated width with used width. These are measured candidate costs, not an
optimality, speed, hardware or release claim. No public resource schema changes.

```bash
python -m pytest tests/team/ecosystem/test_qmod_comparisons.py tests/integration/test_synthesis_qmod_comparisons.py -q
python tests/fixtures/classiq_qmod/verify_sources.py
```
