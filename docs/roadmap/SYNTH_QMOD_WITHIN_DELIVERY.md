# Native compute/apply/cleanup scopes

Date: 2026-11-01. Continue the sixth-plan source path after #992, merged at
a38588eb502e6f3465b5d81d1cc3387787d5dd33 with 57 successful checks and three
configured skips. See the [contract](../api-changes/FQ-SYNTH-QMOD-WITHIN-20261101.md).
Independent developer trials remain 0/2 and release_authorized=false.

## User value and executable workflow

Native source users can compute temporary conditions, apply their signs and
reuse the storage without exporting fake result flags or writing inverse gates.
Run `python -m examples.synthesis_qmod_within`, whose complete public workflow is:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc mark(const x:qnum<2>, const y:qnum<2>) {
    flag:qbit;
    within { flag=x<y; } apply { phase(flag,pi); }
    within { flag=x==y; } apply { phase(flag,pi); }
}
qfunc main(output x:qnum<2>, output y:qnum<2>) {
    allocate(x); allocate(y);
    hadamard_transform(x); hadamard_transform(y);
    mark(x,y);
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=10)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Every input pair has probability 1/16; full amplitudes are
(-1)**int(x<=y)/4. Only x/y are named results. This same source can be saved as a
local .qmod file and passed as pathlib.Path without changing the consumer.
The existing Python `return x <= y` phase entry remains shorter for new Python
programs; this slice supplies bounded compatibility and local scope semantics,
not another preferred Python language or a decorator requirement.

## Ownership, semantics and subtraction review

Only two existing Ecosystem production modules change. syntax.py parses local
declarations and block syntax into its existing private AST; adapter.py translates
them to existing Compiler-owned qreg.local/compute/phase/call/uncompute/release.
The AST never crosses the boundary. Compiler remains the only owner of effects,
initialization, Const/alias checks, dependency proof, selected exact inverses,
zero-pool ownership, resource selection, target guards and bounded expansion.
Runtime and the Program/run/named reader are unchanged. No new public API,
canonical IR, type, registry, scheduler, allocator, executor or dependency is added.

Function-level qbit/qnum declarations must precede use. Within prepares fresh
locals with quantum-dependent unsigned arithmetic/comparisons; later preparations
may depend on earlier locals. Apply admits phase(expr,pi) or Const-only helpers.
Helpers can contain supported scopes themselves. At exit the adapter emits
cleanup in reverse preparation order. Source names can be reused in later
blocks with fresh owned identities, while their widths remain fixed. Bare qnum
infers once; explicit wider storage preserves zero extension. Unused declarations
allocate no storage. Locals stay outside inferred output widths and named results.

The [official within-apply semantics](https://docs.classiq.io/qmod-reference/language-reference/statements/within-apply/)
motivate exact inverse and dependency-preserving actions. This profile admits
only the bounded compute/phase/cleanup case: lexical nesting, local declarations
inside blocks, pure literal/allocation/gate/helper preparation, Output preparation,
arbitrary apply gates/writes, Output calls and manual uncompute/release/free/drop
are refused. Empty apply is valid; empty within is refused. Existing one-bit
polynomial/pi phase rules remain; comparisons must first compute a flag.
Source/token/function and Compiler expansion/node/visit/width/memory bounds remain
in force, including generated local and cleanup operations. Errors retain source
coordinates, and invalid source fails before planning. Phase target guards and
default 512 MiB planned-buffer admission are unchanged. No source evaluation,
SDK/cloud execution or synthesis-time simulation occurs.

## Acceptance and measured costs

111 new cases pass: 51 parser/translation/proof cases and 60 workflow cases.
They cover all sixteen basis inputs under both width/depth objectives, complex
states, external references, absolute/relative/global signs, interference,
chained multiplication/comparison locals, same-name reuse, nested helper calls,
raw selected inverses under retain/recompute, and clean hidden work at 1e-12.
Native, Python and owned-IR producers replace each other behind unchanged
Program/run/named consumers. File/text forms, resource/memory/target constraints,
no execution during synthesis and pre-planning refusals are tested.

364 Qmod-focused cases pass, covering all 543 statements and 234 branch exits
in the two production modules. The twelve public API candidate cases pass;
strict mypy passes 879 production/example sources. Both existing source-overload
typing cases pass with the same flags and cache disabled: the valid case has
zero errors, and the invalid case has exactly ten. Minimum-version Black, Ruff,
API snapshot, architecture, dependencies, vocabulary, repository language,
maturity, generated documentation, links and scope checks pass. Original fixture
verification passes all seven digests.

Local broad/default outcomes are retained separately in the PR. The first
default attempt stopped with 306 passed, 54 skipped, one failure and two fixture
errors because ENOSPC prevented unrelated SDK-wheel fixtures from being written.
An additional whole-Ecosystem coverage attempt also hit fixture/SQLite ENOSPC
(770 passed, nine skipped, 196 errors, exit 3); it is not counted as a pass.
After removing only completed generated test fixtures, the focused 364-case
coverage run above passes. No filesystem guard or test is weakened, and this is
not a claim that the complete local default suite passed.

The broader scoped regression first produced 2541 passes, 21 skips and two
failures in 520.91 seconds. Both failures exposed a real compatibility issue:
an existing helper named apply was rejected by the initial reserved-word change.
Within/apply are now recognized only at block grammar positions; helper/variable
names remain valid, and the private AST distinguishes an empty block from a
call. The final focused run includes all 45 existing native Const workflow cases
and two new contextual-name probes. One empty-helper probe was corrected to
expect the existing register-return refusal, without broadening Compiler rules.
The first broad run is retained as failed; unaffected passes and the focused
correction are separate evidence, not a claimed full rerun of the broad suite.
The complete affected native-source regression then passes 568 cases in
27.20 seconds: all Qmod unit cases, native comparison/Const/helper/phase/within
workflows and parser execution. The unchanged large-original cohort was already
verified by the initial broad run and is not rerun or added to this count.

The following are exact logical CPU complex128 measurements for the example;
width and depth objectives produce the same counts in each row:

| Preparation order | Peak owned local bits | Whole hidden work | Allocated / used | Operations | Depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two sequential scopes | 1 | 6 | 10 / 10 | 56 | 43 |
| Both flags in one scope | 2 | 7 | 11 / 11 | 64 | 54 |

Sequential counts are 4 H, 14 X, 22 CX, 14 CCX and 2 Z. Overlap adds eight CCX.
max_qubits=10 and max_workspace=6 accept sequential scopes and reject overlap.
The compiler preserves source effect order; it does not automatically rewrite
overlap into sequential scopes. These are implementation costs, not an optimum,
performance or hardware claim. CPU semantics and actual resources do not imply
release admission or compatibility with all 216 retained-source candidates.
The original four-source cohort and all source/license/manifest bytes stay intact.

The former blanket local-declaration refusals now test unscoped initialization;
unused declarations are explicitly positive. General consuming Input/free/drop,
arbitrary unitary inversion/control, signed/fixed-point values, arrays/views,
arbitrary phase coefficients, explicit routed-phase targets, GPU/QPU and
independent developer acceptance remain outside this slice.

```bash
python -m pytest tests/team/ecosystem/test_qmod_within.py tests/integration/test_synthesis_qmod_within.py -q
python -m examples.synthesis_qmod_within
python tests/fixtures/classiq_qmod/verify_sources.py
```
