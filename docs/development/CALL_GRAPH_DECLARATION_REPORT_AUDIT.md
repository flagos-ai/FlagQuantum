# Complete call-graph declaration report audit

## Source and native owners

The complete Q-0038 source is `dev_tools/bloq-method-overrides-report.py` at
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged
Git blob is verified. Its direct class discovery dependency is already accepted
in `tools/development/component_discovery.py`. Existing library graph comparison
covers all 393 source examples, 3693 graph nodes and 8550 edges.

The source report checks a framework inheritance convention. FlagQuantum owns
the corresponding contract at the existing Compiler `CallDeclaration` alias,
`SymbolAllocator` and `RawCallees` types in
`flagquantum/compiler/_component_trace/call_graph.py`. Components provide direct
edge declarations. Compiler owns traversal and multiplicity aggregation in
`symbolic_call_graph`, `symbolic_call_graph_union` and `component_callee_counts`.
The existing strict mypy gate checks declarations at these typed consumer
boundaries. Existing package and developer typing commands own report execution.

The native disposition reuses these existing owners. A duplicate inheritance
hierarchy or runtime annotation checker would introduce a competing contract.
The source-only report's exact text and erroneous outcomes remain executable
reference evidence in `tests/unit/test_call_graph_declaration_report.py`.

## Complete definition and dependency mapping

| Source definition or check | Native disposition and evidence |
| --- | --- |
| `_call_graph` traversal override refusal | Traversal is a Compiler function. A declaration object's `call_graph` method cannot replace it. An executable poison-method fixture proves the central consumer still traverses and aggregates. |
| `_call_graph` count override warning | Compiler aggregation consumes direct declarations. A declaration object's `bloq_counts` method cannot replace it. The same fixture obtains the expected 2 by 3 terminal multiplicity without invoking either poison method. |
| `_call_graph` argument name and annotations | Existing `CallDeclaration[Node]` requires node and allocator arguments. Actual strict mypy rejects missing or wrongly typed allocator arguments and unannotated functions. Positional argument names follow the native callable contract. |
| `_call_graph` return annotation | Existing `RawCallees[Node]` admits mapping and iterable pairs. Actual strict mypy accepts both and rejects a scalar return. |
| `report_call_graph_methods` discovery loop | Existing component discovery and package/developer typing scope own selection. The full unchanged original report runs over all 340 source classes. Source discovery still stops on an actual report exception. |
| `main` and script entry | Existing `python -m tools.development.typecheck_command package` and `development` commands run native static checks. All three source entry paths execute unchanged in the reference harness. |

Native signature diagnostics are mypy diagnostics under the existing project
policy. They differ from source print strings. Source class-based discovery maps
to native typed consumers and declaration owners. The optional source-audit mypy
profile remains separately documented and preserves source suppression policy.

## Executable native workflow

Use the following file from a checkout to audit a direct declaration.

```python
from flagquantum.compiler._component_trace.call_graph import CallDeclaration, SymbolAllocator


def declare(component: str, symbols: SymbolAllocator) -> dict[str, int]:
    return {"leaf": 2} if component == "root" else {}


declaration: CallDeclaration[str] = declare
```

```sh
python -m mypy --strict --explicit-package-bases --follow-imports=silent declaration.py
```

The valid declaration passes. A scalar return, wrong allocator type or missing
allocator fails with an assignment diagnostic. An unannotated function fails
with `no-untyped-def`. The focused tests invoke the real checker with the existing
root configuration. This declaration layer describes models. Existing physical
execution continues through the accepted `circuit.append(op)` path.

## Unchanged original behavior and defects

The source accepts only the exact historical `Set[ForwardRef('BloqCountT')]`
return annotation object. The frozen call graph already accepts dictionaries and
deprecates set returns. The complete source corpus therefore produces 340 return
annotation warnings. In the acceptance pytest environment, nineteen classes additionally produce
allocator-annotation warnings, for 359 exact lines in discovery order. The
standalone intake has 340 return warnings. Annotation resolution and import
history affect the exact allocator annotation object. Both observations retain
the unchanged source and all 340 classes complete.
The initial intake counted diagnostic classes. An initial test confused that
number with line count. The expected allocator-warning owners also needed fully
qualified names because two source classes share a short name. Final assertions
check all 359 lines explicitly in original discovery order.

Synthetic source classes execute every branch. A traversal override prints and
raises `ValueError`. A count override only prints, retaining the original
`overriden` spelling. Missing annotation keys first print a signature warning,
then raise `KeyError` because the source indexes the missing key. Additional
annotation keys print a warning and continue. Incorrect allocator and return
annotations print their exact warnings. The discovery loop stops at the first
exception, demonstrated through `report_call_graph_methods`, `main` and the
actual script entry. The original file remains unchanged.

Native strict typing expresses the intended callback contract without reproducing
the source's obsolete literal annotation checks. Whole-library native strict
acceptance remains a separate gap. This closure proves valid and invalid
consumer-bound signatures and the central traversal/aggregation ownership.

## Verification and resource scope

Focused cases cover the complete original file and six actual native typing
witnesses. Batch regression includes accepted source discovery and full library
call-graph comparisons alongside the current development tool group. Strict
mypy, Black, Ruff, Integration scope, architecture, IR and API checks run
separately.

This developer report emits no quantum primitives and has no measurement or work
registers. Its runtime graph witness has modeled leaf multiplicity six. Physical
primitive counts and resource-model estimates remain with the already accepted
execution and modeling modules. Broad default-suite timeout, Algorithms NumPy
ownership and preceding-execution copyright interaction remain open. Pending
Proposals 116/117/118 remain outside this closure.

The final concentrated run passed all 554 cases with 956 visible warnings and
zero skips in 267.61 seconds. Strict mypy passed the changed test harness.
Black, Ruff, Integration scope, architecture, IR, API and diff checks passed.
The initial concentrated run retained its one incorrect corpus-expectation
failure and 553 passing cases as a separate log. The full batch was rerun after
the expectation correction. No source production or frozen reference code was
changed to obtain the passing result.
