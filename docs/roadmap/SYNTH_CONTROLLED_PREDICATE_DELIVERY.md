# Direct controlled-predicate phase delivery

Date: 2026-10-31. Unreleased development evidence. #976 merged as
3ab96490d5de080e48f909cbceee517cb74db04b after 57 successful checks and three
configured skips on a6cb0e3427498208a63ea9f1daba1d1e955532b0. The human
instructed continuing the sixth plan. This slice optimizes already supported
controlled predicates at the existing phase boundary; it does not introduce
source control blocks. See the
[semantic contract](../api-changes/FQ-SYNTH-CONTROLLED-PREDICATE-20261031.md).

## User computation and constraints

Save the following in a Python source file or run
python -m examples.synthesis_controlled_predicate:

```python
import flagquantum as fq
from flagquantum import synth


def marked(enable: int, x: int, y: int) -> int:
    return enable and (x < y)


program = synth.synthesize(
    marked,
    inputs={"enable": synth.UInt(1), "x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=16,
    memory_limit_bytes=16 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("enable", "x", "y"))))
print(program.report["resources"])
```

Every tuple has probability 1/32. Amplitudes are unchanged when enable=0 and
negated only when enable=1 and x<y; the control is coherent. This statement is
verified by complete complex amplitudes and interference, not inferred from a
probability table. Inputs and workspace return to their declared encodings.
Users specify widths, known preparation, the expression, effect and ceilings;
they do not allocate product temporaries or write gates/cleanup/decoding.
int reflects Python's numeric operand-valued and; all phase results still
require a verified one-bit unsigned value.

The public example additionally builds `return enable and flag` with two UInt(1)
inputs and max_qubits=2, max_workspace=0, max_operations=3. It actually executes
two Hadamards and one CZ, at depth 2 with a 64-byte complex128 state buffer.
The computed two-bit comparison example actually executes 10 total qubits,
5 workspace, 50 operations and depth 40; its planned buffer is 16 KiB.
These costs include preparation and target passes and are not raw source costs.

## One Compiler change and reused authorities

The lowerer recognizes a terminal returned multiply whose operands are both
quantum.uint(1). It ensures both operands, omits the product's output/scratch,
and marks directly with CZ. Repeated physical encoding reduces to Z. The exact
selected X/CX/CCX forward prefix is snapshotted and reversed literally after the
marker. Earlier cleanup/recomputations and their physical allocation history
are inverted too; no second implementation or inverse selection is performed.

Reuse SSA, lifetimes, zero/binding guards, both schedules, finite selection,
resource budgets, target pipeline and the same checked Program/run/read path.
No source grammar, opcode, type, report field, public signature/export, contract
schema or serialized format changes. Services, Runtime, Core and Simulation are
unchanged. Value XOR and general phase returns, mixed/classical/wide products and
nonterminal returned definitions retain their existing lowering. Both operands
remain live at the marking boundary even if dependencies must be rematerialized.

A general baseline is independently extracted from the unchanged value evaluator:
its first write to the disjoint output identifies the copy boundary; keep the
preceding prefix, mark its source bit, append that same prefix's inverse, then
remove the unused value-target slot. The baseline retains the value schedule's
allocated width; it is not a second production engine or a regenerated snapshot.

| Raw phase expression / both schedules | General prefix qubits / operations | Direct qubits / workspace / operations |
| --- | ---: | ---: |
| enable and flag, UInt(1) inputs | 4 / 7 | 2 / 0 / 1 |
| enable and (x<y), two-bit x/y | 10 / 87 | 10 / 5 / 81 |

Earlier comparison work dominates the second peak; the optimization does not
promise qubit savings for every expression. Raw baseline costs and actual public
costs above have different scopes. Total allocated slots, source peak workspace
and actual used qubits retain their existing meanings; no minimum-resource,
hardware timing or process peak-memory guarantee is made.

## Acceptance and remaining boundaries

14 new Compiler tests check direct/alias markers with no work, omitted product
events, live operand encodings, literal prefix adjoints, mirrored allocation
traces, exact operation limits, generic fallbacks and unchanged value XOR.
28 new CPU/CLI tests cover 64 public basis executions of all 32 control/data
inputs under both objectives. Seven expression cases, two schedules and both
specialized/general paths compare arbitrary complex states entangled with an
external reference. These include constant true/false, aliases, shared logical
reads and two computed flags. Control interference reads the actual predicate,
including constant-true relative phase; double application restores the state.
Full public complex preparations, zero workspace, exact budgets, plan identity,
value XOR and CLI are checked. Tolerance is 1e-12 with no global-phase quotient.

The scoped coverage run passed 53 checks: 14/14 new executable statements and
8/8 new branch arcs covered; whole-lowerer aggregate coverage 96%. It is not
repository-wide coverage. The largest new reference state is 11 qubits (32 KiB
complex128); separate original Qmod checks keep their original sizes.

Strict mypy passed 876 package files plus the public example. Minimum Black
24.1.0, current Black, Ruff, unchanged API, architecture/dependency/team scope,
language/vocabulary, capability/generated documentation and 757 local Markdown
file link checks passed. Local verification used CPython 3.12.14 / PyTorch 2.14.1
on CPU/complex128. No project dependency, signature, candidate contract,
baseline or historical proposal was changed.

Local default verification passed 4622 tests with 121 skips and stopped at the
same three pre-existing macOS native-QFT count failures (44 versus 28).
Simulation and those expectations were unchanged; local default is not claimed
green. Existing synthesis/construction/arithmetic/Boolean/phase/native-source
regression passed 1563 checks with 21 optional-dependency skips, including all
six original-size public file/text runs of the three unmodified Qmod originals.
Their corpus compatibility count stays 3/3. The separate public API candidate
suite passed 12 checks. These disjoint sets plus 42 new checks total 1617 passes
and 21 skips; coverage and earlier focused runs overlap and are not added again.
PR/CI status is recorded on the PR after publication. Public phase remains logical CPU only; dirty external
workspace, generic controlled Circuit legalization, within-apply blocks,
arbitrary angles, Qmod phase, GPU/QPU/gradient and full-corpus admission remain
outside this slice. Independent human trial records are 0/2 and release
permission is false. No GitHub review or independent trial is invented.
