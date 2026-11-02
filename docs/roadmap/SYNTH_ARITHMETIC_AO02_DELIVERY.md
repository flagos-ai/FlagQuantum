# AO-02: guarded arithmetic implementations and terminal XOR output

AO-02 follows #1010, merged at
`a1a919a66af971f47061b15a2d9fe74ef63e3030` after its exact head passed
57 checks with three configured skips. This implements the eighth planning
revision's unsigned arithmetic stage. Python qfunc, Qmod and the existing scalar
entry share the same Compiler. There is no public API, dependency, canonical IR,
execution backend, selector budget or maturity promotion in this change.

## Unified user workflow

Run `python -m examples.synthesis_arithmetic_implementations` or save this file:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, memory_limit_bytes=64*1024**2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

All sixteen `(x,y,x*x+y)` outcomes, `x,y` in `0..3`, have probability 1/16.
Inputs survive, result storage keeps all four bits, and workspace returns to
zero. Users write the calculation and quantum preparation/composition; they do
not allocate arithmetic temporaries, choose wire numbers, design square/adder
gates, implement cleanup or decode bitstrings.

The same helper composition in Qmod remains supported:

```qmod
qfunc energy(const x:qnum<2>,const y:qnum<2>,output result:qnum) {
    result=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}
```

Pass this source to `synth.synthesize(source, source_format="qmod")`, then use
the same run/named-read workflow. The example executes and compares both forms.
Star import still exposes the same approved synth names; examples use explicit
imports for linting and types. No square-specific user entry or source-language
change is required.

## Independent Circuit comparison

The existing AO-00 finite-domain reference remains independent:

```python
import torch

reference = fq.Circuit(9, dtype=torch.complex128)
for wire in range(4):
    reference.h(wire)
reference.cx(1, 7).cx(3, 7).cx(2, 6).ccx(1, 3, 6)
reference.x(1).ccx(0, 1, 5).x(1)
reference.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8)
reference.ccx(0, 1, 4)
```

Inputs are wires 0/1 and 2/3 MSB first, result is 4/5/6/7, and wire 8 is
clean scratch. Its derivation and all 256 input/target checks remain in the
AO-00 oracle/acceptance files; no production code recognizes this expression
or imports this gate table.

| Complete uniform program | Allocated | Workspace | Operations | Depth | Planned complex128 buffer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen AO-00 / AO-01 Python and Qmod | 18 | 10 | 166 | 124 | 4 MiB |
| AO-02 Python and Qmod | 16 | 8 | 81 | 59 | 1 MiB |
| Independent Circuit | 9 | 1 | 15 | 11 | 8 KiB |

These include input preparation and all cleanup. State-buffer bytes are
`16*2**allocated`, not process peak memory or a runtime speedup. AO-02 has not
met AO-03/04's nine-qubit/24-operation target or AO-05's default promotion gate.

## Implementations and proof boundaries

The production changes stay in Compiler's existing `_arithmetic_kernels.py`
and `_construction/lowering.py`. They reuse source capture, normalization,
full-domain bounds, binding/initialization/effect verification, HybridProgram,
the clean workspace allocator, retain/recompute, resource estimation, target
compilation, the shared Program/run interface and existing numerical kernels.

| Rule | Mathematical basis | Actual target effect and preconditions |
| --- | --- | --- |
| Same-value square | `x*x = sum(x_i*2**(2i)) + sum(i<j, x_i*x_j*2**(i+j+1))` | Modular addition into an internal product; zero scratch restored, input preserved. Requires the same SSA operand, never equal widths alone. |
| Constant shift/add | `c*x = sum(c_j*x*2**j)` | Reuses Cuccaro sweeps without carry-out for modular addition; only zero padding and one carry are needed. Existing power-of-two copy specialization is retained. |
| Terminal XOR addition | `carry[i+1] = a[i]b[i] XOR a[i]carry[i] XOR b[i]carry[i]` | Computes carries independently of the target, XORs sum bits into the actual output, then reverses the carry prefix. Target may be nonzero or entangled. |

For a square, diagonal Boolean terms use `x_i*x_i=x_i`; the two ordered cross
terms combine into one term shifted up a bit. This is a width-independent
arithmetic identity, not a two-bit truth table or a recognizer for `x*x+y`.
Constant multiplication uses the same extracted Cuccaro sweep as the original
adder; its exact modular form omits carry-out while restoring the input/carry.
The original multiplier uses the same controlled-increment gate sequence as
before. Neither accumulator kernel is treated as an XOR writer.

All internal product/result allocations are proven zero before accumulator
use. A fresh Output can use that proof, while arbitrary-target writes keep the
stronger `target XOR f(inputs)` contract. No public output-state assertion is
added: the terminal adder is correct on both fresh and arbitrary targets.
Its prefix never reads or writes output wires, so reversing that prefix cleans
scratch after the write even for a coherent target. This also permits safe
terminal multiply/add composition: keep the live product, XOR its sum directly,
then clean the product. It does not implement arbitrary ADD-to-XOR fusion.

Equal operand maps make doubling a shift with no carries. The existing
disjoint-constant/no-carry proof also permits direct XOR copies with no scratch:
`x+256` for one-bit x now needs ten total qubits and zero workspace, preserving
the full nine-bit result and arbitrary high target bits. Proved-zero high result
bits are left unchanged by XOR; declared ports are never narrowed using an
input preparation or sampled value.

Binding alias, uninitialized reads, unknown workspace, effect/version and
unsigned-width validation still run before emission. No sharing crosses a
register mutation, region or computation epoch. Phase-mode emission remains
unchanged. Exact value adjoint/control reuse reverses the actual selected gates
and allocation history, including terminal carry compute/cleanup events.

## Bounded adoption and provenance

Each existing schedule retains its real AO-01 candidate and checks one bounded
arithmetic implementation family. Adopt it only when workspace, operation count
and ASAP depth all do not increase and at least one improves. The two public
schedule rows, objective ordering, hard-constraint consumer and incomplete-search
contract remain unchanged; this is not AO-04 implementation-by-target search.

`ArithmeticCandidate` retains the AO-01 candidate in `conservative`, the original
representation/candidate through AO-01's `normalization`/`baseline`, actual
alternative cost in `implementation_cost`, and applied `(ValueId, rule)` entries
in `implementation_rules`. Original source mappings and full-domain bit facts
remain attached. Budget refusals are retained rather than creating fictitious
original candidates. These are private records, not new public report fields.

An original resource refusal may be rescued by a proved implementation that fits:
two-bit addition fits nine qubits/two workspace and exactly fourteen arithmetic
operations. Its conservative path needs eleven qubits/four workspace and 41
operations. Resource refusals are distinguished from semantic errors; semantic
and binding errors are not swallowed. A small constant multiplier whose shift/add
candidate increases gates/depth keeps the actual conservative candidate.

Optional duplicate circuit emission retains the existing private 10,000-original-
operation probe bound; a skipped attempt records its budget diagnostic and keeps
the valid candidate. The 4096-node/256-bit, 100,000-operation/event and dependency
depth limits remain. This bounds work structurally, not elapsed wall time.
It neither promises all constants improve nor claims global or target optimality.

Terminal binding removes the width tradeoff from the old simple pressure chain.
Its historical exact cost/refusal assertions remain checked on the original
emitter; current tests check the improved executable. A shared-dependency
branched expression `(t+x)*(t+y)`, `t=x*y`, independently exercises a real
18-versus-17-qubit schedule tradeoff, target constraints, recoveries and budgets.
No fixed benchmark workload or original Qmod fixture is replaced.

## Frozen evidence and reproduction

The original files remain byte-for-byte unchanged:

- `ao00.json`: SHA256 `c5f084ae48066f8055a94a3be02d0d30d71c233783c7fd1234ccb2b3d464c864`, measurement revision `6fc0ad1f87b037dcbe1935d4546de5e1dbe841e6`.
- `ao01.json`: SHA256 `e54047e626c60c132d902d7007ff59f55ca8b2df99e8530988dad9805fca426d`, measurement revision `4cd4bafc0ae1fb5564afe2409a9a5d2117054aaa`.
- The six original measured corpus/oracle/runner source files and all original Qmod/license/source receipts are unchanged.

AO-02 measurement source revision is
`b5c3db4b487a4b144a83a1e50fc8541475708774`. The new `ao02.json` uses the same
ninety-case configuration, targets, seed, independent oracles and external
large-case deadline. `ao02-comparison.json` records 62 improvements, 27 unchanged
cases and one incomplete case relative to AO-00, with no observed logical
regression/tradeoff. 89 compile; `mixed_products-16x16` reaches the same original
30-second deadline. Timeouts are not successes. There are 6,018 basis/target
checks and 78 bounded coherent-state checks; larger cases explicitly use gate
interpretation rather than a fabricated dense-state execution.

The supplementary `ao02-paired.json` records sixteen real candidate pairs:
eight nonexample expression/domain cases times both schedules, with 16,512
exhaustive input/target checks on both candidates. These are pre-target arithmetic
blocks with cleanup and no preparation, distinct from the full-program table.

| Retain block | AO-01 work/operations/depth | AO-02 |
| --- | --- | --- |
| square, two-bit x | 8 / 92 / 72 | 8 / 62 / 48 |
| square, three-bit x | 12 / 276 / 216 | 12 / 162 / 126 |
| `3*x`, four-bit x | 12 / 246 / 192 | 9 / 182 / 177 |
| `3*x`, five-bit x | 14 / 347 / 270 | 10 / 215 / 209 |
| `5*x`, four-bit x | 14 / 279 / 216 | 11 / 199 / 193 |
| addition, two/two bits | 4 / 41 / 35 | 2 / 14 / 8 |
| addition, one/three bits | 5 / 56 / 51 | 3 / 13 / 8 |
| multiply/add, one/one/two bits | 5 / 45 / 39 | 3 / 15 / 12 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.synthesis.arithmetic_baseline --output /tmp/ao02.json
python -m benchmarks.synthesis.normalization \
  --baseline benchmarks/synthesis/baselines/ao00.json \
  --current /tmp/ao02.json --output /tmp/ao02-comparison.json
python -m benchmarks.synthesis.implementations --output /tmp/ao02-paired.json
```

The current logical guard does not guarantee nonregression after every target
conversion/routing choice; all nine existing target sentinels remain recorded.
No speedup, peak-RSS, hardware execution, optimality, release certification,
additional Qmod coverage or distributed scalability claim is made. AO-03/04
still own joint Boolean optimization and target-aware candidate enumeration;
AO-05 owns the full promotion gate, including repeated warmed timing review.

## Acceptance

Kernel tests enumerate complete small input/accumulator domains, modular overflow,
inverse action, arbitrary XOR targets, unequal widths, alias refusal and exact
complex amplitudes. Integration tests execute all sixteen two-bit basis inputs,
both native/Qmod uniform programs, relative-phase superpositions with every
nonzero target, both schedules, external controls, adjoints and zero workspace.
The retained conservative path and real resource/budget rescues are checked.

Two complete generations at the same measurement revision match every one of
the ninety statuses, circuit hashes, resources, strategies, verification records,
nine target sentinels and two spelling variants. The sixteen paired receipts
also repeat exactly. Timing observations are not required or claimed to match.

Validation receipts:

- Frozen-source compiler/construction/synth regression: 2,581 passed in
  415.94 seconds. Focused arithmetic/workspace/artifact/source-typing acceptance:
  109 passed in 21.32 seconds. Qmod parser acceptance: 82 passed. The six updated
  public-workflow/original-source suites pass 198 cases in 131.37 seconds.
- Strict mypy passes 884 production/example/paired-runner sources. The native
  example and matching Qmod execute with all sixteen expected outcomes. API,
  architecture, dependency, capability-maturity, language, hygiene and local
  documentation-link checks retain their existing contracts.
- The broader local default tier reports 17,620 passed, 527 skipped, 28 failed
  and six errors. One newly stale original-Qmod resource assertion (23 versus
  the actual 21 qubits) was fixed and all 82 parser tests pass. The remaining
  33 failure/error node identities match the AO-01 environment baseline:
  optional/Aer and older installed-package mismatches, unrelated simulation
  expectations, evidence pins and socket prerequisites. This is not a claim
  that the whole local default tier is green.
- Benchmark/release tier: 1,016 passed, 95 skipped, one baseline-proven
  no-PYTHONPATH subprocess failure importing `_FORBIDDEN_FALLBACKS` from the
  older installed package. CPU-distributed tier with socket permission:
  317 passed, 170 skipped, one existing macOS abrupt-rank-loss diagnostic-string
  failure. Neither failure is hidden by changing unrelated code or assertions.

During an exploratory related-suite run, editing test source after collection
invalidated two inspect-based capture positions. That run is not passing
evidence; the frozen-source rerun above completes the full related suite.
All historical resource tables and raw receipts remain available; current
resource assertions follow the actually optimized circuits. Original Qmod bytes,
declared ports, source/license hashes and all state/cleanup checks are preserved.

Local logs: `/private/tmp/fq-ao02-frozen-related.log`, `fq-ao02-final-focused.log`,
`fq-ao02-final-acceptance.log`, `fq-ao02-qmod-parser.log`,
`fq-ao02-public-regression.log`, `fq-ao02-final-mypy.log`,
`fq-ao02-measurement.log`, `fq-ao02-repeat.log`, `fq-ao02-paired.log`,
`fq-ao02-paired-repeat.log`, `fq-ao02-default.log`,
`fq-ao02-benchmark-tier.log` and `fq-ao02-distributed-tier.log`.
Remote exact-head PR CI is a separate gate; no repository-wide green or promotion
is inferred from the focused local checks.

## Main synchronization

The branch also merges main `6649717cd9859ceb64eaa525e07ec32932b70485` without changing or retagging the original
measurements. Its production Python bytes match measurement revision
`b5c3db4b487a4b144a83a1e50fc8541475708774`; the additional main changes are
outside this arithmetic implementation. Post-sync combined arithmetic,
workspace, artifact, source-typing and Qmod-parser acceptance passes 191 cases
in 15.43 seconds. Strict mypy again passes 884 sources, the native/Qmod example
runs, API protection passes and all 825 local Markdown links resolve.
Logs: `/private/tmp/fq-ao02-post-sync-acceptance.log`,
`fq-ao02-post-sync-mypy.log` and `fq-ao02-post-sync-example.log`.

## CI regression repair

The first remote run at `a238df9ed57f7570c2dd74503cecb0fb1c2c55cd`
exposed ten integration/example assertion failures, repeated across coverage
shards. `cpu-core`, `coverage` and `CI complete` were downstream gates, not
additional independent defects.

The repair changes no production compiler code or historical measurements:

- Keep every exhaustive basis/XOR, complex-amplitude, cleanup, gradient and
  routing check. Update current resource assertions to actual AO-02 costs:
  14 operations for the terminal addition block, 77 for the polynomial block,
  16/81/59 for the uniform native example, and 14/74/57 for the constant-y
  Qmod example (qubits/operations/depth).
- Preserve the historical 18-qubit manual Circuit and frozen resource files.
  Construct each expected statevector with that executable's own workspace
  width, rather than indexing both the 18- and 16-qubit programs with ten
  workspace bits.
- Include an eight-qubit coupling map for the optimized chain. That chain now
  selects retain for both objectives; preserve its routed phase/cleanup checks.
  The runnable selection example uses shared dependencies `(t+x)*(t+y)`,
  `t=x*y`, to retain a real 17-versus-18-qubit schedule tradeoff. Its original
  `pressure` function and the 34-row historical comparison workload remain
  unchanged. Both selected schedules execute with exact phase and clean work.

Focused reproduction of all affected files and three affected algorithm-example
cases: 36 passed, 24 deselected in 12.17 seconds. Full AO-02 and exact-head remote
CI results are tracked separately in the delivery ledger; this focused result
does not imply repository-wide CI success or merge authorization.
