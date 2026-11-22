# Share CZ classification and synthesis

Status: proposed private compiler implementation; retain the first complete
observations under the admission below. This is a `single_device_fast_path`
change, not native-cost routing admission or qualified SDK performance evidence.

## Mechanism and boundaries

The [rejected precheck](COMPILER_CZ_COST_PRUNING.md) repeated Weyl decomposition
for reducing blocks. The private synthesis implementation now returns its known
entangler count with its proposal. The existing public `synthesize_two_qubit`
signature, validation, unsupported behavior and return value stay unchanged.
Both callers use one implementation of the existing Qiskit-derived KAK engine;
no new synthesis formula, dependency, numerical tolerance or accepted operator
criterion is introduced.

Native block analysis supplies its existing CZ objective and effective operation
limit. After validating the matrix and computing the existing Weyl classification
once, the engine may decline a non-reducing CZ proposal before creating its local
factors. Only CZ/RZ/RX requests qualify. The raw general form with `k` CZs has at
most `k + 10 * (k + 1)` instructions. That complete bound must fit the effective
limit before pruning; otherwise full emission preserves the existing priority of
instruction-growth refusal over non-reduction refusal. Every reducing proposal
still completes synthesis, raw growth checks, local cleanup, complete-block
operator checks and separate native/topology evidence replay. No value is cached
between calls or reused to bypass independent verification.

The public wrapper is required to retain its protected contract. The private
shared implementation has two current consumers: public synthesis and bounded
native block analysis. It does not add a public contract type. Existing failure
injection tests move to the shared implementation while retaining their original
incorrect-operator, unsupported-proposal, non-native and resource assertions.
The reducing controls additionally require exactly one target Weyl decomposition;
the resource-priority scenario requires full emission when its bound does not fit.

An initial naive precheck failed the existing resource-priority test. Its patch
and log are retained with the resulting evidence; it was withdrawn before the
full controls. The historical precheck and matrix-cache rejections remain valid
and are not relabeled as successes of this implementation.

## Fixed complete-workflow controls

Use integrated control `5a84f706ab9aeaab0ad3a5472bac292616fe39fe` and unchanged
SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`. Freeze the source hashes and
driver before executing the first full control. Both variants use AST-cloned
functions behind the same cached module-attribute patch list and equal context
lifetimes. Preserve Python 3.12.14, the current garbage collector and one
numerical thread. Use these already inspected nonces without redraw:

- `0101010101010101010101010101010101010101010101010101010101010101`
- `e99f9400cba6007b9be3900694e5ea6440bf570cd07046e060ca7856689a47ad`
- `5a8e1eff6bad7ac6c3dd0f2cab4d63e392b504e26acb76b1be22ab61a4945573`

For each nonce use every width 6/8/12, QAOA/Ising/hopping/QFT/repetition and
line/ring/grid/sparse combination. Run both ordinary-restored and target-artifact
workflows: 360 baseline/candidate pairs and 720 independent acceptances. Require
identical normalized output operation/layout hashes, acceptance metrics and
adapter audit counts. Capture native rewrite records and refusal counters without
requiring equality of timestamp-bound target/optimization identities. Preserve
all failures and attempt the whole declared set. Reconstruct and independently
replay every output without importing a participating SDK. Width six checks full
operators; larger widths check three probes only. These are inspected controls,
not held-out programs or a formal campaign.

After quality acceptance, use target-artifact QFT6/grid, repetition6/ring and
QFT12/grid with nonce 0, two warmups and nine retained alternating samples per
variant. Trace Python allocations separately around `adapter.execute` only.
Fixture/configuration setup and independent acceptance stay outside that scope.
Do not overlap local tests, profiling or artifact transfers with these samples.
Retain every sample and the first completed run. Require at least 5 percent lower
median compilation time in all three controls and no larger traced peak in any
control before proposing adoption. Do not tune cases, thresholds or repeat runs
to rescue an adverse result. Shared-host results are not certified latency,
process RSS, significance or SDK rankings; passing this necessary development
gate would not establish general superiority or qualification.

Also retain all six named stage cases from
`evidence/cz-cost-pruning-reduction-controls.json`. Compare complete operators
using the independent SDKBench numerical interpreter, exact proposal/refusal
equality, target Weyl-call counts, two warmups and twenty alternating retained
stage samples. Any extra target decomposition in a reducing block rejects this
proposal. Report every stage cost, including regressions; a faster refusal must
not hide increased work on reducing inputs.

A separate supervisor bounds each process group and preserves SDKBench's own
timers. Formal Linux resource qualification stays user-deferred. No machine,
service, driver or kernel change or restart is authorized.

## Existing user workflow and verification

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4).h(0).cx(0, 3)
result = compile(circuit, coupling_map=CouplingMap.line(4))
assert result.metadata["routing"]["mapping_restored"] is True
```

Expected: an equivalent, topology-legal circuit with restored output placement.
This change supports native target-artifact compilation behind existing APIs.
Run the focused tests with:

```sh
python -m pytest tests/unit/test_native_block_optimization.py \
  tests/unit/test_compilation_target_cz_resynthesis.py \
  tests/unit/test_compilation_two_qubit_synthesis.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Whole-repository CI and public-contract checks remain necessary. No snapshot,
default, result schema, comparison setting, resource budget or evaluator may be
changed to make the proposal pass.
