# Complete-operator routing and native synthesis controls

Source: `780d1a1522da67f6e510d94f7e2a034fa4bdb7c0`.

One hundred deterministic numeric 3/4-qubit circuits were attempted with each
of auto, sabre, sabre_layout and persistent_layout on a line target. The existing
256-added-operation native legalization budget and existing native block budget
were retained. Every accepted result was compared as a complete operator with
the source, including initial placement, and separately with the routed program.
Final identity placement and every emitted two-qubit target edge were checked.

| Strategy | Accepted | Resource rejected |
| --- | ---: | ---: |
| auto | 100 | 0 |
| sabre | 100 | 0 |
| sabre_layout | 100 | 0 |
| persistent_layout | 94 | 6 |

The 394 accepted outcomes have maximum complete-operator residual
`1.4385101935290473e-14`, below the diagnostic `1e-10` threshold. The helper's
native-stage check independently retains its existing `1e-11` threshold. Six
persistent-layout cases (77, 86, 89, 94, 95, 99) were refused by native-gate
legalization before block optimization. These failures are retained, not counted
as acceptance and not corrected by increasing resources.

The first one-strategy-per-case exploratory driver stopped at its first resource
refusal. Its exact source and failed log are retained. The corrected exploratory
run retained 98 acceptances and two refusals; the subsequent all-strategy study
reuses the same generated inputs and attempts all 400 outcomes. No performance
conclusion follows. These are inspected development controls, not prospective
SDKBench inputs or an SDK comparison. Full-source CI and formal qualification
remain separate requirements.

The [raw archive](evidence/routing-complete-operator-controls.json) retains all
inputs/outcomes, exact drivers, original failed attempt, helper digest and source
identities. The inspected source has byte-identical production `flagquantum/`
content to integration baseline `02bacc099cdcd1ac08510c8cf944df6e018c94c9`.

## Regression coverage

The unit suite retains cases 0 and 75 for three- and four-wire acceptance and
all six inspected persistent-layout budget refusals. Each case attempts the four
strategies, for 32 added checks. Every accepted program must preserve the full
source operator up to a single global phase at the existing `1e-11` tolerance,
including nonidentity initial placement, and independently preserve the routed
operator. Every two-qubit gate must use a target edge and the final layout must
restore logical order.

The six known refusals are permitted only with the existing native legalization
exception and exact resource reason. They are also allowed to become accepted
when routing improves, in which case they must pass all operator and topology
checks. Other refusals fail the test. No baseline or synthesis budget is raised.
The recorded 100-case diagnostic keeps its original `1e-10` threshold; the new
regressions use the stricter existing helper threshold rather than relabeling the
historical evidence.

These controls support the existing compile-to-native-target workflow; they do
not change its public API, routing policy or capability maturity. Execution is
local CPU `single_device_fast_path` validation, with no distributed or hardware
claim.

Run the affected workflow checks from the repository root:

```sh
python -m pytest tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Expected behavior: the accepted source-to-topology-to-native programs preserve
complete operators, topology and layout; inspected over-budget baselines refuse
before block optimization. Broader CI is required before integration.


## Local validation

On the integration baseline after the source-staging repair, the focused unit
and physical-evidence integration files passed 163 tests. The sparse checkout's
`pr-default` tier passed 121 tests with 42 deselected, and `pr-runtime` passed
42 tests with 121 deselected. These counts describe the restored sparse scope,
not the full repository selection. Black, Ruff, language, local documentation
links and the new archive's revision vocabulary/origins checks passed.

The first focused run failed the 32 new cases because the new generator lacked
its `random` import; 131 existing cases passed. The import was corrected before
the successful run. Strict mypy remains unsuccessful: it reports 14 errors in
eight files, including pre-existing negative-test annotations and imported
production modules. The unmodified baseline file reports the same 14 diagnostics
with corresponding source lines. No ignores, contract changes or checker changes
were added.
Full-source CI remains required.
