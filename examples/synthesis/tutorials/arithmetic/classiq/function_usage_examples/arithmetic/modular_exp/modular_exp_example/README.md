# modular exp example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
null
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 7 physical qubits, 18 operations, depth 13.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_modular_exp.py](../../../../../../../../../tests/integration/test_synthesis_qmod_modular_exp.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `878bfca97f83fcdbfc1d27efcdd342ea3d53b8d9`.

Original SHA-256: `e67dd3b8cfb6d161076583dcc8c3da15f667eb962841d7d9fa40477386962845`.

Approved SHA-256: `e67dd3b8cfb6d161076583dcc8c3da15f667eb962841d7d9fa40477386962845`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
