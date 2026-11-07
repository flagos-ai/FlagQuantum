# bernstein vazirani

Category: algorithms / foundational.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/foundational/bernstein_vazirani/bernstein_vazirani.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/foundational/bernstein_vazirani/bernstein_vazirani.qmod'
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

Recorded resources: 5 physical qubits, 3 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_static_powers.py](../../../../../../../tests/integration/test_synthesis_qmod_static_powers.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `15320d5d6f4bc8a24563120449d99ddc55e37409`.

Original SHA-256: `13fbe0785d5230fe9b82b49360fc68957c91d507a0d0503ccaa5c605809c36c0`.

Approved SHA-256: `13fbe0785d5230fe9b82b49360fc68957c91d507a0d0503ccaa5c605809c36c0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
