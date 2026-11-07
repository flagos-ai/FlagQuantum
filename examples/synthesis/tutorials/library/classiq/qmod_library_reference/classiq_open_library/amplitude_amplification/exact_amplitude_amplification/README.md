# exact amplitude amplification

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/amplitude_amplification/exact_amplitude_amplification.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](../../../../../../../algorithms/amplitude_amplification/exact_amplification/main.py) |
| Circuit comparison | [Runnable three-form comparison](../../../../../../../algorithms/amplitude_amplification/exact_amplification/comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/amplitude_amplification/exact_amplitude_amplification.qmod'
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

Recorded resources: 2 physical qubits, 26 operations, depth 19.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.algorithms.amplitude_amplification.exact_amplification.main
python -m examples.synthesis.algorithms.amplitude_amplification.exact_amplification.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_qmod_exact_amplification.py](../../../../../../../../../tests/integration/test_synthesis_qmod_exact_amplification.py)
- [validate_qmod_exact_amplification.py](../../../../../../../../../tools/validate_qmod_exact_amplification.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `4e37c930f338b7a4d24a54f0eb9a208d062138ad`.

Original SHA-256: `d9240f1f7eb8ec93a9cd3a2aa26c561961ac8063b6d118c2a220a5f71f373475`.

Approved SHA-256: `d9240f1f7eb8ec93a9cd3a2aa26c561961ac8063b6d118c2a220a5f71f373475`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
