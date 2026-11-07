# harmonic oscillator

Category: applications / differential_equations.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator.qmod'
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

Recorded resources: 7 physical qubits, 3821 operations, depth 3339.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_guarded_unitary.py](../../../../../../../../tests/integration/test_synthesis_qmod_guarded_unitary.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `99a80d230cacd164050598a7d1cef83acaa84b08`.

Original SHA-256: `2b4b1f4b11dc040aac3ab1b516a88e4aba185f84f8f683910db8c542c0672eca`.

Approved SHA-256: `2b4b1f4b11dc040aac3ab1b516a88e4aba185f84f8f683910db8c542c0672eca`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
