# classical shadows

Category: algorithms / metrology.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/classical_shadows/classical_shadows.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/classical_shadows/classical_shadows.qmod'
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

Recorded resources: 2 physical qubits, 3 operations, depth 3.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.algorithms.metrology.classiq.paper_implementation_project.classical_shadows.classical_shadows.main
python -m examples.synthesis.algorithms.metrology.classiq.paper_implementation_project.classical_shadows.classical_shadows.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_bell.py](../../../../../../../../tests/integration/test_synthesis_qmod_bell.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `ea21facd0aec2d550ae8eed1e9e9deaebf30e796`.

Original SHA-256: `9cc3ae7502ec4cddec362a755fd4480ea642333b159cf11e8ae333c9e03ab8a9`.

Approved SHA-256: `9cc3ae7502ec4cddec362a755fd4480ea642333b159cf11e8ae333c9e03ab8a9`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
