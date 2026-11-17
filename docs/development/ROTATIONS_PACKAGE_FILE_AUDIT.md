# Rotations package source-file audit

Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`
contains seven import statements and eleven class exports in
`qualtran/bloqs/rotations/__init__.py`. The unchanged file has Git blob
`e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8`. Q-0666 covers that
initializer. The native owners are internal compiler modules under
`flagquantum/compiler/registers/phase/`. Notebook operations still enter
through `circuit.append(op)`.

| Locked export | Native owner and entry | Executed notebook path |
| --- | --- | --- |
| `HammingWeightPhasing` | `hamming_weight_phasing.hamming_weight_phasing_ir` | Hamming weight, direct |
| `HammingWeightPhasingViaPhaseGradient` | `hamming_weight_phasing.hamming_weight_gradient_phasing_ir` | Hamming weight, gradient |
| `AddIntoPhaseGrad` | `phase_addition.phase_add_ir` | Phase gradient, plain addition |
| `AddScaledValIntoPhaseReg` | `scaled_phase_addition.scaled_phase_add_ir` | Phase gradient, scaled addition |
| `PhaseGradientState` | `phase_gradient.phase_gradient_state_ir` | Phase gradient, state preparation |
| `PhasingViaCostFunction` | `cost_function_phasing.cost_function_phasing_ir` | Cost function, square production and erasure |
| `ProgrammableRotationGateArray` | `programmable_rotation_array.programmable_rotation_array_ir` | Programmable rotation gate array |
| `QvrPhaseGradient` | `qvr_phase_gradient.qvr_phase_gradient_ir` | Quantum variable rotation, gradient |
| `QvrZPow` | `qvr_zpow.qvr_zpow_ir` | Quantum variable rotation, direct |
| `RzViaPhaseGradient` | `rz_phase_gradient.rz_via_phase_gradient_ir` | Rz via phase gradient |
| `ZPowConstViaPhaseGradient` | `zpow_phase_gradient.zpow_via_phase_gradient_ir` | ZPow via phase gradient |

The source check compares the blob, exhaustive import AST and runtime object
identity. The native check compares each consumer binding to the compiler
owner. The cost-function route reaches the owner through
`square_cost_phasing`. The seven notebook kernels exercise the paths above.

The initializer has no gate stream, register layout or separate resource
cost. Its routes reuse each module's documented gate, complex amplitude,
work-bit and resource evidence in Q-0669, Q-0672, Q-0675, Q-0680, Q-0683,
Q-0686 and Q-0689. Outstanding symbolic execution, source decomposition
and register binding limitations remain under the implementation items
Q-0668, Q-0671, Q-0674, Q-0679, Q-0682, Q-0685 and Q-0688.
