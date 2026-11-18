# Complete basic-gate package exports

Q-0229 covers all 42 exports in the original basic-gate initializer at locked
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`e093c1d7663e8019904a93a5c565e87c256ed724`. The source test preserves the
complete import order and verifies every export's identity with its defining
module. The introductory module docstring has no executable responsibility.

Paths below are relative to `flagquantum/compiler/registers` unless a different
owner is named. Groups enumerate every original export exactly once.

| Original exports | Authoritative native owner and retained consumer |
| --- | --- |
| CNOT | `primitives/controlled_not.py`, Core CX and Simulation factorized CNOT. `test_cnot_file_audit` preserves ordered controls, exact tensors, callbacks, adapters and Bell append. |
| Discard | `component_classical.py`, Simulation trace. Existing discard contract and tensor-trace tests preserve LEFT classical consumption and unnormalized trace. |
| DiscardQ | `primitives/measurement.py`, Simulation quantum trace. Existing discard tests distinguish physical reset, trace and projection. |
| GlobalPhase | `primitives/global_phase.py`, scalar carrier binding. Complete phase tests retain absolute phase and observable controlled relative phase. |
| CHadamard, Hadamard | `primitives/controlled_hadamard.py`, Core H and native source capture. Hadamard and controlled-component consumers retain CH atomic lowering and distinct source cost. |
| Identity | `primitives/identity.py`. Complete identity suite retains per-wire i emission, exact tensors and zero source QEC cost. |
| OnEach | `bits/on_each.py`, Simulation classical OnEach. File and typed-binding suites retain Split/child/Join order and numerical callback behavior. |
| Power | `component_power.py` and `primitives/power.py`. Complete Power tests retain repetitions and exact measured adjoint. |
| CRz, CZPowGate | `primitives/controlled_rotation.py`. Rotation suites retain two half-angle phases/two CX or And/phase/measured cleanup, respectively. |
| Rx, Ry, Rz | `primitives/pauli_rotation.py`. Source matrix and capture tests preserve finite angle conventions and symbolic model metadata. |
| XPowGate, YPowGate, ZPowGate | Same Pauli-power owner. Controlled phase tests preserve the source global-shift convention. |
| SGate, TGate | `primitives/pauli_gate.py` and phase cost identities. Existing two-direction and mixed-control suites preserve source phase and exact feedback. |
| SU2RotationGate | `phase/su2_rotation.py`, Simulation matrix and Ecosystem source boundary. Complete suite retains scalar/ZXZ decomposition, recovered matrix and precision allocation. |
| CSwap, TwoBitCSwap | `primitives/controlled_swap.py` plus broadcast binding. Exact Fredkin remains distinct from the relative-phase four-T implementation. |
| Swap, TwoBitSwap | `primitives/swap.py`. Shared broadcast and classical-swap suites retain operand order and occupied targets. |
| Toffoli | Core CCX and `primitives/broadcast_ccx.py`. Source capture/control/tensor suites distinguish an atomic CCX from synthesized T gates. |
| MeasureX, MeasureZ | `primitives/measurement.py` and `component_measurement.py`. Existing branch/channel tests preserve unnormalized amplitudes, surviving order, feedback and cleared physical wire. |
| MinusEffect, PlusEffect | Simulation basis-vector effects and Compiler `component_basis_vectors.py`. Effect suites preserve LEFT consumption and zero-probability vectors. |
| MinusState, PlusState | `primitives/basis_state.py` and component basis-vector binding. State/tensor suites preserve RIGHT output and signed X amplitudes. |
| XGate, YGate, ZGate | `primitives/pauli_gate.py` and Core atomic opcodes. Existing source capture/control suites preserve exact matrices, phases and dedicated control leaves. |
| CYGate, CZ | Same Pauli/control owners. Source gates and feedback CZ retain separate physical roles. |
| IntEffect, IntState | Compiler basis-vector bindings and Simulation integer basis tensors. Existing suites preserve MSB-first values, signature width and projection without normalization. |
| OneEffect, ZeroEffect | Same basis-vector/effect owners with scalar Z ports. |
| OneState, ZeroState | Same basis-state owners with scalar RIGHT Z ports. |

## Interface and direct-dependency evidence

Users continue to append native operations. Core owns signatures/opcodes,
Compiler owns physical binding and measured erasure, Simulation owns numerical
callbacks and tensors, and Ecosystem terminates external objects. The package
is integrated through those owners. No original package-name mirror, second IR
or duplicate basic-gate dispatcher is introduced by this initializer closure.

The concentrated suites reuse actual source calls and primitive operand order,
controlled global phase, complex spectator states, applicable small measurement
histories and clean scratch. LEFT effects and trace are not physical unitary
inverses. Measurement outcomes remain unnormalized when comparing branches.

The initializer has no gate body, physical work or measurement branch. Individual
primitive inventories remain separate from logical QEC/peak models. Examples
include atomic CCX versus a declared Toffoli cost, atomic CH versus declared
Clifford/rotation costs, reset versus quantum trace, and per-wire identity
emission versus zero source QEC cost. No package-level total is asserted.

Q-0229 closes the entire export responsibility. Original child parents retain
their documented symbolic/foreign/static admission, typed OnEach, source
measurement and discard discovery, generic control, external-hook and drawing
boundaries. Existing whole-file audits remain authoritative for those details.
Source skips/xfails remain visible. No frozen source was modified.

The shared batch passed 792 concentrated native/reference cases without skips
or warnings. All four changed Python test files passed strict mypy, Black and
Ruff. Compiler scope, architecture, multilevel IR and protected API checks
passed. The source initializer blobs and child reference files were unchanged.
Logs are `ftqc-reference-env/g2-package-exports-*.log`.
