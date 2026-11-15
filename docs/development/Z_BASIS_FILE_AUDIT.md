# Computational-basis source files

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files are
Apache-2.0, copyright Google LLC.

| Inventory | Frozen Git blob | Disposition |
| --- | --- | --- |
| Q-0273 `qualtran/bloqs/basic_gates/z_basis.py` | `48231b5d507f076ed3ea95a4d8486f375cca7d01` | Open for generic foreign object dispatch, wire-symbol protocols and external physical binding. |
| Q-0274 `qualtran/bloqs/basic_gates/z_basis_test.py` | `35de62edcda9ce3230d31012f1fe757c4572bc38` | Closed. All original scenarios map to native evidence, including the two previously migrated notebooks. |

## Binding and dependencies

The basis state creates a right `q` port on a clean wire. The effect consumes
a left `q` port and returns an unnormalized overlap. Integer variants create
or consume the ordered big-endian `val` port. The circuit loader uses the
compiler bit bindings, the tensor projection uses the simulation contraction
owner, and the classical callbacks check concrete values before consumption.
Z and CZ preserve the through-port basis values and apply the phase
independently. Positive scalar control on Z binds to the atomic CZ with
`q1, q2` order. Additional controls reuse the accepted conjunction and
measurement-feedback paths with independently owned predicate workspace.

Logical MeasureZ consumes quantum `q` and returns classical `c`. The native
Program path consumes the quantum identity. A separate circuit path clears the
physical slot by applying a conditional X after the measurement. No additional
workspace is allocated by either atomic measurement path. The held-slot
operation can be appended to a dynamic circuit through the same append boundary
used by the other register operations.

## Original test coverage

| Original tests | Native evidence |
| --- | --- |
| `test_zero_state`, `test_zero_effect`, `test_one_state`, `test_one_effect`, `test_zero_state_manual`, `test_multiq_zero_state`, `test_one_state_manual`, `test_zero_effect_manual`, `test_one_effect_manual`, `test_zero_state_effect` | Existing scalar left and right ports, callbacks, input refusals, complex columns, covectors and zero modeled cost. Both matching ket/effect compositions give scalar one. |
| `test_int_state`, `test_int_effect`, `test_int_state_manual`, `test_int_effect_manual` | Existing ordered Join/Split and basis leaves, full complex emittance and projection, matching and invalid value checks. The exact eight-bit source value 255 now joins the gate and tensor comparison. The native integer call model independently verifies one abstract Clifford. |
| `test_zgate`, `test_to_cirq`, `test_pl_interop`, `test_zgate_manual`, `test_cz_manual`, `test_cz_phased_classical` | Zero and one followed by Z yield the original Z and X/Z gate streams and full complex vectors. Atomic Z/CZ have native Cirq and PennyLane round trips. Existing phase-aware callbacks, full matrices, QEC comparisons and control-history tests cover exact signs and predicate cleanup. |
| `test_meas_z_supertensor` | All four original Zero, One, Plus and Minus preparations give identical complete classical density matrices. Existing copy tensor and environment-trace comparisons cover the full channel and pure-contraction boundary. Existing unnormalized branch tests cover complex spectator amplitudes. |
| `test_meas_z_classical` | The unchanged original measurement word 52 maps to the same ordered eight slots on reversed physical wires. The measured physical slots return to zero. |
| `test_states_and_effects_notebook`, `test_measurement_notebook` | The previously accepted native `register_states_effects.ipynb` and `register_measurement.ipynb` preserve the source topics and independently execute. See the corresponding notebook audits. |

| Operation | Emitted primitives | Independent source resource model |
| --- | --- | --- |
| Zero / One preparation | 0 / 1 X | Zero QEC gates |
| Eight-bit integer 255 preparation | 8 X | 1 abstract Clifford, 8 qubits |
| Z / CZ | 1 Z / 1 CZ | 1 Clifford each |
| Eight-bit word 52, measurement and slot reuse | 3 preparation X, 8 measurements and 8 conditional X instructions | 1 abstract preparation Clifford and 8 logical measurements |

The unchanged non-notebook source run reports 34 passes, 15 mode skips and
two expected failures for the disabled integer counts checks. The two source
notebook tests use the independent acceptance evidence from their tutorial
batches. The 929-case concentrated native run and both native notebook reruns passed without skips.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_measurement.ipynb
```
