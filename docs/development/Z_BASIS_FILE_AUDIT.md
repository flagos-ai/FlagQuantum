# Computational-basis source files

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files are
Apache-2.0, copyright Google LLC.

| Inventory | Frozen Git blob | Disposition |
| --- | --- | --- |
| Q-0273 `qualtran/bloqs/basic_gates/z_basis.py` | `48231b5d507f076ed3ea95a4d8486f375cca7d01` | Closed. Actual atom binding and complete native wire symbols supplement the accepted numerical, measurement and model owners. |
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

## Complete source-interface closure

Drawer now provides `basis_vector_wire_symbol` and `pauli_basis_wire_symbol`.
The vector callback reuses the existing directional register glyph owner and
replaces only its text attributes. It preserves integer values and source
side semantics without constructing foreign register objects. Z retains its
constant TextBox and CZ retains two filled circles with exact unknown-name
ValueError. Complete score geometry covers all scalar state/effect factories,
integer state/effect factories, Z, CZ and the classical output of MeasureZ.
MeasureZ uses the existing default register-label callback and classical lanes.

Actual Z and CZ sources pass both-direction named capture, concrete JSON,
exact single-leaf streams and complete complex operators. Reordered physical
bindings with an entangled spectator pass full-vector execution and inverse
restoration through `circuit.append(op)`. Existing source/native Cirq and
PennyLane, all raw measurement branches, zeroed retained slots and independent
costs remain the evidence for their respective owner boundaries.

| Remaining source definitions | Complete native disposition |
|---|---|
| `_ZVector`, hidden-field transformer and four subclasses/factories/DocSpecs | Existing state/effect signatures, complete tensor columns and covectors, classical guards and source constructor limits are reused. All actual factories now also compare complete directional score geometry. |
| Z/CZ signature, adjoint, tensor, classical phase and control adders | Actual source capture covers atom matrices and directions; existing classical phase and exact controlled recipes cover each model, operands, feedback and clean work. |
| Z/CZ external conversion and wire symbols | Existing SDK round trips plus actual named captures and complete source score callbacks, including unknown-register behavior |
| MeasureZ signature, classical callback, copy/discard tensor, factory and DocSpec | Existing full channel, every raw outcome, classical value and physical-slot reuse checks, plus actual factory and complete classical score |
| `_IntVector` validation, dtype, both composite builders, model, string and callbacks | Existing integer binding tests preserve big-endian Join/Split, all source small/concrete/symbolic and zero-width boundaries, complex tensor effect and independent abstract-Clifford model. Both actual eight-bit factories now supply complete score geometry and value text. |

A source atomic effect deliberately refuses Cirq conversion. For a three-bit
integer state, the actual source adjoint first returns a wrapper and consumes
its declared ports, then recursive decomposition raises the original ValueError
at `<0|`. This exact stage distinction is tested with a properly allocated
source manager. The existing bidirectional named emitter retains that failure.
It cannot turn a postselected covector into a unitary inverse. Measurement
quantum-to-classical ports similarly use the dynamic and tensor owners, while
the through-port capture boundary rejects them explicitly. These are distinct
physical boundaries, whose existing complete numerical evidence is reused.

Q-0273 closes. Q-0274 remains closed with its original-test evidence. No source
file is changed. Emitted X/Z/CZ instructions, basis loads, held-slot cleanup and
independent logical resource estimates retain the separate table above.

The whole-file batch passed 3,574 concentrated cases without skips. Its 32
warnings preserve paired source/native NumPy overflow behavior in X and CNOT
classical callbacks. Both changed Python files passed strict typing, Black
and Ruff. Compiler scope, architecture, multilevel IR and protected API checks
passed.
