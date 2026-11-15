# Phase block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0288 | `phase.ipynb` | `c832c8d59cbbfcebd8210010b9aa44af23655ec4` | Closed by the full native tutorial. |
| Q-0289 | `phase.py` | `d7e6b03237f63238e511a4e7aa5056808a646593` | Open for generic controlled wrappers, zero-wire numerical execution, and the original packed-child binding failures. |
| Q-0290 | `phase_test.py` | `c41738844b4eb08517b4c3245d7cc47f1d6bdf1b` | Closed by the original source scenarios mapped below. |

## Binding and dependency ownership

The phase wrapper forwards the ordered system, ancilla and resource widths,
normalization and the original signal preparation. It adds its error term
to the child's error metadata. The child owns every through port, all
temporary work and results. The wrapper uses a caller-chosen existing
wire as its physical phase carrier and needs no additional work. It emits
the phase before the child in the forward direction and after the
bound child adjoint in the inverse direction.

The original logical graph contains one Hadamard-containing child and one
wire-free GlobalPhase atom. The scalar's lack of quantum operands lets the
reference graph traverse it independently of the child. The physical carrier
has an explicit instruction order on its chosen wire. The original quarter
phase and the original negation both preserve the full absolute complex
matrix, including the signs of its entries.

| Original tests | Native verification |
| --- | --- |
| `test_phase`, `test_phase_signature`, `test_phase_params` | Exact one-qubit Hadamard fixture, through-port and source child calls, normalization/error and zero auxiliary widths, and the existing native emitted-IR codec round trip. |
| `test_phase_tensors`, `test_negate` | Quarter phase and negation, exact absolute matrix and adjoint, owned primitive carrier and H operands, independent QEC model and logical width. |
| `test_phase_signal_state` | Source custom oracle's one-bit selection and five-bit junk widths. Native forwarding retains the identical supplied preparation object and its callable H behavior. |
| `test_notebook` | Independent execution of the ten-cell frozen notebook and the twelve-cell native counterpart, with original signature and call graph and a complete phase-sensitive append and inverse example. |

| Fixture | Physical emission | Independent source model |
| --- | --- | --- |
| Quarter phase or negation of Hadamard | Two X, two phase, one H, zero measurements, zero temporary work | One Clifford, one peak logical qubit; the scalar contributes zero QEC categories |

The unchanged source baseline passes nine non-notebook cases with one
expected upstream serialization failure. The native codec evidence is for
the emitted CircuitIR. The shared numerical and ownership acceptance for
T, CNOT, measured Add, nested phase, and generic SELECT/LCU consumers remains
in [Phase block encoding](PHASE_BLOCK_ENCODING.md). In cases where the
unchanged source refuses named children under packed phase ports, the existing
evidence preserves the failure and tests the native composition against the
unchanged child tensor and scalar independently.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_phase_block_encoding.ipynb
```
