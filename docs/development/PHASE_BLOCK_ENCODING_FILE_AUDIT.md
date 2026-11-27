# Phase block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0288 | `phase.ipynb` | `c832c8d59cbbfcebd8210010b9aa44af23655ec4` | Closed by the full native tutorial. |
| Q-0289 | `phase.py` | `d7e6b03237f63238e511a4e7aa5056808a646593` | Native definitions and scalar binding covered; remains open for the original packed-child binding failures. |
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

## Complete implementation map

| Source definition | Native owner and evidence |
| --- | --- |
| `alpha`, `system_bitsize`, `ancilla_bitsize`, `resource_bitsize`, `epsilon`, `signal_state` | `PhaseBlockEncoding` forwards the child's metadata and adds `eps`; existing metadata, symbolic and custom signal tests are reused. |
| `signature` | `source_phase_block_signature` preserves packed port order, zero-width omission and symbolic width expressions. |
| `build_call_graph` | Independent `resource_model` retains one scalar and one child call; controlled and symbolic declarations retain their source cost profiles without deriving costs from emission. |
| `build_composite_bloq` | `make_on` emits the scalar before the bound child; the adjoint reverses their order. Generic controlled binding uses the existing predicate owner. |
| `__str__` | `label(child_label)` preserves the original caption, including symbolic phi. The caller supplies the child symbol so no external SDK object enters Algorithms. The exponent is in pi units despite the abbreviated source caption. |
| `_phase_block_encoding`, `_PHASE_DOC` | The actual original H factory and its single documented example are checked against native child replacement, full operators, models and all four source snapshots. |

## Zero logical ports and physical storage

`import_block_encoding` now captures legitimate scalar BlockEncoding objects
with empty signatures. Ordinary forward/adjoint snapshots reserve one explicit
clean workspace carrier, including for an empty identity body. Controlled
snapshots use the existing control instead. `CircuitBlockEncoding` permits
nonnegative concrete widths, omits absent system ports at child binding, and
keeps source models independent of physical storage. Core still requires a
positive physical wire count; this does not introduce a zero-wire Circuit or IR.

The scalar fixture uses the unchanged source BlockEncoding/GlobalPhase APIs.
Both accepted stock identity preparations, zero/one auxiliary widths, both
directions and controlled/ordinary interfaces are exercised. Complete complex
operators and CircuitIR round trips preserve arbitrary states on through ports
and scalar carriers. Owned clean work is restored; no measurement, reset or
unrequested control is introduced. Existing measured Add/SELECT/LCU tests
continue to cover branch feedback and nontrivial coherent work ownership.

| Scalar case | Actual primitive emission | Independent source model |
| --- | --- | --- |
| Nonzero scalar, ordinary | 2 X + 2 phase; one physical carrier when all logical widths are zero | Zero QEC gates and zero peak logical qubits for the empty logical signature |
| Nonzero scalar, positive-controlled | One phase on the control; no additional carrier | Original controlled scalar model, including its logical control |
| Child scalar -0.2 plus wrapper +0.25, ordinary | 4 X + 4 phase; full operator exp(i*pi*0.05) I | Zero QEC gates and zero peak logical qubits |
| Empty scalar identity | Zero instructions; one explicitly reserved physical carrier | Zero QEC gates and zero peak logical qubits |

### Remaining source boundaries

The original `Phase(Unitary(GlobalPhase(...)))` still raises a missing-system
TypeError. This differs from the valid scalar BlockEncoding fixture. Original
named SELECT/LCU children still fail under Phase's packed ports, and the original
QubitCount query fails on that route. Native composition is checked against the
unchanged child and scalar separately; it is not labelled unchanged whole-source
execution. Q-0289 remains open under the frozen source-defect closure criteria.

Unresolved symbolic phase values remain declaration-only. In this reference
environment, tensor contraction of a SymPy-valued scalar also fails at the
autoray SymPy backend; native numerical admission rejects it. This diagnostic
does not establish a mathematical or Cirq execution mismatch, and the reference
is not modified or silently coerced.

## Current batch acceptance

The scalar/factory file passes all 30 cases. The concentrated 407-case block,
LCU, sparse and feedback regression passes without skips (51.94 s). All 169
Algorithms and touched adapter files pass strict typing. Black, Ruff, both team
scopes, architecture, multilevel IR and protected API checks pass. The existing
append example runs and distinguishes the four-T source estimate from its
additional scalar representation instructions. CPU/representation integration
passes 60 cases (45.45 s); the unchanged native tutorial passes in an independent
kernel (2.78 s).

Serial scalar/snapshot coverage passes 36 cases (4.02 s), covering 78% of the four
entire modules; this is not a claim of complete whole-module branch coverage.
The first concentrated run passed 406 cases and exposed the TensorProduct
zero-width test's obsolete adapter-rejection assertion. The corrected test
checks the original missing-system TypeError now reached after scalar admission.
The unchanged upstream failure is retained rather than repaired or suppressed.


## Faithful complete-source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes the historical open
status for Q-0289. Complete source review covers all metadata and signal
forwarding, packed signature, scalar/child call graph, complete composite body,
caption, original factory and DocSpec. The later [constructor audit](PHASE_BLOCK_CONSTRUCTOR_AUDIT.md)
adds source-ordered exact ordinary/positive owners and predicate controls.
The forward controlled source visits child then phase; controlling an inverse
and inverting a controlled graph remain distinct recipes. Existing implementations
and full-complex/branch evidence cover those paths; no new algorithm is added.

The wire-free scalar stays a logical atomic boundary. Its actual native carrier
instructions and clean work are reported separately from the source's zero QEC
cost. Child through ports and private work keep their original ownership.
Complete small operators, measured-child history certificates, numerical
private-zero checks, nonlinear phase binding, source-free append and IR
restoration are reused. Unknown children require explicit owned exact pairs;
matching signatures alone never establish their bodies or inverses.

Original packed Phase with named SELECT/LCU ports, missing-system scalar
Unitary, related QubitCount and deep tuple-codec failures remain unchanged.
Independent child-plus-scalar execution is not called successful whole-source
execution for those failures. Unresolved symbolic numerical/SDK boundaries
remain explicit. The modern native profile and source analytical categories
are not inferred from counts of physical phase instructions.

Batch validation passes 256 concentrated LCU/Phase/rewrite cases and 44
independent Cirq 1.6 constructor/snapshot cases. All 76 reviewed frozen blobs
are unchanged. Existing strict production, 32 local integration, unchanged
original tests/notebook and actual Kickmix Simulator evidence is reused because
no Python implementation changes in this acceptance entry.
