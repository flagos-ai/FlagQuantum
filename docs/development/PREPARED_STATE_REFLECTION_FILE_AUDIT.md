# Prepared-state reflection source closure

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Google LLC 2023
Apache-2.0 attribution is retained in the native implementation.

| Inventory | Frozen Git blob | Responsibility |
|---|---|---|
| Q-0663 `reflection_using_prepare.py` | `4a8467fde126bd596be69286d88c198726a4557a` | Complete declaration, binding, controls, factory metadata, phase and declared resources. |
| Q-0664 `reflection_using_prepare_test.py` | `df403c18060daf172e68c4a80099c306743a7a3a` | All original tests, including the restored notebook dependency, run unchanged. Original xfails are tracked below. |
| Q-0665 `reflections.ipynb` | `e4d3316f906b697634149f4e61903333d947e7af` | Frozen notebook restored. Native tutorial migration is the next independent module. |

## Ownership and complete member mapping

`algorithms/primitives/prepared_state_reflection.py` owns
`PreparedStateReflection`. The existing compiler preparation recipe owns the
unprepare/mark/prepare instruction order. Core owns register metadata, signature
validation and IR. Existing packed PREPARE binding owns named child ports and
native directional callbacks. Foreign objects stop at the ecosystem snapshot
importer. This is a `single_device_fast_path` with explicit workspace ownership.

| Original member | Native mapping and evidence |
|---|---|
| fields and defaults | `prepare_gate`, `control_value`, `global_phase`, `precision`; both original PREPARE forms admitted. |
| control/selection registers, signature | Ordered metadata retained, including symbolic declarations. |
| `reflection_around_zero` | `around_zero` composes the accepted `IdentityPrepare`. Empty selection is supported. |
| `decompose_from_registers` | `make_on` binds named ports, invokes inverse PREPARE then forward PREPARE, and composes the existing zero-selective reflection. |
| circuit diagram info | `wire_symbols` preserves positive/open control and repeated `R_L` labels. Original three decomposed diagrams execute in the source suite. |
| `build_call_graph` | `resource_model` delegates to `source_reflection_declared_cost`; 90 width/control/coefficient fixtures match original QEC and peak queries. Existing symbolic zero-reflection formula remains independently available from the compiler. |
| `adjoint` | Same-instance source declaration retained. Physical inverse uses `make_on(adjoint=True)`. |
| `get_ctrl_system` | First single control specializes the declaration. Grouped/repeated controls reuse accepted shared control binders, with both existing polarities, original controlled tensors and every small erasure history verified. |
| two factories and DocSpec | Five equal weights at precision .01 and widths 1/2/3 at coefficient -1/control 1 preserve signature, normalization, declared costs and wire labels. |

Logical words use explicit MSB-first wire order. Reflection scratch starts clean
and ends zero. Preparation junk is an explicit logical child output whose
lifetime follows that child, as in the existing walk binding. Native binders
validate port and workspace sizes. A non-identity named PREPARE requires a native
callback, while an already packed PREPARE owns its callback. Symbolic metadata
remains declarative until concrete widths are supplied.

```python
import flagquantum as fq
from flagquantum.algorithms.primitives.prepared_state_reflection import PreparedStateReflection

reflection = PreparedStateReflection.around_zero((1,), global_phase=-1)
circuit = fq.Circuit(2).h(0)
circuit.append(reflection.make_on(
    selection={"reg0_": (0,)}, reflection_workspace=(1,)))
# Selected amplitudes become (1, -1)/sqrt(2); borrowed qubit 1 returns to zero.
```

The complete operator and coherent spectator states are checked for widths 0–3,
three control choices and five coefficients. Every logical input column and every
small forward measurement branch retain their full complex amplitudes. Every
combined forward/inverse measurement history restores input amplitudes and
borrowed workspace. JSON round trips, ordinary append, named provider ordering
and packed source-capture provider replacement are included. Existing tests
retain exact comparable mark primitive order, operands and measurement feedback,
plus the nontrivial three-state preparation composition.

## Separate resource and execution accounts

The original declared dictionary collapses PREPARE and its adjoint when they are
equal keys. The native physical scan still executes both directions. At open
control with coefficient one the original graph omits two physically executed
X gates. Both quirks remain explicit reference assertions. Arbitrary controlled
phases use the shared controlled scalar model. Uncontrolled scalar phases have
zero QEC declaration cost. Neither convention substitutes for counting emitted
instructions.

For width three the shared native mark emits eight T-family primitives and two
measurements, with two controlled scratch qubits or three uncontrolled scratch
qubits including the phase anchor. Identity PREPARE contributes two explicit
passes of one-wire identity instructions. The accepted three-state composition
emits 44 T-family primitives, eleven measurements and four Rz instructions.

The actual Cirq capture of a two-bit identity reflection controlled on one with
coefficient i reserves zero work and zero measurements, and uses a coherent
synthesis stream with local matrix gates. The native Bloq mark path uses one
clean work qubit and one measurement. Each source capture stream is preserved
and independently matches its complete operator. These lowering inventories
remain separate. The comparable original MultiControlZ call chain is checked at
its shared Bloq primitive level by `test_prepare_reflection.py`.

## Source defects and original test boundaries

Original `adjoint()` returns self even for coefficient i. Its product with the
original forward matrix is -I. The declaration preserves that behavior while
the compiler mathematical inverse conjugates the coefficient.

The unchanged original suite passes fourteen cases including its notebook and retains three original
xfails, including two alias-reflection simulations marked "Broken Cirq
simulation" and one factory counts mode. The unreachable Dirac-notation helper
belongs to those original xfailed simulations. No successful execution is claimed
for those cases. The restored original notebook test passes in an isolated reference kernel.
Its native tutorial migration remains Q-0665, independently open.

The native batch uses complete frozen source hash assertions, focused regression,
strict typing, Black/Ruff, owning-team scope, architecture, multilevel IR and
protected API checks. The concentrated regression passes 282 cases without skips or warnings.
All four changed Python files pass strict mypy, Black and Ruff. Both owning-team
checks, architecture, multilevel IR and protected API checks pass. The unchanged
source suite retains its two unknown-marker warnings.
