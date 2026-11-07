# Structured controls for linear-combination block encodings

G2-BE09 consumes the original LinearCombination.get_ctrl_system chain from
Qualtran v0.7.0, commit bcb17bb25360ed68ce3bc084645db3e929799389. Source blobs
match the pinned inventory and both local reference locations:

- LinearCombination: a3d8aa7fafa7a6f3fcaf095826751209427cd255;
- ControlledViaAnd: 3947ddf86be37f4ff3ba58fcc316a8a64922f84f;
- controlled.py dispatch: 6fabe760e793f1b39f825d76f80e92b6e5fa5c28.

The actual fallback dispatch matters. A single positive control selects the
specialized positive parent. Other quantum control specifications select
ControlledViaAnd: a single negative control flips before and after that parent;
multiple controls compute CtrlSpecAnd once, use the positive parent, then erase
the predicate. The parent contains unconditional PREPARE, positive-controlled
SELECT and unconditional PREPARE-adjoint. Its adjoint preserves this wrapping
and takes the original directional child path; it does not reverse measured IR.

Native LinearCombinationBlockEncoding.make_on accepts existing ControlRegister
groups, physical controls, control_workspace and control_classical_bits. It
uses the existing controlled_component_ir, not a new control graph. The wrapper
binds the existing positive path once. It never adds coherent controls to reset
or measurement instructions. A direct positive control tuple and structured
control groups are distinct binding modes and cannot be supplied together.

Control storage is owned separately from system/ancilla/resource and every
stage workspace. For n external bits, a multi-bit predicate reserves n-1 clean
qubits and classical slots. These qubits stay live across both preparation
calls and SELECT. Stage work can be reused between sequential calls, but cannot
alias the live predicate or original controls. Private predicate and stage work
must clear; logical signal/junk/failure outputs retain all complex amplitudes.

Models use existing control_predicate_cost and CostCall live bystanders. Supply
the directional positive singly-controlled SELECT model. For n>1, predicate
calls retain the payload as bystanders and the positive parent retains the
other 2n-2 live control/predicate wires. Width takes a maximum over calls.
legacy_shims selects the predicate profile only; callers supply all child
models in the same profile. No model is inferred from emitted gate counts.

Nine specifications cover positive/negative scalar bits, vector and matrix
shapes, unsigned integer words, mixed signed/bit registers, one's complement,
and stored Montgomery/bounded integer words. Across two directions and default
or supplied oracles there are 36 original gate configurations. Default factories
use eight-bit coefficient precision for gate/model checks; supplied oracles
also compare both QEC profiles, for 54 resource comparisons. Gate dependencies,
operands, measurements, feedback and original peak widths match.

Twenty independent original full-matrix cases cover arbitrary complex logical
states and every small measurement history, including control superpositions,
nonzero junk and entangled bystanders. Full vectors retain all private work and
unnormalized branch probabilities, with no phase alignment or partial trace.
Default-factory state tests use one-bit coefficient precision; eight-bit parity
comes from complete source gate flows. Six additional cases execute the
original get_ctrl_system returned adder and compare its composed matrix with
the returned controlled root. This verifies the source caller interface as well
as inspecting the root object. Invalid storage/mode and no-emission model tests
are separate from those quantum correctness checks.

Run:

```sh
python -m examples.algorithms.linear_combination_controls
```

The user still writes circuit.append(op). The two-bit control is coherently
spread over all four words and activates on 01. The example checks the projected
(X-3Z)/4 block on that word, identity on inactive words, clean shared predicates
and stage scratch, then full exact inverse restoration.

| Quantity | Original default model | Emitted IR |
|---|---:|---:|
| T equivalents / T and T-dagger | 24 | 30 |
| Measurements | 4 | 4 |
| Peak logical qubits / reservation | 11 | 11 |

The six-T difference is the original abstract four-T versus exact seven-T
Fredkin cost in two PREPARE calls. Shared predicate gates themselves match the
original primitive decomposition. No physical hardware measurement is implied.

G2-BE09 accepts external structured quantum controls on an uncontrolled parent.
G2-BE10 below adds the precontrolled specialization. Classical controls remain
a separate source path. Original child cost failures and zero-total-ancilla
boundaries remain explicit; this is not whole-file or G2 completion.


## Extending an existing positive control (G2-BE10)

Use `existing_control=(wire,)` with the new structured `controls` and
`control_registers` in `make_on`. This binds the source
`LinearCombination(is_controlled=True).controlled(spec)` root. The existing
control is positive; it is a logical input/output, not a private predicate.
`control=(wire,)` remains the distinct direct-binding mode and cannot be
combined with either extension mode.

The shared `extend_controlled_component_ir` reserves one work wire and feedback
slot per new control bit. A single new bit computes the original two-input And
with its literal activation, calls the original positive parent, then erases
that And. Multiple new bits preserve the outer CtrlSpecAnd and inner existing-
control conjunction. New controls precede the existing control in the logical
layout, and the inner predicate is the last wrapper work wire. Parent work stays
separate from stage work across unconditional PREPARE and unprepare.

`adjoint=True` takes the direction of the specialized root: add controls first,
then take its adjoint. This preserves the original And/erasure implementation
around a directionally bound positive LCU. It does not request the different
source chain that takes the adjoint before a generic negative-control wrapper.
The latter chain remains outside this binding. No measured instruction stream
is mechanically inverted.

Pass `existing_control=True` to `resource_model` with matching structured words
and directional, profile-specific positive SELECT/PREPARE/unprepare children.
Both structured modes now use the shared controlled-component cost composition
instead of a copied wrapper formula. Logical payload width comes from actual
ports rather than a child's peak allocation. The default alias convenience
model supplies the modern QEC profile; the legacy default-oracle composition
tests explicitly supply original legacy child models. They certify composition
of those inputs, not an unimplemented legacy alias factory.

Validation covers nine shaped/integer control specifications, default eight-bit
coefficient and supplied oracles, both directions, exact source primitive counts,
wire dependencies and feedback. There are 36 original gate configurations and
72 QEC-category/peak-width comparisons. Fourteen independent full original
matrices exercise arbitrary complex logical inputs, all small measurement
histories, live junk and clean private storage. Invalid existing-control/storage
bindings fail, and resource-only queries never emit a circuit. All 131 new tests
pass; the concentrated regression passes 717 tests without skips. Strict types,
formatting, lint, scope and architecture checks also pass. Source blobs remain unchanged: LinearCombination
`a3d8aa7fafa7a6f3fcaf095826751209427cd255`, specialized_ctrl
`98b64995010da5863218dbf93b39bbffbd600a2f`.

```sh
python -m examples.algorithms.linear_combination_controls --stacked
```

Users still append the bound operation. The example keeps all eight new/old
control words coherent, validates the projected block and restores the entire
complex state with the adjoint. Its model reports 28 T equivalents, five
measurements and 13 peak logical qubits; its emitted IR contains 34 T/T-dagger
instructions. The existing six-T Fredkin accounting difference remains explicit.
G2-BE11 below supplies repeated scalar root execution; its original returned-adder
defect and classical controls remain separate boundaries. This does not certify
the whole file.


## Repeated scalar control roots (G2-BE11)

The next source call after a two-control root does not create another nested
predicate. `_MultiControlledFromSinglyControlled.get_ctrl_system` prepends the
new scalar activation to its existing `cvs` vector. Its returned root computes
one MultiAnd over the full ordered vector, invokes the original positive LCU,
and erases that predicate. No algebraic flattening of a grouped-control request
is used to infer this path.

Bind the previous root's packed logical wires with `existing_control` and its
literal newest-first activations with `existing_control_values`. Values must
end in one, the original positive LCU control. Add exactly one structured bit
with the ordinary `controls` and `control_registers`. For k existing bits this
reserves k clean wrapper work wires and k feedback slots. Those outputs enter
and leave zero, while logical controls, signal and junk remain logical state.
The compiler's existing `controlled_component_ir`/MultiAnd emits the full vector;
`controlled_component_cost` supplies the matching model and private lifetimes.
The algorithm only validates/binds these components. `adjoint=True` follows the
returned root's adjoint and asks the positive child for its own direction.

```python
# Existing root 01, then prepend a new zero control: ordered root 001.
op = encoding.make_on(
    binders,
    system=system, ancilla=ancilla, resource=resource,
    controls=((0,),), control_registers=(ControlRegister(1, (0,)),),
    existing_control=(1, 2), existing_control_values=(0, 1),
    control_workspace=predicate, control_classical_bits=predicate_bits,
    prepare_workspace=work, prepare_classical_bits=stage_bits,
    traversal_workspace=work[:1], traversal_classical_bits=stage_bits,
    workspaces=((), ()), classical_bits=((), ()),
)
circuit.append(op)
```

Run the self-contained version:

```sh
python -m examples.algorithms.linear_combination_controls --repeated
```

The example keeps all eight control words coherent, verifies the projected
block and restores the entire complex state using the root adjoint. The model
has 28 T equivalents, five measurements and 13 peak logical qubits. The emitted
stream has 34 T/T-dagger instructions. The six-T difference is still the two
original abstract-versus-exact Fredkin costs, not a control-predicate mismatch.

Source returned-root acceptance and source returned-builder acceptance are
separate. Sixteen probes execute the unchanged later `get_ctrl_system` adder:
it raises TypeError at `np.concatenate(ctrl_soqs, in_soqs[...])`, even though
its returned root decomposes and has valid costs. Native root binding does not
repair that source callback or certify its equivalence. Grouped multi-bit
extension of an already conjoined root routes through that callback while
building its composite; two probes reproduce the TypeError and failed
QubitCount. Native execution and the combined model report reject that specific
request rather than replace it with a different root. Original reference code
and tests remain unchanged.

Validation covers eight existing activation vectors of lengths two through four,
both new polarities, two directions, eight-bit default coefficient oracles and
supplied oracles: 64 exact primitive-count/dependency configurations and 128
QEC-category/width comparisons. Profile-specific child inputs retain the BE10
legacy-alias distinction. Twenty-six full original matrix tests enumerate all
small branch histories. Four additional cases use non-Hermitian S/T children,
verify distinct adjoint matrices, literal source phase gates and absolute complex
branch phases without phase alignment. Invalid ownership, activation and mode
contracts and circuit-free model queries complete 250 new tests. The concentrated
regression passes 967 tests without skips; strict types, lint, formatting, scope
and architecture boundaries pass. No whole-file
or G2-completion claim follows from these subcapabilities.
