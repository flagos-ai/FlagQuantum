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

This accepts external structured quantum controls on an uncontrolled parent.
Stacking new controls on a precontrolled parent and classical controls remain
separate source paths. Original child cost failures and zero-total-ancilla
boundaries remain explicit; this is not whole-file or G2 completion.
