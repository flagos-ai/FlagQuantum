# Compiler

Transform quantum programs while preserving their numerical meaning,
measurement behavior, and required gradients. The design direction is one
verified path from quantum-classical source to target-legal output.

Compiler owns program transformations over Core semantics. Runtime chooses
execution; Simulation supplies numerical kernels; Compute and Remote adapt
resources. Compiler does not run the program.

## Start here

From the repository root:

```bash
python -m examples.compiler_optimize
python -m examples.target_aware_compilation
python -m examples.compiler_synthesis
```

The first example checks optimization against the original circuit, prints the
gate count at every implemented optimization level, and shows a reserved level
failing closed. The second checks routing legality and numerical equivalence on a
concrete topology. The third spells gates a target cannot run in the gates that
target publishes, and checks each rewrite against the original on the shipped
statevector engine; run it before changing an Euler form, an entangler cost, or a
ladder. Use
`optimize(program)` for target-independent optimization and
`compile(program, coupling_map=...)` for target-aware compilation. Both take
`optimization_level=`, which selects how much of the pass library runs and
defaults to `2`, the level that runs all of it. Use
`synthesize_one_qubit`, `synthesize_two_qubit`, and
`synthesize_state_preparation` for basis rewriting; they are not stable
`fq.compiler` exports, so reach them by module path.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Canonical optimization | [pipeline.py](pipeline.py) |
| Declared optimization levels and the pass order each one runs | [optimization_levels.py](optimization_levels.py) |
| The pass registry, the pass binding, and the fixed point | [pass_manager.py](pass_manager.py) |
| Cancellation of a declared inverse pair | [inverse_cancellation.py](inverse_cancellation.py) |
| Resets on a wire still in the zero state | [zero_state_reset.py](zero_state_reset.py) |
| Commutation rules and the block partition | [commutation.py](commutation.py) |
| Cancellation across a proven commuting gap | [commutation_cancellation.py](commutation_cancellation.py) |
| Diagonal gates before a measurement | [diagonal_before_measure.py](diagonal_before_measure.py) |
| Connectivity and routing | [coupling.py](coupling.py), [directed_topology.py](directed_topology.py), [routing.py](routing.py), [sabre.py](sabre.py), [topology_legalization.py](topology_legalization.py) |
| Qubit layouts and the layout restore | [layout.py](layout.py) |
| Initial placement on a device | [layout_planning.py](layout_planning.py) |
| Native-gate and target requirements | [native_gate_legalization.py](native_gate_legalization.py), [target_legalization.py](target_legalization.py) |
| Named-gate identities and the basis search | [basis_translation.py](basis_translation.py) |
| Conversion into a named basis | [basis_conversion.py](basis_conversion.py) |
| Exact quarter-turn rotations | [angle_synthesis.py](angle_synthesis.py) |
| One-qubit Euler angles | [one_qubit_synthesis.py](one_qubit_synthesis.py) |
| One-qubit run folding | [one_qubit_optimization.py](one_qubit_optimization.py) |
| Two-qubit block folding | [two_qubit_optimization.py](two_qubit_optimization.py) |
| Two-qubit block splitting | [two_qubit_optimization.py](two_qubit_optimization.py) |
| Two-qubit KAK angles and entangler cost | [two_qubit_synthesis.py](two_qubit_synthesis.py) |
| State-preparation ladders from amplitudes | [state_preparation_synthesis.py](state_preparation_synthesis.py) |
| The runnable path through all three synthesis entry points | `python -m examples.compiler_synthesis` |
| Dependency scheduling | [schedule_legalization.py](schedule_legalization.py) |
| Emission and round-trip checks | [target_emission.py](target_emission.py), [target_conformance.py](target_conformance.py) |
| OpenQASM interchange | [openqasm.py](openqasm.py), [openqasm_gates.py](openqasm_gates.py), [openqasm_import.py](openqasm_import.py) |
| Structured hybrid programs | [_hybrid/](_hybrid/README.md) |
| Static resource estimation | [resource_estimation.py](resource_estimation.py) |
| QIR base-profile emission | [qir.py](qir.py) |

## Replace a fact the passes read

The optimizer's passes read two opcode tables -- the self-inverse opcodes and
the opcodes whose parameter is a rotation angle -- and one derived partition of
the program into commuting blocks. Those are registered analyses, not module
globals: each pass declares what it `uses` and what it `preserves`, and the
manager hands it exactly those facts.

Nothing in this section is a stable public API. `PassRegistry`, `PassManager`
and `PassSpec` live inside this package so that the optimizer keeps one
composition authority, and an out-of-tree pass reaches the optimizer through
`flagquantum.ecosystem.extensions.pass_admission` rather than by editing the
default pipeline. The names below are for working on Compiler itself.

Two properties are enforced before the first pass runs. A name that no analysis
is registered under is refused where the pass is registered, so a misspelled
declaration cannot leave a half-applied pipeline. And an analysis that says it
is independent of the program is computed once for a whole fixed-point run,
which is only sound if every pass that reads it also declares that it preserves
it -- so a reader that omits the claim is refused rather than silently handed a
stale table.

A fact that does depend on the program, such as the commuting-block partition,
is recomputed from the program the round actually produced, and only when a
reader that declared it reads it. Replacing one `AnalysisSpec` in the registry
changes what a pass removes without touching the pass.

## Reach a Clifford+T basis

A target publishes a gate set and no more, and the named-gate identities in
`basis_translation.py` reach every opcode they name. A rotation is the one
family they cannot reach: `rz`, `phase`, and `u1` are the same gate up to a
global phase, no identity rewrites one of them, and the Euler synthesis needs a
z-rotation opcode to rewrite into -- so a target publishing only `h`, `s`, `t`,
and `cx` used to refuse a rotation outright.

`angle_synthesis.py` closes the half that has an exact answer. `rz(n * pi/4)` is
a single-qubit Clifford+T operator for every integer `n`, so it collapses onto
eight residues, and each residue carries a short word over `t`, `tdg`, `s`,
`sdg`, and `z`. `basis_translation.translate` consults those words as one more
candidate rewrite, ranked by the same rule it applies to every rule: a fully
native rewrite beats a shorter one that names a gate the target lacks.

```python
import math

from flagquantum.compiler import angle_synthesis

angle_synthesis.quarter_turns(3 * math.pi / 4)  # 3
angle_synthesis.QUARTER_TURN_WORDS[3]  # (('s', 't'), ('tdg', 'z'))
```

Two properties are worth knowing before you read the table. Each residue's first
word carries the minimum number of `t` gates, and the tests re-derive that by
enumerating the vocabulary rather than trusting the table. Each residue also
carries a fallback word, so a target publishing `t` but not `tdg` still reaches
`rz(-pi/4)`; without it `{t, s}` is not a group and would miss a residue.

What the module refuses is as load-bearing as what it answers. The classification
is an exact equality against a multiple of `pi/4`, so `pi/8`, `pi/3`, and `0.3`
fail closed rather than being answered with a rotation nobody asked for, and a
trainable angle is refused by the same check so a rotation never leaves the
autograd graph by being replaced with a constant word. Approximating an
arbitrary angle to a target accuracy is the Ross-Selinger problem, needs exact
arithmetic in `Z[omega, 1/sqrt(2)]` and integer factorization, and is not
implemented here; the parity contract records that gap rather than papering over
it.

```bash
pytest tests/unit/test_compilation_angle_synthesis.py
```

## Convert a program into a named basis

Native-gate legalization answers a question about a device. The loop underneath
it was never about one: choose a rewrite, check every leaf, bound the added
operations, record the replacement. `basis_conversion.py` is that loop, and
`native_gate_legalization.py` keeps the part that really is about a device --
reading `gates.native` out of a capability snapshot, matching it through the
capability matcher, and the wording its own refusals use.

```python
import flagquantum as fq

from flagquantum.compiler.basis_conversion import convert_basis

result = convert_basis(fq.Circuit(2).h(0).cx(0, 1), gates=("h", "cz"))
print([item.name for item in result.program.instructions])  # ['h', 'h', 'cz', 'h']
print([item.source_opcode for item in result.decompositions])  # ['cx']
print(result.source_basis, result.target_basis)  # ('cx', 'h') ('cz', 'h')
```

The answer is a report, not a bare program: the result, both bases as distinct
opcode names, one `GateDecompositionRecord` per rewritten instruction naming the
source position and the replacement opcodes, both content hashes, and a
`conversion_identity` over the whole claim. When nothing needed rewriting,
`changed` is False and `program is result.source_program`, so a caller can tell
"already in the basis" from "converted to it".

A named basis is validated before it is used, and every offending entry is
reported at once rather than the first. A bare string is refused instead of
being iterated character by character, aliases are canonicalised, and a channel
or a non-gate is refused by name -- there is no vendor basis table here and no
default basis, because a conversion that guessed would be a conversion nobody
asked for.

What it will not do is approximate. A gate outside the basis with no exact
decomposition is refused by name, and so is a decomposition whose own leaves
escape the basis, so a converted program is always exactly the program it was
given, up to the one global phase the IR cannot record. The four refusal kinds
are listed in `REFUSAL_KINDS` and each one names the instruction it is about.

The identity table is extensible without a second registry. A caller holding a
verified identity the table does not carry appends it with
`basis_translation.with_equivalence_rule`, which validates the rule and returns a
new frozen mapping, and passes the result to `convert_basis`:

```python
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.basis_translation import (
    EQUIVALENCE_RULES,
    with_equivalence_rule,
)
from flagquantum.core.ir import Instruction


def tdg_to_x_t_x(instruction):
    # Tdg is X T X up to one global phase, the phase FlagQuantum IR cannot hold.
    return tuple(
        Instruction(name, instruction.wires, metadata=dict(instruction.metadata))
        for name in ("x", "t", "x")
    )


rules = with_equivalence_rule(EQUIVALENCE_RULES, "tdg", tdg_to_x_t_x)
converted = convert_basis(
    fq.Circuit(1).gate("tdg", 0), gates=("h", "s", "t", "cx"), rules=rules
)
print([item.name for item in converted.program.instructions])
# ['h', 's', 's', 'h', 't', 'h', 's', 's', 'h']
```

Registration is not permission to leave the basis: the search expands a
registered rule's leaves through the table again, and every leaf is re-checked,
so a rule that names a gate the target lacks is refused exactly like a built-in
one. A new rule is appended rather than put first, so a rewrite the table
already had keeps winning a tie; displacing the built-ins takes an explicit
`replace=True`.

```bash
pytest tests/unit/test_compilation_basis_conversion.py
pytest tests/unit/test_compilation_basis_translation.py
```

## Cost a static program

`estimate_resources(program)` reports what a program contains and how its
dependencies pack, without running it: the operation count per opcode, the
schedule depth, the per-wire depth, the widest operation, the T family (`t` and
`tdg`) and its depth, and the channel count. The depth is the compiler's own list
schedule over the declared dependencies, so it is a lower bound for a real device
rather than a duration: no gate timing, connectivity, or routing overhead enters
it. A program whose instruction sequence has data dependence is refused with a
`CapabilityError` naming the instruction index and the reason, because a
straight-line list has no loop bound to estimate. Fault-tolerant costing is
layered above this unit rather than folded into it;
`flagquantum.algorithms.logical_resources` charges the tally and the depth on a
rotated surface code.

The record says which instruction sequence its figures are figures *of*. As
written, the basis is `ESTIMATE_BASIS`, `static_instruction_sequence`. Pass
`gates=` and the program is converted into that named basis first, and the basis
is reported as `named_basis:` followed by the basis's own canonical names -- so
a quarter-turn rotation counts as one `rz` in the first case and as one `t` in
the second, and neither count is a claim about a run. The conversion's own
identity travels on the record as `conversion_identity`, so a count over a
rewritten sequence is traceable to the rewrite that produced it.

Converting runs no more than counting does, and a basis that cannot express the
program exactly is refused rather than approximated: `estimate_resources(c, gates=("h", "t", "cx"))`
on a program carrying a `y`, or a channel, raises the same
`BasisConversionError` [`convert_basis`](#convert-a-program-into-a-named-basis)
raises. Naming a basis never turns a refusal into a number.

```python
import math
import flagquantum as fq
from flagquantum.compiler import estimate_resources

as_written = estimate_resources(fq.Circuit(1).rz(0, math.pi / 4))
print(as_written.basis, as_written.operation_counts)
# static_instruction_sequence {'rz': 1}

in_basis = estimate_resources(
    fq.Circuit(1).rz(0, math.pi / 4), gates=("h", "t", "cx")
)
print(in_basis.basis, in_basis.operation_counts, in_basis.t_count)
# named_basis:cx,h,t {'t': 1} 1
```

```bash
pytest tests/unit/test_resource_estimation.py
```

## Emit QIR

`qir.py` writes one program as QIR base-profile LLVM IR text: the
`FlagQuantumEntryPoint` definition with its `entry_point` attribute, the
`__quantum__qis__*` calls for the lowered program, a `__quantum__qis__mz__body`
per measured wire, the `__quantum__rt__tuple_record_output` and
`__quantum__rt__result_record_output` calls that record the output, the
declarations those calls need, and the module flags that declare the QIR version
and the two dynamic-management settings. `translate(program, format="qir-2.0")`
reaches the same text, so a consumer can be pointed at either entry point. QIR is
an open specification, so nothing here depends on a vendor component.

What it does not do is the half its name does not cover. Both dynamic-management
flags are `false`, so no `__quantum__rt__qubit_allocate_array`,
`qubit_release_array`, or `read_result` call is emitted and a program cannot
allocate a qubit at runtime through this path. Profile-QIR, the adaptive
profile, pulse-level generation, and runtime library extension calls are absent.
Only the constant-index pointer kind is emitted, and this is text rather than an
object file: no LLVM compilation runs, so conformance is the repository's own
semantic check plus a host C parser accepting the text as LLVM IR syntax. An
opcode with no QIS instruction is lowered to a native sequence equal to the
requested gate up to an unconditional global phase, which recorded measurement
output cannot observe; a gate the base profile cannot carry is refused by index
and opcode rather than approximated.

```bash
pytest tests/hybrid_compiler/test_qir_emission.py
```

## Save and load OpenQASM

`emit_openqasm` writes one program as OpenQASM 2 or 3 text, and the stable
`fq.from_openqasm` reads that text back. It accepts only the subset the emitter
writes and refuses everything else with an `OpenQASMImportError` that names the
reason in `issue_code`:

```python
import flagquantum as fq
from flagquantum.compiler.openqasm import emit_openqasm

text = emit_openqasm(fq.Circuit(2).h(0).cx(0, 1))
program = fq.from_openqasm(text)

program.instructions            # the imported gates, in source order
program.measurement_qubits      # the qubit behind each classical bit
fq.run(program.to_circuit(), outputs=fq.counts(), shots=256).counts[0]
```

Import returns the program rather than a `Circuit` because a `Circuit` cannot
carry a terminal measurement; call `to_circuit()` for the sampling path and
`to_ir()` when the measurement mapping must be preserved. It never executes
anything, and it does not read arbitrary third-party OpenQASM.

`openqasm_gates.py` holds the gate spellings both directions read. `openqasm.py`
emits, `openqasm_import.py` parses, and `target_conformance.py` checks an
emission against an expected program; the three share one table, so a spelling
one direction accepts and the other writes cannot appear. `tools/check_openqasm_import_contract.py`
reads [openqasm-import-v1-candidate.json](../../contracts/openqasm-import-v1-candidate.json)
against the shipped importer: the refusal vocabulary, the version lanes, the
exposed signature, the root additions, and a reader for every declared rule.

A transformation is acceptable when it preserves the relevant state,
measurement, and gradient references, produces legal output, and has bounded
code growth. An optimization must not remove a trainable gate solely because
its present angle is zero.

A reduction is only as strong as the reordering it is allowed to perform, so
[commutation.py](commutation.py) answers whether two instructions commute and
partitions a program into commuting blocks, one ordered partition per qubit. It is
a rule source of proof rather than of record: there is no shipped rule table, and
no rule reads a parameter value, so a trainable angle and a batch of angles are
decided exactly like a compile-time constant. Every claimed pair is checked
against the runtime's gate matrices by `tests/unit/test_compilation_commutation.py`,
which is why the reach is measured rather than asserted -- over every placement of
every declared arity-one and arity-two unitary pair it answers 3602 of the 3826
pairs that really do commute and never answers a pair that does not.
[commutation_cancellation.py](commutation_cancellation.py) is the only consumer:
it removes a parameter-free self-inverse pair whose two occurrences share a proven
commuting block, which is the case `merge_self_inverse` cannot see because a gate
stands between them. On a chain of six controlled gates interleaved with rotations
on their controls it removes 495 of the 708 instructions the previous pipeline
left. Where the residue is declined it is declined on purpose and enumerated, not
unnoticed.
[benchmarks/compiler_commutation_cancellation.py](../../benchmarks/compiler_commutation_cancellation.py)
holds the rule-source sweep, the per-population delta, and a per-opcode comparison
with Qiskit's `CommutationAnalysis` plus `CommutativeCancellation`.

Canonical optimization removes identities, cancels self-inverse pairs, adds up
adjacent rotations that share one opcode, and folds a same-wire run that *mixes*
opcodes -- `x rz(0.4) x`, four `t` gates spelling one `z`, the `h` that routing
walked between two rotations -- into the shortest sequence that reproduces it.
[one_qubit_optimization.py](one_qubit_optimization.py) is that last pass, and it
is the Compiler-layer counterpart of Qiskit's `Optimize1qGates`. It is exact
rather than exact-up-to-phase: a bare `u3` cannot carry a determinant, so the
leftover is emitted on the same wire as `phase` or `rz` instead of being dropped,
which is what Qiskit does with the `global_phase` field its DAG has and
FlagQuantum IR does not. A run already spelled in one z-rotation/pulse alphabet
is left alone, because the target's own lowering would re-spell the fold into
more gates than the run had; that decline is measured, not assumed. No run
carrying a trainable angle is folded at all -- composition reads numbers, and
`merge_adjacent_rotations` keeps those angles in the autograd graph.
[benchmarks/compiler_one_qubit_optimization.py](../../benchmarks/compiler_one_qubit_optimization.py)
holds the per-length reduction, the statevector exactness, the decline, and the
count difference from Qiskit's pass.

Re-spelling a run **into a declared target basis** is the one part of that
counterpart that is not here, and it is blocked by arithmetic rather than by a
missing routine. The determinant of a word over a target's arity-1 gates is the
product of its factors' determinants, so a word over `{rz, sx, x}` -- the three
arity-1 gates of the `ibm-rz-sx-cx` snapshot -- carries a determinant whose
argument lies in `{0, pi/2, pi, 3*pi/2}` degrees, and a run containing `t`,
`tdg`, `phase`, `u1`, or `u3` carries one outside that set. No word over that
basis reproduces such a run at any length. Qiskit's `Optimize1qGatesDecomposition`
takes that population anyway and pays the difference into `dag.global_phase`,
which its DAG has and `CircuitIR` does not; measured over 4000 seeded mixed runs,
a word that agrees only up to a global phase exists for all 4000 and is three
gates shorter on mean, while one that agrees entry for entry exists for 230 to
425 of them. The gap is the missing field, not missing synthesis. That second
column is reported as a reference rather than a constant because it counts a
floating-point coincidence: which runs the five-gate template happens to hit
exactly depends on exact-equality branches reading `cmath`, so it moves with the
platform's C library. The bound the obstruction actually implies is reported
separately and does not move.
[benchmarks/compiler_one_qubit_decomposition.py](../../benchmarks/compiler_one_qubit_decomposition.py)
measures the determinant subgroup per basis, the two columns per declared pair,
and the phase `native_gate_legalization` already drops on 76 to 78 of 80 seeded
entangled programs; [FQ-IR-GLOBAL-PHASE-20261006.md](../../docs/api-changes/FQ-IR-GLOBAL-PHASE-20261006.md)
is the proposal for the field.

Two-qubit block folding is the same idea one arity up, with one difference that
matters. `collapse_one_qubit_runs` composes a maximal run over one *wire*; the fold
composes a maximal **block** over one *ordered wire pair*, whose members are the
two-qubit gates on that pair plus the contiguous single-qubit gates immediately to
their left that sit on one of the pair's wires. A member's wires are read as a
Kronecker product on the side the pair gives it, so the block's product is the 4x4
operator the program computes, and the product is then compared entry for entry
against each declared two-qubit opcode. That draw-in is what reaches a conjugation
by a single-qubit gate -- `h(1) cz(0, 1) h(1)` is exactly `cx(0, 1)` -- which
neither this pass's two-qubit-run predecessor nor `collapse_one_qubit_runs` can see.
The conjugated wire has to be the one whose role changes, because the operand order
`(0, 1)` and its reverse `(1, 0)` are different blocks: the same three gates with the `h`s on `0`
compose to `cx(1, 0)` and are declined rather than relabelled.
[two_qubit_optimization.py](two_qubit_optimization.py) is the pass, and it is the
Compiler-layer counterpart of Qiskit's `ConsolidateBlocks` run at the granularity
this IR can prove. It is exact rather than exact-up-to-phase, and it emits at most
one gate: a block whose product is no declared opcode is left where it was, and a
trainable or batched angle is never read.
A larger product is not a more likely one: drawing a single-qubit gate into a block
can turn a product that *was* one declared gate into one that is not, so the widening
is not monotone by construction. The pass therefore keeps the narrower rule it
replaced as a floor -- a block it cannot re-spell is offered to that rule one
all-two-qubit sub-run at a time -- and the floor is measured rather than asserted:
with the sub-run fallback removed, the widened rule is 53 instructions longer than
the rule it replaced over one benchmark population while rescuing folds on 40 of that
population's 120 circuits, and 2 instructions shorter with it.
[benchmarks/compiler_two_qubit_optimization.py](../../benchmarks/compiler_two_qubit_optimization.py)
holds the membership table, the reach and its decline, the boundary sweep with
`false_yes` required to be zero, and a Qiskit anchor that reports what
`OptimizeCliffords` and `CollectCliffords` actually do to a gate-level circuit.

The fold has an inverse, and it is the one form of "take a two-qubit unitary apart"
this IR can state exactly. `split_two_qubit_blocks` reads one two-wire instruction
that carries a matrix and whose opcode no operator schema declares, and emits the two
single-qubit instructions whose Kronecker product it is. It is the counterpart of
Qiskit's `Split2QUnitaries`, with one difference worth stating: that pass sets
`new_dag.global_phase` on the branch it acts on because its factors are normalized,
while the factors here are scaled by the positive real Frobenius norm of the
product's largest 2x2 block and so need no phase field -- which is the field
`CircuitIR` does not have. Measured over the benchmark's 96 seeded products: the
emitted pair reproduces the source statevector to 2.5e-16 and the source matrix to
2.5e-16 entry for entry, with the phase column at 5.6e-17, and nothing is divided
out. A declared opcode is never a candidate, because its name is what the schema
fixes and its declared arity is what two one-wire halves would contradict, and a
Haar-random SU(4) is declined by the same arithmetic -- 0 of 300 accepted.
The input class is narrow because the pass is exact rather than because it is
prudent, and the class is not drawn by a tolerance: a product is recognized by
multiplication and rejected by multiplication. `clifford-t` in
[benchmarks/compiler_two_qubit_synthesis.py](../../benchmarks/compiler_two_qubit_synthesis.py)
publishes neither a z-rotation nor a pulse, so it serves no arm of that measurement
and reports a zero comparable count rather than an agreement it cannot make.
The payoff is reach rather than length, and it is measured as such. A two-wire matrix
instruction is the one shape the single-qubit passes cannot read, so before this pass
it had exactly one route out of `native_gate_legalization`: `synthesize_two_qubit`,
which needs an entangler. A target that publishes a z-rotation and a pulse and no
entangler at all reached none of that population and now reaches all of it. Where an
entangler does exist both routes serve the input and the split is the shorter one on
16 cases of 96, equal on 79, and longer on exactly one -- the identity factor, where
the entangler route's ten leaves are not padded but genuinely paid. The `shipped` and
`hand_split` arms of that benchmark are deliberately not compared by length: the
hand-written route names `x` where the shipped one holds an opaque matrix, and a
basis that publishes `x` keeps the name, so a length comparison would report name
preservation as a property of the pass.
`tests/unit/test_compilation_split_two_qubit_blocks.py` holds the exactness on raw
amplitudes, the refusal directions, and the falsifiability control: the phase-blind
route over the same input has an overlap magnitude of 1 and a raw difference above
0.1, so nothing in that file compares overlaps.


A gate that is its own inverse is one thing; a gate whose inverse is a *different*
opcode is another, and `merge_self_inverse` only saw the first kind, so it left
`s(0) sdg(0)` standing as two instructions although the pair is the identity.
[inverse_cancellation.py](inverse_cancellation.py) closes that with
`merge_inverse_pairs`, and it holds no opcode table of its own: it reads the
`OperatorSchema.adjoint` declaration through `operator_schema.inverse_operator`,
the same declaration `Circuit.adjoint` already consumes. That keeps the inverse
relation to one source of truth, and
`tests/unit/test_compilation_inverse_cancellation.py` enforces it structurally --
the module contains no opcode spelled as a code string and no module-level table.

The declared rule is a rule of proof rather than of record: every declared unitary
was multiplied by its declared inverse on the runtime's own matrices over seeded
draws, and the worst residual is 2.22e-16 against a smallest wrong-partner
residual of 1.22 -- a control, because a residual of zero would mean nothing if
the same comparison could not produce a nonzero one.

The pass removes a pair only when the gap between the two members is empty on
their own wire, and that boundary is measured rather than asserted:
[benchmarks/compiler_inverse_cancellation.py](../../benchmarks/compiler_inverse_cancellation.py)
drives every pair through ten gaps and reports, per row, whether the gap's
operator commutes with the pair and whether the pair was in fact removable --
measured against whole-register operators, since several of these gates act
trivially on `|0...0>` and a state comparison would call a row redundant for a
reason unrelated to the shape. Of the 60 rows, 6 gaps are empty and 12
instructions are removed there, 32 gaps commute and 24 instructions are removed
across the sub-case where the gap misses the pair's wire entirely, 22 gaps do not
commute and are never crossed, and 20 rows are declined although the pair was
removable: reach left on the table, reported as a number rather than as a promise.
A refusal is therefore either correctness or deferred reach, never a single total.
The module also drives Qiskit's `InverseCancellation` over the identical circuits
and finds the two agreeing on every row.
One identity is a property of the IR rather than of any opcode table. A `reset`
is not in the operator schema at all, yet a reset on a wire that is still in
`|0>` computes its own outcome -- which can only be zero -- and leaves the wire
where it already was, so it is the identity.
[zero_state_reset.py](zero_state_reset.py) is the rule that sees that, and it is
one sentence long: a `reset` is removed when no instruction that survives before
it touches its wire. It reads no matrix and no parameter, so the module holds
exactly one gate name and imports neither the operator schema nor a matrix
source, and a test asserts both -- a second gate name or a matrix import would
mean the rule had become something else. What it rests on instead is that a
`CircuitIR` register starts in `|0...0>`, and that is checked as a property of
the type rather than assumed: the IR has eight fields, none of which can carry
another initial state, and that test fails the day one can.

The pass is called first in the fixed-point loop, because a reset it can remove
is removable whatever the other passes do and removing it hands them a shorter
program. The loop, not that ordering, is what earns the reach on a wire the other
passes only empty out later: `x(0) x(0) reset(0)` needs a second round, and the
benchmark counts those rows apart from this rule's own reach instead of
attributing them to it.

A refusal here is either correctness or deferred reach, never one total.
`tests/unit/test_compilation_zero_state_reset.py` and
[benchmarks/compiler_zero_state_reset.py](../../benchmarks/compiler_zero_state_reset.py)
split them. A reset behind anything on its own wire is left alone; where that
"anything" provably leaves the wire in `|0>` -- a `z`, an `s`, a zero-angle
rotation -- the refusal is deferred reach, and the benchmark measures it with a
64-trajectory average reduced state of the reset's own wire rather than inferring
it from the shape's name. That average is also what keeps a `measure` honest: a
reset behind a measure is not removable even though the trajectory that drew a
zero did leave the wire in `|0>`, because the other trajectories did not. Over 24
shapes and 30 resets the pass removes 12, the measurement agrees all 12 were
removable in fact, and 13 more were removable and were not removed. A removal
that was *not* removable in fact would be a correctness defect, and that count is
measured to be zero rather than argued to be zero.

The benchmark's second instrument follows from the rule rather than being a
detail. A reset reads a random draw, so removing one shifts the generator stream
and a fixed seed no longer reproduces the same per-shot trajectory even though no
distribution moved. A measure-free program is therefore compared on the exact
final state and a measuring program on outcome shares, and the payload records
which instrument spoke for which population. The share comparison carries a
control that deletes a reset whose wire has left `|0>` and moves a share by 0.51
against a 0.03 tolerance, so the tolerance is a bound something exceeded rather
than a number that happened to hold.

A measurement ends a wire's contribution to the recorded outcome: whatever a later
gate does on that wire cannot reach the classical bits the measurement already
wrote. A gate that commutes with the measurement -- one whose operator is diagonal
in the computational basis -- therefore leaves the outcome distribution exactly as
it was, and [diagonal_before_measure.py](diagonal_before_measure.py) removes it.
The rule is a declaration rather than a matrix read, and for a structural reason:
this layer may not import `simulation/**`, and no `OperatorSchema` field records
diagonality, so the pass asks the operator instead. A one-wire gate is diagonal
when the polar angle of its canonical Euler triple -- the table
[one_qubit_synthesis.py](one_qubit_synthesis.py) already owns -- is exactly zero,
and a two-wire gate when its opcode is one of four. Both rules are state
independent: an `x` that happens to be invisible behind a Hadamard preparation is
still not diagonal and is still kept.

Two consequences are deliberate. A trainable `rz`, `phase` or `u1` is reported
diagonal while a trainable `rx`, `ry` or `u3` is not, because the first three have
a tabulated polar angle of exactly zero and the last three do not; a zero-valued
constant is what the rule sees, and a trainer's current value is not. And the
declaration is per opcode rather than per value on two wires, because nothing in
the repository declares two-wire diagonality -- the two hand-written opcode sets
that do exist live in `simulation/` and `runtime/` and disagree with the matrix
census in both directions.

Qiskit's counterpart is `RemoveDiagonalGatesBeforeMeasure`, which states the same
idea as class membership: `RZGate`, `ZGate`, `TGate`, `SGate`, `TdgGate`,
`SdgGate`, `U1Gate` and `CZGate`, `CRZGate`, `CU1Gate`, `RZZGate`. The declaration
here names the same four two-wire opcodes and reaches three rows a class list
cannot express -- an `i` (there is no `IGate` in that list), a `phase`/`p`
(`PhaseGate` is not a `U1Gate` subclass) and a `cphase` (`CPhaseGate` is not a
`CU1Gate` subclass) -- and a further reach that no list of classes can express at
all: `u3(0, phi, lam)`, `rx(0)` and `ry(0)`, which are diagonal as values rather
than as names.

Like the reset rule, a refusal here is correctness or deferred reach and never one
total. A candidate carrying a caller-supplied matrix, a condition, the dynamic
flag, an unknown opcode, or a successor on its own wire that is not an
unconditional non-dynamic measurement is declined, and the decline is conservative:
a false "no" costs an optimization, a false "yes" changes the program. Over 196
shape rows the pass removes 26, none of which was not removable in fact, and
declines 170 -- of which 81 were invisible in fact, 42 behind a condition and 39
because a later instruction stood on the candidate's wire. Those two are counted
apart because they are different limits. The two-wire reach is the largest single
gap: when only one of the candidate's wires is measured, all eleven two-wire
candidates are invisible in fact and all eleven are declined, because the rule
requires a measurement after every wire the candidate touches.

`tests/unit/test_compilation_diagonal_before_measure.py` and the benchmark
[benchmarks/compiler_diagonal_before_measure.py](../../benchmarks/compiler_diagonal_before_measure.py)
hold that evidence. The benchmark measures every removal against an exact outcome
distribution rather than a sampled one, and carries a control that deletes a
non-diagonal gate and an entangler and moves the distribution by 0.49 against a
1e-12 tolerance, so the tolerance is a bound something exceeded rather than a
number that happened to hold. It also runs Qiskit's own pass over the identical 196
programs: the two agree everywhere the class lists allow a comparison, Qiskit
removes 20 where this pass removes 26, and the six rows they differ on are exactly
the three class-membership facts above. The payload records that scope rather than
claiming agreement everywhere.

A *named* gate has no matrix on this layer, so it can only leave a program for
a basis that does not carry it through a closed identity.
[basis_translation.py](basis_translation.py) is that table, plus the
deterministic search that composes it, and it is the Compiler-layer counterpart
of the equivalence library and basis search Qiskit ships. Each rule is an exact
closed form that forwards the source instruction's own parameter objects and
metadata, so a trainable angle stays in the autograd graph; no rule introduces an
angle the caller did not write; and the search branches on opcode names only,
never on a value. Seventeen of the eighteen entries reproduce their source
exactly and `cphase` is equal to it up to one global phase, which FlagQuantum IR
has no field to record. The table stores the shortest statement of each identity
rather than its closure, so the search expands each rule's leaves through the
table again: a `cz` basis plus a z-rotation reaches all eleven declared two-qubit
opcodes by name, where the four hand-written rules this replaced reached two of
them. The two three-qubit entries cover the whole of this IR's multi-controlled
surface -- `ccx` as its fifteen-gate standard form and `cswap` as a `ccx` around
two `cx` -- and they reach the three bases that publish a `cx` or a `cz` sink.
Qiskit splits the same operation into a Gray-code, a recursive and a V-chain
construction, but at two controls the first of those is `CCXGate` itself and the
other two exist to trade ancillas for fewer entanglers at five controls and up,
which this IR cannot express. The Gray-code *statement* is still measured against
the taken one, and declined: seven declared leaves against fifteen, but eight
two-qubit gates against six.
`ion-trap-rz-rx-rzz` is the recorded gap -- every entangling rule routes down to
`cx` or `cz`, and that basis publishes neither.
[benchmarks/compiler_basis_translation.py](../../benchmarks/compiler_basis_translation.py)
holds the per-rule fidelity, the per-basis reach, and the difference from Qiskit's
`BasisTranslator` on the same target bases.

Native-gate legalization rewrites one-qubit instructions by Euler synthesis
whenever the target publishes a z-rotation and a pi/2 x-rotation: every declared
single-qubit unitary becomes z-rotations plus `sx`, or plus `rx` at a fixed
`pi/2` when the basis publishes `rx` instead. An exact identity is tried first, so
a basis that can carry a gate exactly keeps its exact form. Such a basis cannot
carry the global phase of `h`, `s`, or `t`, and FlagQuantum IR has no field for
it, so those rewrites are equal up to one global phase -- the same contract every
`U3`-based hardware basis publishes. A consumer that compares raw statevectors
instead of measurement statistics has to know that; flag records of it belong in
the capability registry, which `capability-maturity.toml` owns.

Two-qubit KAK synthesis extends that to a matrix-carrying instruction on two
qubits, over any supercontrolled entangler the basis publishes. Six declared
arity-2 opcodes reach a supercontrolled Weyl point: `cx`, `cz` and `cy` as they
stand, and `rzz`, `ryy` and `rxx` at an angle of `pi/2`. `cphase` is excluded on
purpose -- its only supercontrolled angle is `pi`, where it is `cz`, so it would
add a spelling rather than reach. Carrying the interaction rotations is what
opens a trapped-ion or flux-tunable-coupler basis, whose only two-qubit gate is a
rotation and which publishes no `cx` at all; such a basis now reaches all eleven
declared two-qubit unitaries where it previously reached one. The cost is one
entangler for `cx`, `cz`, and `cy`, two for the controllized rotations, and three
for `swap`, which is the far corner of the Weyl chamber; all eleven declared
two-qubit unitaries reproduce their source to within `1.1e-15` over any of the six
entanglers, up to one global phase.
[benchmarks/compiler_two_qubit_synthesis.py](../../benchmarks/compiler_two_qubit_synthesis.py)
holds the reach, cost, and phase measurement, and cross-checks the entangler
count against Qiskit's `TwoQubitBasisDecomposer` over the whole table.

An operator no target vocabulary names reaches this stage as an instruction
carrying a matrix, and four implementations answer it: the KAK route above, the
product-first `_local_replacement`, the `_matrix_replacement` that chooses
between them, and `split_two_qubit_blocks`, which re-spells such a matrix as its
two single-qubit factors. They are one boundary with several implementations, so
[test_two_qubit_decomposition_conformance.py](../../tests/team/compiler/test_two_qubit_decomposition_conformance.py)
states their shared postconditions once and drives all four through every one of
them. The routes do not share an observable convention, and the suite asserts two
rules rather than one: the synthesis routes preserve the source unitary up to one
global phase -- the instrument is the overlap, which cannot see one, and every
route's worst deviation over the 41-case family is below `1.8e-14` -- while the
fold route is exact entrywise, at `1.2e-16`, because `_product_factors` reads the
Kronecker scale into the emitted factors and records no phase at all. The same
input also shows why the phase assertion is an arc rather than an equality: the
phase of the KAK answer genuinely depends on the entangler, spreading over an arc
of up to `5.71` radians across the six of them, and 40 of the 41 cases move by at
least a quarter turn. The floor is a quarter turn and not a half turn because every
such difference is a whole number of quarter turns while which multiple a case
lands on is not portable: one point of the family spreads by two quarter turns on
arm64 and by one on x86-64, since the phase it is read from sits on a grid
boundary and the two `libm`s round it either way. Reach is asserted as a count per
route per entangler arm rather than as a capability, because a product of two
single-qubit unitaries is answered by the product route on a target that publishes
no entangler, while the KAK route refuses every one of them.

The suite also records one diagnosed refusal rather than a passing test. A
two-qubit unitary whose Weyl coordinates have `b == c` makes `m2` in
`_weyl_decomposition` exactly degenerate, and `_diagonalize_m2`'s acceptance
tolerance of `1e-13` sits below that class's intrinsic reconstruction residual of
about `2.3e-9`, so the diagonalizer fails on all 100 attempts and raises the
two-qubit Weyl decomposition error rather than a decomposition. Which matrices hit
it depends on the local frames a degenerate Weyl point is dressed in, not on the
point alone, and `legalize_native_gates` propagates that bare `ValueError` where
every other failure it can reach a caller with is wrapped in
`NativeGateLegalizationError`. The suite therefore pins a family measured to be
refusal-free under a ten-frame sweep per point, and
`test_the_near_product_band_is_refused_by_both_routes` documents the escape as a
limitation rather than endorsing it.

State-preparation synthesis turns an amplitude vector into a circuit, which is the
one place in this package where the input is a classical vector rather than a
program. `synthesize_state_preparation(amplitudes, qubits=None, z_rotation="rz",
pulse_opcode="sx", entangler="cx", metadata=None)` returns the leaves of a
uniformly controlled ladder -- a magnitude pass over the register, then a phase
pass over it, both in the Möttönen-Vartiainen-Bergholm-Salomaa construction -- or
`None` when the named basis cannot carry one, which is the same refusal
`synthesize_two_qubit` gives. Level `j` acts on wire `j` controlled by wires `0`
to `j-1`, each level's control flips are spelled by `synthesize_two_qubit` out of
the caller's entangler, and the result is a plain tuple of `Instruction` with no
runtime selected, no device named, and no state executed. It normalises the vector
rather than requiring one, since a global phase and a scale are both unobservable.

Two constraints in that signature are load-bearing and neither is a preference.
`z_rotation` may only be `rz`: `phase` and `u1` are exactly `exp(1j * theta / 2)`
times `RZ(theta)`, so over a ladder, where the angle is the branch index, the
offset becomes branch dependent and the level stops being one controlled rotation.
And the `ry` groups are emitted in the general Euler form
`RZ(lam) PULSE RZ(theta - pi) PULSE RZ(phi - pi)` rather than through
`synthesize_one_qubit_matrix`, whose shorter spellings move the dropped global
phase by a full `pi` when the polar angle reaches `pi`. A ladder multiplies its
branches' dropped phases into relative phases, so it needs the one form that holds
a constant. Both refusals are therefore fail-closed: a basis this construction
cannot hold is reported as `None` rather than answered with a replacement that was
never verified.

An amplitude vector also has an existing implementation in this repository:
`flagquantum.algorithms.primitives.state_preparation`, which `algorithms.svd` and
`algorithms.pca` call and which the change that added the compiler-side entry point
did not edit. That makes the pair the replacement ARCH-012 clause 2 asks for, and
[test_state_preparation_synthesis.py](../../tests/team/compiler/test_state_preparation_synthesis.py)
is where the two meet, because `tests/**` is the only path both domains share. The
suite states the boundary's postconditions once and drives both implementations
through all of them behind the shipped simulation entry point, and it asserts three
properties as differences so that no check is a claim never seen to fail: the short
Euler forms' phase really does move by `pi` at `theta = pi` where the general form
holds one constant; a `phase` ladder really is `O(1)` away from its `rz`
counterpart, by a different phase on each branch; and a single zero branch of a
magnitude ladder may drop its `RY(0)` group but not its flips, while a whole zero
ladder may drop both. On the uniform superposition the leaf count is exactly
`4 * n + flip_cost * (2**n - 2)` at every width, and the flip cost is the roster's
own measurement of one `cx` rather than a number this package keeps: one leaf for
`cx`, `6` for `cy`, `7` for `rzz` and `9` for `cz`, all four of them arithmetic
because their routes cross only polar angles `one_qubit_synthesis._zyz_angles`
produces exactly. The two entanglers that are not degenerate are a host quantity
instead, because their trailing local factor's polar angle is a general
`2 * atan2(...)` that `_leaves` compares to `pi/2` with `==`: `rxx` costs `10` or
`13` leaves and `ryy` `11` or `13`, depending on which way the platform's own
rounding falls. The suite states that one dimension as the set it is.
Measured, the two implementations agree to `2.6e-08` through `5.7e-08` entrywise,
which is the frozen reference's `complex64` floor rather than this boundary's
rounding: the compiler side reaches `3.9e-16` against the exact direction. The
entry point is not re-exported from [__init__.py](__init__.py); it is reached by
module path, and no capability label is claimed for it here.

The ten passes are one boundary with ten implementations of the same task --
delete work that cannot change an observable -- so a new or rewritten pass is
accepted only when it satisfies the same postconditions as the rest.
[test_optimization_pass_conformance.py](../../tests/team/compiler/test_optimization_pass_conformance.py)
states that contract once and drives every pass through all of it: the program
scaffolding survives, no pass grows a program or emits on a wire the source never
used, no pass mutates its input or answers differently twice, each wire keeps the
order it carried, and the observable the pass may move is preserved. Operations on
*disjoint* wires are allowed to move relative to each other, because they commute
and a pass that folds one wire's stack at a time does reorder them; the order that
may not change is the one two surviving operations share a wire in.

The observable is not one assertion. Eight passes act on unitary opcodes, and for
them the state vector is compared as a vector rather than through the overlap
`1 - |<in|out>|`, because an overlap is blind to the one defect this repository has
already paid for: a fold that drops a global phase. The other two reason about
`reset` or `measure`, which `operator_schema` does not declare, so the statevector
engine cannot execute those programs at all and their observable is the sampled
outcome distribution. That split is derived from the operator schema in the suite
rather than restated there. Each instrument is calibrated by a control that has to
fail it: a pass that appends an identity, a pass that advances one surviving
rotation by a full turn -- a phase of pi, which the overlap cannot see and the
vector difference measures at 0.638 or more -- and the two removals the dynamic
passes refuse, taken by hand, which each move a share by 1.000 against a legitimate
removal's worst 0.0197.

The roster the suite checks is the loop's own composition, not a copy of it:
`optimize` runs its passes to a fixed point and imports them inside the function,
so the suite records the calls the loop actually makes and compares the recording
with its enumeration. A pass added to the loop and not to this file fails there.
Since the composition is named rather than written inline, the enumeration is the
declaration in [optimization_levels.py](optimization_levels.py) and the recording
is still what proves the loop runs it.

## How much optimization runs

`optimize` and `compile` take `optimization_level`, an integer naming how much of
the pass library runs. The levels are Qiskit's, because the two libraries' pass
sets correspond closely enough that a user moving between them should not have to
learn a second scale, and `optimization_levels.py` records the correspondence per
pass rather than only the numbers:

| Level | Stages |
| --- | --- |
| `0` | None. The program is returned unchanged. |
| `1` | Reset and identity removal, self-inverse and inverse-pair cancellation, adjacent-rotation merging, one-qubit run folding. |
| `2` | Everything `1` runs, plus diagonal-gate removal before measurement and rotation merging across a proven commuting gap. |
| `3` | Declared and reserved. `CompilationError`, because the unitary-synthesis stage Qiskit's level 3 moves into the optimization loop needs a target basis and `optimize` has none. |

`2` is the default and the level every earlier release effectively ran, so a
caller that names no level receives the program it used to receive; level `0`
returns the submitted program instruction for instruction, which is what
supporting "do not optimize" as a level rather than a flag has to mean. The level
that ran is recorded on the result under `metadata["optimization"]`, so a level-0
program that had nothing to optimize is still distinguishable from one that was
never offered to the optimizer at all.

Level 3 is refused rather than approximated. Qiskit's level 3 differs from its
level 2 by moving `UnitarySynthesis` into the fixed-point loop, and that pass can
never be a stage of this ladder: it rewrites each instruction it cannot keep into
a *target's* native gates, so it needs a target basis, while `optimize` is
target-independent by contract and is handed no target. The capability is in the
package and not in this entry point --
[`native_gate_legalization.legalize_native_gates`](native_gate_legalization.py)
performs that same rewrite against a `TargetCapabilitySnapshot` -- so "write the
missing pass" is not the work that would unblock level 3, and the refusal says so
rather than leaving a reader to look for a pass that already exists. Answering a
request for level 3 with a level-2 program would be a silent downgrade of exactly
the kind this repository forbids, and narrowing the ladder later is a breaking
change while widening it is not, so the refusal is the version that can be
corrected cheaply.
A value that is not an integer is refused for the same reason: `True` and `2.0`
both compare equal to an implemented level, so accepting them would let an
unvalidated value select a pass set.

The declaration is checked against execution rather than against itself.
[test_optimization_levels.py](../../tests/team/compiler/test_optimization_levels.py)
resolves every declared name through the loop's own roster, asserts level 2 is the
whole library and level 1 a subsequence of it, and asserts the reach of each level
over a seeded family: level 1 is strictly shorter than level 0 on all forty seeds
and level 2 strictly shorter than level 1 on twelve, because level 1 already folds
a single-qubit run into one Euler form and level 2's extra reach is what survives
that. Each level is also checked for preserving the state and for never growing
the program, so the levels form a ladder rather than four independent settings.

Routing strategies are one boundary with several implementations, so a new or
replaced strategy is accepted only when the shared conformance suite in
[test_routing_conformance.py](../../tests/team/compiler/test_routing_conformance.py)
passes over it. That suite also runs an independent router and a replacement
router through the compiler, the topology legalizer, and the dynamic-circuit
runtime, which is how the boundary is shown to be replaceable rather than
merely defined.

The reference lookahead SWAP search Qiskit ships was measured against these
strategies before being rejected as a fifth one. It passes the boundary above:
at search depth and width one it reproduces the shipped planner's plan, and every
program it compiles is legal on the device and numerically equal to its source.
It is rejected on cost. No objective of it beats `sabre_layout`, and the objective
Qiskit documents fails closed on programs where one operation stays stranded
behind the front layer, so it cannot be routed at all on part of the workload.
[benchmarks/compiler_lookahead_swap.py](../../benchmarks/compiler_lookahead_swap.py)
holds the measurement and the commands that reproduce it.

The randomized layer-permutation search Qiskit shipped as `StochasticSwap` was
measured the same way, and rejected on cost as well. Its plan is sound: the
placements it records are the replay of its own SWAPs, and every two-qubit
operation it places sits on a device edge. It still retains 1.617 times the SWAPs
`sabre_layout` retains and beats that strategy on none of the 140 measured
programs, and Qiskit's own compiled implementation of the same algorithm retains
0.5% fewer SWAPs on the same basis -- effectively the same count -- so the
shortfall is the algorithm rather than the port. [benchmarks/compiler_stochastic_swap.py](../../benchmarks/compiler_stochastic_swap.py)
holds the measurement and the commands that reproduce it. The checked-in port is
also the only runnable form of the algorithm left to this repository, because
Qiskit 2.0 removed the pass and this repository certifies Qiskit 2.x.

What those strategies cost is a checked-in number rather than an impression.
[benchmarks/compiler_routing_quality.py](../../benchmarks/compiler_routing_quality.py)
compiles the same 140 programs through `flagquantum.compiler.compile` -- the entry
point a caller uses, so the programs are optimized before they are routed, as a
caller's are -- and reports the SWAPs each of the five entry points retains after
post-routing optimization. `sabre_layout` retains 2018, `sabre` 2352, the
automatic selection 4310, `restore_after_each_gate` 4660 and
`persistent_layout` 4716, with every entry point routing all 140 programs, no
two-wire operation off the device, and every compiled program equal to its source
to `2.8e-16`. These are the counts at the default optimization level, `2`: a
caller who names a lower level runs fewer of the optimization passes that remove
inserted SWAPs afterwards, so the level is part of what a retained count means.
Two numbers there are worth carrying forward. The automatic
selection never resolves to a SABRE strategy, because its cost estimate can rank
only `restore_after_each_gate` and `persistent_layout`, and that scope costs
`2.14` times what the best available strategy costs -- so the price of the
restriction is measured rather than merely stated. And the estimate is good inside
that scope: it picks the cheaper of its two candidates on 132 of the 140 programs
and lands within `1.005` of always picking the better one. The eight misses are
not a bias in the estimate. It minimizes the SWAPs a plan *inserts*, as its
objective field states: its planned count equals the router's planned count on all
140 programs for both candidates, and its objective with its declared tie-break
reproduces the selection on all 140. The misses are the gap between two questions.
This benchmark scores the SWAPs *retained*, and the post-routing optimization
removes 5.25% of one candidate's planned SWAPs and 5.49% of the other's, so a
program where `persistent_layout` plans fewer and retains no fewer is scored as a
miss. Those counts are a reading of the shipped router, and
they are only reproducible because that router resolves a physical pair to one
route: `CouplingMap.shortest_path` expands from the lower-indexed endpoint rather
than from whichever endpoint the caller passed, so the two operand orders are
exact reverses of each other and a device that has already answered other queries
routes exactly as a cold one does. A pair on an even ring has two routes of equal
length, so the route is a tie-break, and a tie-break that reads the device's
history would make the count depend on which pass asked first.
`DirectedCouplingMap.shortest_path` breaks the same tie the same way, because SWAP
placement travels the same undirected graph whichever device describes it.
[test_compiler_routing_quality.py](../../tests/benchmark_contract/test_compiler_routing_quality.py)
holds those readings, including a test that fails if the automatic selection
becomes an alias of one candidate or quietly widens its scope.

Routing moves two-qubit operations onto device edges by inserting SWAPs. No
strategy here synthesizes an operation that touches three or more qubits, so such
an operation is carried through unchanged, and only when the device already
carries the couplings its operands interact over; it is refused otherwise.
Decomposing it is a caller or native-gate step.
[test_multi_wire_routing_locality.py](../../tests/team/compiler/test_multi_wire_routing_locality.py)
holds that boundary, including the case where a chosen layout would move a
legal multi-qubit operation onto non-adjacent physical qubits.

A placement is a tuple of physical qubits, one per logical qubit: the argument
`route_to_directed_topology` takes as `initial_layout`. [layout.py](layout.py)
`Layout` carries the same assignment as a value, and since a routed program on a
device wider than the program leaves slots idle, `Layout` takes a
`physical_slot_count` and reports `None` for an idle slot. Routing on a plain
`CouplingMap` may only use qubits the program owns and so refuses `initial_layout`
outright; a non-identity placement therefore requires a `DirectedCouplingMap`,
where the idle slots are a workspace that the inverse routing SWAPs clean.

[Implementation details](IMPLEMENTATION.md) document stage ownership and
migration constraints. [Testing policy](../../docs/development/TESTING.md)
defines the additional checks for each change.
