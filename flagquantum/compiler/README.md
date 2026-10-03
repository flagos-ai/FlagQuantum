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
```

The first example checks optimization against the original circuit. The second
checks routing legality and numerical equivalence on a concrete topology.
Use `optimize(program)` for target-independent optimization and
`compile(program, coupling_map=...)` for target-aware compilation.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Canonical optimization | [pipeline.py](pipeline.py) |
| Cancellation of a declared inverse pair | [inverse_cancellation.py](inverse_cancellation.py) |
| Resets on a wire still in the zero state | [zero_state_reset.py](zero_state_reset.py) |
| Commutation rules and the block partition | [commutation.py](commutation.py) |
| Cancellation across a proven commuting gap | [commutation_cancellation.py](commutation_cancellation.py) |
| Diagonal gates before a measurement | [diagonal_before_measure.py](diagonal_before_measure.py) |
| Connectivity and routing | [routing.py](routing.py), [sabre.py](sabre.py), [topology_legalization.py](topology_legalization.py) |
| Wire layouts and the layout restore | [layout.py](layout.py) |
| Initial placement on a device | [layout_planning.py](layout_planning.py) |
| Native-gate and target requirements | [native_gate_legalization.py](native_gate_legalization.py), [target_legalization.py](target_legalization.py) |
| Named-gate identities and the basis search | [basis_translation.py](basis_translation.py) |
| Conversion into a named basis | [basis_conversion.py](basis_conversion.py) |
| Exact quarter-turn rotations | [angle_synthesis.py](angle_synthesis.py) |
| One-qubit Euler angles | [one_qubit_synthesis.py](one_qubit_synthesis.py) |
| One-qubit run folding | [one_qubit_optimization.py](one_qubit_optimization.py) |
| Two-qubit KAK angles and entangler cost | [two_qubit_synthesis.py](two_qubit_synthesis.py) |
| Dependency scheduling | [schedule_legalization.py](schedule_legalization.py) |
| Emission and round-trip checks | [target_emission.py](target_emission.py), [target_conformance.py](target_conformance.py) |
| OpenQASM interchange | [openqasm.py](openqasm.py), [openqasm_gates.py](openqasm_gates.py), [openqasm_import.py](openqasm_import.py) |
| Structured hybrid programs | [_hybrid/](_hybrid/README.md) |
| Static resource estimation | [resource_estimation.py](resource_estimation.py) |
| QIR base-profile emission | [qir.py](qir.py) |

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
them. The two three-wire entries cover the whole of this IR's multi-controlled
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
wires, over any supercontrolled entangler the basis publishes. Six declared
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
placements it records are the replay of its own SWAPs, and every two-wire
operation it places sits on a device edge. It still retains 1.617 times the SWAPs
`sabre_layout` retains and beats that strategy on none of the 140 measured
programs, and Qiskit's own compiled implementation of the same algorithm retains
0.5% fewer SWAPs on the same basis -- effectively the same count -- so the
shortfall is the algorithm rather than the port. [benchmarks/compiler_stochastic_swap.py](../../benchmarks/compiler_stochastic_swap.py)
holds the measurement and the commands that reproduce it. The checked-in port is
also the only runnable form of the algorithm left to this repository, because
Qiskit 2.0 removed the pass and this repository certifies Qiskit 2.x.

Routing moves two-wire operations onto device edges by inserting SWAPs. No
strategy here synthesizes an operation that touches three or more wires, so such
an operation is carried through unchanged, and only when the device already
carries the couplings its operands interact over; it is refused otherwise.
Decomposing it is a caller or native-gate step.
[test_multi_wire_routing_locality.py](../../tests/team/compiler/test_multi_wire_routing_locality.py)
holds that boundary, including the case where a chosen layout would move a
legal multi-wire operation onto non-adjacent physical wires.

A placement is a tuple of physical wires, one per logical wire: the argument
`route_to_directed_topology` takes as `initial_layout`. [layout.py](layout.py)
`Layout` carries the same assignment as a value, and since a routed program on a
device wider than the program leaves slots idle, `Layout` takes a
`physical_slot_count` and reports `None` for an idle slot. Routing on a plain
`CouplingMap` may only use wires the program owns and so refuses `initial_layout`
outright; a non-identity placement therefore requires a `DirectedCouplingMap`,
where the idle slots are a workspace that the inverse routing SWAPs clean.

[Implementation details](IMPLEMENTATION.md) document stage ownership and
migration constraints. [Testing policy](../../docs/development/TESTING.md)
defines the additional checks for each change.
