# API Change Proposal 066: Typed classical control for the dynamic-circuit path

## Status

**Proposed. No code is written by this proposal and no behaviour changes until
the repository owner approves it and the acceptance items below are measured.**

One plan row names this document: `W5-10`, whose deliverable column is this file.
The recorded critical-path block places the node `W5-10/11` after `W5-04` and
before `W6-01/06`, but the per-row dependency column shows `W6-01` depending on
`W5-11` and `W6-06` depending on `W5-04`, so this document is not a recorded
blocker for either of them. `W0-05b` is the sibling row and its draft,
[`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md),
governs the promotion label this document is often assumed to grant.

Thirteen premises below were measured on this checkout before the proposal was
written, because engineering decision principle 6 requires inspecting the existing
repository before adding a contract and forbids creating a second source of truth,
and because clause 5 of the current control sequence requires an absent capability
to appear as an explicit, owned gap rather than as silence. They are marked
**Measured** where they appear, with the literal command and the literal output.
Everything in "Required evidence" is a requirement on the implementation, not a
result: nothing here has been built, and no tolerance is proposed here because
none has been measured.

This proposal does not raise a capability maturity level. `capability-maturity.toml`
records `capabilities.dynamic_circuits` at `level = "experimental"`, and a
proposal cannot move that label.

## Problem

### 1. The public dynamic surface is one name, and its stability lives in a docstring

**Measured.** `flagquantum/dynamic.py` is five lines:

```console
$ wc -l flagquantum/dynamic.py
       5 flagquantum/dynamic.py
```

Its module-level names are `DynamicCircuit` only, and its `__all__` is
`("DynamicCircuit",)`. The stability marker is a module docstring, quoted
verbatim from `flagquantum/dynamic.py:1`:

```python
"""Candidate-stable dynamic-circuit construction contract."""
```

There is no stability marker on the class itself. `DynamicCircuit` is defined at
`flagquantum/runtime/dynamic/circuit.py:15` and its own docstring is
`Circuit builder for mid-circuit measurement and classical feedback.` Its public
methods, obtained by `inspect.getmembers` filtering to non-underscore functions,
include exactly four dynamic ones — `measure`, `reset`, `conditional`, `state` —
plus the whole inherited static surface of `Circuit`, including `run_distributed`
and `to_device`, which is where premise 9 bites.

### 2. `DynamicCircuit` is not exported from the root, and `run_dynamic` only from `experimental`

**Measured.** The root package does not carry the name:

```console
$ grep -n "DynamicCircuit\|dynamic" flagquantum/__init__.py
[exit code: 1]
$ python -c "import flagquantum as fq; print(hasattr(fq, 'DynamicCircuit'))"
hasattr(fq, 'DynamicCircuit') = False
```

The candidate namespace is the only import path, and the experimental facade is a
separate lazy module. `flagquantum/experimental/dynamic.py:1` is the marker and
`flagquantum/experimental/dynamic.py:8-12` is the export set:

```python
"""Unstable task-level dynamic-circuit execution and backend assessment."""
_PUBLIC_NAMES = (
    "assess_dynamic_backend",
    "run_dynamic",
)
__all__ = _PUBLIC_NAMES
```

`flagquantum/dynamic.py:3` and `flagquantum/dynamic.py:5` are the candidate
side: `from .runtime.dynamic.circuit import DynamicCircuit` and
`__all__ = ("DynamicCircuit",)`.

### 3. `run_dynamic` is defined once, its docstring states no stability, and its type check calls the class experimental

`run_dynamic` is defined at `flagquantum/runtime/dynamic/execution.py:763`, is
re-exported by `flagquantum/runtime/dynamic/__init__.py:14` and `:31`, and reaches
users only through `flagquantum/experimental/dynamic.py:17`. Its exact signature
and full docstring, read from the source, are:

```python
def run_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
    strategy: str = "auto",
    max_batched_bytes: int = 256 * 1024**2,
    noise_model: NoiseModel | None = None,
    _feedback_plan: DynamicFeedbackPlan | None = None,
) -> DynamicExecutionResult:
    """Execute dynamic shots using a reference or vectorized trajectory path."""
```

The docstring says nothing about stability. The stability statement is instead an
error string at `flagquantum/runtime/dynamic/execution.py:780`:

```python
raise TypeError("run_dynamic requires an experimental DynamicCircuit")
```

which is itself inaccurate, because `flagquantum/dynamic.py:1` calls the class
candidate-stable. The result type is `DynamicExecutionResult`
(`flagquantum/runtime/dynamic/result.py:13`), which
`contracts/dynamic-circuit-v1-candidate.json` lists under
`excluded_from_stable_extension`.

### 4. `CircuitIR` carries no classical register, no result bit, and no condition

**Measured.** `flagquantum/core/ir/__init__.py` declares the two dataclasses. `Instruction`
(`flagquantum/core/ir/__init__.py:339-346`) has exactly five fields:

```console
CircuitIR: ['n_wires', 'instructions', 'version', 'dtype', 'shape', 'observables', 'measurements', 'metadata']
Instruction: ['name', 'wires', 'params', 'matrix', 'metadata']
```

`ObservableNode` (`:386-389`) adds `coefficient`. `MeasurementNode`
(`:406-409`) adds `kind`, `shots`. **No field in any of the four types holds a
classical register, a result bit, or a condition.** A classical bit is a key
inside the untyped `Instruction.metadata` mapping; a condition is a tuple inside
the same mapping; a classical register exists only as a width computed at
runtime by `classical_width` (`flagquantum/runtime/dynamic/_conditions.py:43`).
`CircuitIR.validate` (`:460`) validates wires, versions, dtypes, shapes,
observables, and measurement kinds, and never inspects `metadata` for a
classical-control key.

`docs/public_api_v1.json` lists `CircuitIR`, `Instruction`, and `IR_VERSION` among
its 34 `stable_exports`, so the shape above is a protected contract, not an
internal one. `IR-007` decision 2 rejects "any change to `IR_VERSION` or to
`CircuitIR` schema `1.0`", and its rejected-alternatives section rejects
extending the public `CircuitIR` schema to carry dynamic semantics by `§ 5.1`,
by `IR_006`, and by the compatibility promise. `IR-007` is currently `Status:
Proposed`.

### 5. A conditional instruction is constructible, and one of the two spellings is refused only at the execution boundary

**Measured.** Both metadata spellings construct without error, because
`Instruction.__post_init__` (`flagquantum/core/ir/__init__.py:348`) only requires an
unknown opcode to carry `is_dynamic` or `matrix`; it never validates a condition:

```console
--- probe A: metadata with conditions ---
OK constructed: Instruction(name='x', wires=(1,), params={}, matrix=None, metadata={'conditions': ((0, 1),)})
--- probe B: metadata with condition_clauses ---
OK constructed: Instruction(name='x', wires=(1,), params={}, matrix=None, metadata={'condition_clauses': (((0, 1),),)})
```

The refusal is deferred to the shared helper, where the two spellings diverge:

```console
$ PYTHONPATH=$PWD .venv/bin/python -c "<probe below>"
conditions -> instruction_conditions => ((0, 1),)
condition_clauses -> instruction_conditions RAISED ValueError : complex condition clauses require a DNF-aware execution path
both -> instruction_condition_clauses RAISED ValueError : instruction cannot define both conditions and condition_clauses
condition_clauses -> instruction_condition_clauses => (((0, 1),),)
```

So `conditions` is the public spelling and is a conjunction;
`condition_clauses` is a private disjunctive-normal-form spelling
(`flagquantum/runtime/dynamic/_conditions.py:19`) that only the private hybrid
compiler produces (`flagquantum/compiler/_hybrid/dynamic_lowering.py:428`). The
refusal at `_conditions.py:11` is a bare `ValueError`, not a member of the stable
error hierarchy in `flagquantum/errors.py`.

### 6. The planner refuses dynamic programs wholesale, before it can diagnose them

**Measured.** The refusing function is `_validated_plan_program` at
`flagquantum/runtime/planner/__init__.py:715`, and the guard is at `:740-747`:

```python
if any(
    instruction.metadata.get("is_dynamic") or instruction.metadata.get("conditions")
    for instruction in source_ir.instructions
):
    raise CapabilityError(
        "fq.run does not execute dynamic trajectories; use "
        "fq.experimental.dynamic.run_dynamic(..., shots=...)"
    )
```

A probe that builds `h(0)`, `measure(0, classical_bit=0)`,
`conditional("x", 1, classical_bit=0, equals=1)` and calls `fq.plan` produces:

```console
RAISED CapabilityError : fq.run does not execute dynamic trajectories; use fq.experimental.dynamic.run_dynamic(..., shots=...)
```

Two observations follow. First, the guard tests `is_dynamic` and `conditions` and
does not test `condition_clauses`, so a program built from the private DNF
spelling is not caught by this guard. Second, because the guard fires first, no
planner-side diagnosis of a malformed or ill-ordered condition is reachable
through `fq.plan`. `fq.run` produces the same `CapabilityError`, and
`DynamicCircuit.state` refuses with `CapabilityError` from
`flagquantum/runtime/dynamic/circuit.py:90`.

### 7. The condition/measurement def-use rule is enforced, at shot time, with an untyped error

**Measured.** A condition on a classical bit that no measurement produced is
rejected — but only when shots are executed, and with a bare `RuntimeError`:

```console
=== condition on a bit that was never measured ===
RAISED RuntimeError : classical bit 3 was read before measurement
=== condition on bit 0, no measure at all ===
RAISED RuntimeError : classical bit 0 was read before measurement
```

The two raise sites are `flagquantum/runtime/dynamic/execution.py:301` (reference
trajectory) and `:534` (batched trajectory):

```python
if classical[bit_index] < 0:
    raise RuntimeError(f"classical bit {bit_index} was read before measurement")
```

`RuntimeError` is not in the stable hierarchy in `flagquantum/errors.py`, and it
is not a `CapabilityError` or a `ValidationError`. This is not only a style
finding: the frozen contract already requires the other class. Read from
`contracts/dynamic-circuit-v1-candidate.json`:

```json
"error_contract": {
  "invalid_wire_or_condition": "ValidationError",
  "state_on_dynamic_program": "CapabilityError",
  "wrong_python_type": "TypeError"
}
```

So `invalid_wire_or_condition` is contractually `ValidationError`, and the two
`RuntimeError` sites above are non-conforming today. `API_CHANGE_PROPOSAL_010`
recorded the same rule in prose, and criterion 2 of
[`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md)
depends on it.

### 8. A provider cannot express "I execute conditionals" as a Core capability fact

There are three separate declaration mechanisms today, and none of them is the
Core one.

**Measured, mechanism one.** `CloudBackendProfile`
(`flagquantum/deployment/cloud.py:40-54`) carries `supports_dynamic_circuits:
bool = False` (`:50`), `max_classical_bits: int | None = None` (`:51`), and
`dynamic_dialect: str | None = None` (`:54`). `assess_dynamic_backend`
(`flagquantum/runtime/dynamic/deployment.py:38`) reads them by `getattr` and
appends the string `backend_does_not_declare_dynamic_circuit_support` (`:47`).
Two shipped profiles set the flag: `flagquantum/remote/qpu/braket.py:164` sets it
to `is_iqm`, and `flagquantum/remote/qpu/azure.py:86` sets it to `False`.

**Measured, mechanism two.** `DynamicFeatureSet`
(`flagquantum/runtime/dynamic/conformance.py:19-30`) declares
`mid_circuit_measurement`, `reset`, `supported_conditional_gates`,
`max_condition_bits`, `condition_values`, `supports_disjunctive_conditions`,
`returns_mid_circuit_measurements`, and `supports_repeated_measurement`. It is an
executor-side description used by `assess_dynamic_features` (`:132`), and it has
no field for a per-rank or feedback-carrying execution.

**Measured, mechanism three.** The Core vocabulary is closed and contains no such
fact. `CAPABILITY_NAMES` (`flagquantum/core/target_capabilities.py:23-45`) is a
frozenset of 19 names, and the probe refuses a conditional-execution name
outright:

```console
declared_facts='gates.conditional_execution': RAISED TargetDescriptionError: target description contains unknown v1 capability names: 'gates.conditional_execution'
declared_facts='control.conditionals': RAISED TargetDescriptionError: target description contains unknown v1 capability names: 'control.conditionals'
```

`TargetDescription` (`flagquantum/ecosystem/extensions/target_sdk.py:252-284`)
exposes only `observed_facts` and `declared_facts`, and
`_validate_declared_fact_names` (`:341`) restricts declared names to
`AUTHORITATIVE_STATIC_DECLARATION_ALLOWED` (`target_capabilities.py:47-60`). The
declared collection `artifacts.profiles` is an open string array validated only as
an array (`target_capabilities.py:320-326`), but the emission profiles that give
those strings meaning are a closed registry, `EMISSION_PROFILES` at
`flagquantum/compiler/target_emission.py:89-107`: `openqasm-2.0`, `openqasm-3.0`,
`qcis-1.0`, `qir-2.0`. None of the four is a hybrid or conditional profile, so
`artifacts.profiles` cannot currently state the fact either. `target_legalization.py`
does not read a condition at all: its only `metadata` access is
`item.metadata.get("fq_output_kind", item.kind)` at
`flagquantum/compiler/target_legalization.py:128`.

### 9. The sharded entry point silently executes a conditioned gate unconditionally

**Measured.** There is a sharded multi-rank path in `flagquantum/runtime/`, and it
has no dynamic guard. `flagquantum/runtime/execution.py` contains zero occurrences
of the strings `is_dynamic`, `conditions`, or `conditional`, and the distribution
package contains none either:

```console
$ grep -cE "is_dynamic|conditions|conditional" flagquantum/runtime/execution.py
0
$ grep -rn "world_size" flagquantum/runtime/dynamic/ | grep -v pycache | wc -l
0
$ grep -rnE "dynamic|conditional|feedback" flagquantum/runtime/distributed/*.py | wc -l
0
$ ls -d flagquantum/_compiler
ls: flagquantum/_compiler: No such file or directory
```

`Circuit.run_distributed` (`flagquantum/circuit.py:622`) forwards to
`flagquantum/runtime/execution.py:231`, which reaches
`simulate_distributed_statevector_local`
(`flagquantum/runtime/executors/statevector/local_execution.py:345`). That loop
builds one matrix per instruction and applies it. Its only metadata read in the
whole file is the string `metadata-only` in a docstring at `:377`; the matrix
lookup `_instruction_matrix` reads `instruction.matrix`, `instruction.name`,
`parameter_slots`, and `parameter_constants` and never
`instruction.metadata` (`flagquantum/simulation/statevector/operations.py:124`).

A program that carries a condition and no dynamic measurement therefore executes
its conditioned gate unconditionally, with no error, no blocker, and no metadata:

```console
ir: (Instruction(name='h', wires=(0,), params={}, matrix=None, metadata={}), Instruction(name='x', wires=(1,), params={}, matrix=None, metadata={'conditions': ((0, 1),)}))
returned LocalDistributedStatevectorResult
  state present
dynamic-path state: tensor([[0.0000+0.j, 0.7071+0.j, 0.0000+0.j, 0.7071+0.j]])
static h,x state   : tensor([[0.0000+0.j, 0.7071+0.j, 0.0000+0.j, 0.7071+0.j]])
```

The two states are identical: the condition was dropped. This is exactly the
silent flattening to a static circuit that criterion 6 of
[`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md)
forbids.

When the program also carries a dynamic measurement, the same path fails with an
untyped `KeyError` rather than a typed refusal:

```console
$ PYTHONPATH=$PWD .venv/bin/python -c "from flagquantum.dynamic import DynamicCircuit; c=DynamicCircuit(2); c.h(0); c.measure(0, classical_bit=0); c.conditional('x',1,classical_bit=0); c.run_distributed(world_size=1)"
...
  File ".../flagquantum/simulation/statevector/operations.py", line 124, in _instruction_matrix
  File ".../flagquantum/simulation/gate_matrix.py", line 118, in gate_matrix
    gate = GATE_MAT_DICT[name]
KeyError: 'measure'
```

`flagquantum/simulation/gate_matrix.py:118` subscripts a plain dict, so an opcode
the table does not know is a `KeyError` and not a `CapabilityError`.

There is no sharded path that carries feedback, because there is no sharded
dynamic path at all: `run_dynamic` has zero occurrences of `world_size`, and
`capability-maturity.toml:614` records `distribution_semantics = "single_process"`
for `capabilities.dynamic_circuits`. Nothing here is a capacity claim.

### 10. `while` and repeat-until-success do not exist, and one of them is explicitly refused

**Measured, negative search.** Searches for a repetition construct in Python,
documentation, and contracts return nothing that is a construct; the single hit in
the tree is prose:

```console
$ grep -rniE "repeat[-_]until|while[-_]loop|RepeatUntil|repeat_until_success" --include=*.py . | grep -v "/.venv/" | wc -l
0
$ grep -rniE "repeat.until.success|while.loop|repeat_until|RepeatUntil" flagquantum/ docs/ contracts/ tests/ 2>/dev/null
docs/development/HYBRID_COMPILATION_PHASE11_EVIDENCE.md:36:tensor state, while loops, break/continue, exceptions, arbitrary mutation,
$ grep -rnE "\"(if|while|for)\"|'(if|while|for)'" flagquantum/core/ flagquantum/runtime/ | wc -l
0
```

The only producer of a repetition construct is the private hybrid compiler, which
refuses a data-dependent loop at capture:

```console
$ grep -n "While\|while" flagquantum/compiler/_hybrid/capture.py
202:            elif isinstance(statement, ast.While):
203:                self.fail(statement, "control.while", "while is unsupported")
```

Its loop operation is `scf.for` (`flagquantum/compiler/_hybrid/schemas.py:37`),
bounded by construction, and `BoundedLoopUnrollPass`
(`flagquantum/compiler/_hybrid/transforms.py:187`) preserves any loop whose bound
is not a constant with the remark `loops_preserved_dynamic_bounds`
(`flagquantum/compiler/_hybrid/transforms.py:207`). `IR-007` decision 2 rejects
"unbounded loops" from Phase 4 scope, and `IMPLEMENTATION.md:69-70` records that
"Conditional measurement/reset and measurement-dependent termination remain
outside the profile". There is no
repeat-until-success construct anywhere, so this row asks for a construct that has
no precedent, no producer, and no consumer in the repository.

### 11. The private program level already exists and already owns `if` and bounded loops

The plan row assumes the classical-control work lands in a new internal level, and
`IR-007` fixes that level's layout at `flagquantum/_compiler/**`. That directory
does not exist (**Measured**:
`ls -d flagquantum/_compiler` → `No such file or directory`), and no
`ProgramModule` symbol exists in the tree (`grep -rn "ProgramModule" --include=*.py
flagquantum/` returns no match). But a program-level IR with typed SSA values and
structured control flow does exist, privately, at
`flagquantum/compiler/_hybrid/`: `HybridProgram`/`Region`/`Block`/`Operation`
(`flagquantum/compiler/_hybrid/model.py:203`, `:188`, `:170`, `:137`), the closed
vocabulary `scf.for`, `scf.if`, and `scf.yield`
(`flagquantum/compiler/_hybrid/schemas.py:37-39`), a verifier
(`flagquantum/compiler/_hybrid/verifier.py`), and an ordered pass pipeline with
mandatory re-verification (`flagquantum/compiler/_hybrid/passes.py:88`,
`run_pass_pipeline`). Its `IMPLEMENTATION.md` states, quoted literally with line
numbers, that it is "private Compiler machinery and is not exported from
`flagquantum` or `flagquantum.compiler`" (`:5-6`), that it "does not own static
circuit semantics, target execution, numerical state, devices, providers,
gradients, or framework capture" (`:8-9`), and that "loop measurements, durable
sessions, accelerators, and distributed execution fail closed or remain outside
the contract" (`:125-126`).

Two measured consequences follow. A `while` cannot be lowered through this path
either — the capture refusal in premise 10 is the only entrance. And the
classical-register question the plan row raises has a home already, inside this
private package, that is not the public `CircuitIR`.

### 12. `API_CHANGE_PROPOSAL_062` and the plan row disagree about `run_dynamic`

`API_CHANGE_PROPOSAL_062` decision 3 states that `run_dynamic`, providers,
dialects, and native results "remain experimental regardless of the outcome for
`DynamicCircuit`, and this proposal does not schedule their promotion" (`:61-63`),
and its non-goals repeat the exclusion as "Adding public syntax for conditions,
classical registers, or loops beyond what `DynamicCircuit` already exposes".
`IR-007` decision 7 states the same. The plan row
`W5-10` asks instead for `run_dynamic` to leave the experimental namespace, and
`062` is the row `W0-05b` names as the promotion standard. This is a genuine
conflict between two recorded documents, not a reading of either, and this
proposal resolves it fail-closed rather than by picking a winner.

### 13. Every dynamic end-to-end test runs at the trajectory boundary, none at a sharded one

**Measured.**

```console
$ grep -rln "DynamicCircuit" tests/ --include=*.py | wc -l
10
$ grep -rln "run_dynamic" tests/ --include=*.py | wc -l
9
$ grep -rlnE "run_distributed|to_device" tests/ --include=*.py | xargs grep -ln "classical_bit\|conditional" 2>/dev/null
[exit code: 1]
```

The ten files are `tests/unit/test_dynamic_architecture.py`,
`tests/unit/test_dynamic_circuit_candidate.py`, `tests/test_amazon_braket_provider.py`,
`tests/test_braket_iqm_dynamic.py`, `tests/test_dynamic_circuit.py`,
`tests/test_dynamic_conformance.py`, `tests/test_dynamic_feedback.py`,
`tests/test_dynamic_noise.py`, `tests/test_dynamic_observability.py`, and
`tests/api_contract/test_experimental_namespace_contract.py`. The nine that call
`run_dynamic` reach it as `fq.experimental.dynamic.run_dynamic`, that is, through
the experimental facade. Six files call `run_distributed` or `to_device` —
`tests/test_distributed_statevector.py`, `tests/test_native_circuit.py`,
`tests/test_backends.py`, `tests/unit/test_canonical_runtime_namespaces.py`,
`tests/unit/test_flagos_transport_observability.py`, and
`tests/distributed/test_mps_accelerator_backward_evidence.py` — and none of the
six contains the strings `classical_bit` or `conditional`, so no test exercises a
classical-control program through a sharded path. That is why the silent drop in
premise 9 has never been observed by the suite.

## Decision

### 1. The promotion label is not this document's to move, and this document does not restate the standard

`DynamicCircuit`'s transition from candidate-stable to stable is governed
entirely by the eight criteria in
[`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md),
which is the `W0-05b` draft. This proposal does not restate those criteria, does
not weaken them, and does not grant the promotion. It adds only the measurements
and the typed failures that criteria 1, 2, 4, 5, and 6 need in order to be
assessed. Where a decision below touches a criterion, the dependency is named
rather than re-specified.

**Refused:** any statement in this document that the candidate label changes.
062's own Status header records "no change is made by this proposal until every
criterion below holds" (`:5-6`), and the promotion is therefore an action of 062 or
of its successor, taken by the API owner.

### 2. `run_dynamic` stays in `flagquantum.experimental.dynamic`

**Refused.** This proposal does not move `run_dynamic` out of the experimental
namespace, and does not add it to the candidate-stable namespace
`flagquantum.dynamic`, and does not add a root export for it. The reason is that
062 decision 3 and `IR-007` decision 7 both record that `run_dynamic` remains
experimental, and `contracts/dynamic-circuit-v1-candidate.json` lists
`run_dynamic` under `excluded_from_stable_extension`. Premise 12 records the
conflict with the plan row; the owner resolves that conflict by amending the plan
row or by amending 062, and this proposal takes the fail-closed side until then,
because moving the name is a public-surface change that no measurement in this
document can justify.

**Migration:** none, because nothing moves. The required evidence list includes the
item that records the owner's resolution.

### 3. Public `CircuitIR` schema `1.0` gains no field, and `IR_VERSION` does not change

**Refused.** No classical-register field, no result-bit field, and no condition
field is added to `CircuitIR`, `Instruction`, `MeasurementNode`, or
`ObservableNode`, because `docs/public_api_v1.json` protects all four names,
because `IR-007` decision 2 rejects any change to `IR_VERSION` or to `CircuitIR`
schema `1.0`, and because `IR-007`'s rejected-alternatives section already rejects
extending that schema to carry dynamic semantics.

The plan row's phrase about classical registers entering the IR is satisfied
instead inside the internal level that `IR-007` decision 1 admits —
`ProgramModule` with classical values, result values, and explicit def-use — and
that level does not exist yet. Until it does, the only classical-register
representation is the one measured in premise 4: keys inside
`Instruction.metadata`. Decision 4 gives that representation a validated contract
instead of a schema change.

**Migration:** none, because the serialized schema is untouched.

### 4. The four private classical-control keys become one validated internal contract

**Added,** as an internal Core-owned module `flagquantum/core/classical_control.py`,
not exported from `flagquantum` and not added to any public contract:

```python
CLASSICAL_CONTROL_KEYS: tuple[str, ...] = (
    "is_dynamic",
    "classical_bit",
    "conditions",
    "condition_clauses",
)

def validate_classical_control(
    instructions: Iterable[Instruction],
    *,
    n_wires: int,
) -> int: ...

def is_classical_control_instruction(instruction: Instruction) -> bool: ...
```

`validate_classical_control` returns the classical width (replacing the private
`classical_width` in `flagquantum/runtime/dynamic/_conditions.py:43`) and fails
closed on: an unknown key inside the measured key set; a negative `classical_bit`;
a condition value outside `{0, 1}`; both `conditions` and `condition_clauses` on
one instruction; a non-canonical clause list; and any key the four-key set does
not name but a caller passes as a fifth classical-control key. Error classes come
from the stable hierarchy: `ValidationError` for a malformed value, `TypeError`
for a wrong Python type, exactly the mapping already frozen as
`error_contract` in `contracts/dynamic-circuit-v1-candidate.json`.

The reason for a single validator rather than per-call-site checks is measured in
premise 6: the planner guard tests two of the four keys and not the other two, and
premise 5 shows the two condition spellings already diverging across five call
sites. One validator removes the possibility of a third spelling.

### 5. A condition on an unmeasured classical bit is refused at build time with `ValidationError`

**Added, and a behaviour change.** `DynamicCircuit._append_dynamic`
(`flagquantum/runtime/dynamic/circuit.py:18`) calls `validate_classical_control`
against the instructions appended so far, so a condition naming a classical bit
that no preceding `measure` produced is refused with `ValidationError` when the
condition is built, and never reaches a shot. `run_dynamic` calls the same
validator on the whole program before its first `shots` subdivision, as a
redundant gate for a program assembled by direct `Instruction` injection.

The two shot-time raise sites measured in premise 7 —
`flagquantum/runtime/dynamic/execution.py:301` and `:534` — change from
`RuntimeError` to `ValidationError` so that one defect has one error class.

This change restores conformance with the frozen contract rather than departing
from it: `contracts/dynamic-circuit-v1-candidate.json` already records
`error_contract.invalid_wire_or_condition = "ValidationError"`, so the observed
`RuntimeError` is the deviation. It is still an observable break for a caller that
catches `RuntimeError` today, which is why the migration paragraph below exists,
and the same contract records
`rules.future_breaking_changes_require_api_owner_approval = true`, so the change
is listed in the checklist as requiring the API owner's approval even though it
makes the code match the contract. It is also the fail-closed direction required
by engineering decision principle 9.

**Why this is the earliest knowable stage:** the def-use relation is a property of
the instruction sequence, which is complete at build time, so nothing about the
defect requires a shot, a seed, an execution strategy, or a rank. Diagnosing it per
trajectory, once per shot, is strictly later and strictly more expensive.

**Migration:** any caller that catches `RuntimeError` around `run_dynamic` for
this defect must catch `ValidationError` instead. `ValidationError` is a
`ValueError` and a `FlagQuantumError`; it is not a `RuntimeError`. This is a
breaking change inside an experimental surface, so it needs the updated
documentation and test evidence in the checklist, not a deprecation cycle.

### 6. The sharded entry point refuses a classical-control program instead of dropping it

**Added.** `flagquantum/runtime/execution.py:231` `run_distributed` gains a leading
fail-closed refusal. When any instruction in the input IR satisfies
`is_classical_control_instruction`, it raises `CapabilityError` with a message that
names the blocker `dynamic_conditional_execution_unsupported` and names
`fq.experimental.dynamic.run_dynamic(..., shots=...)` as the supported entry.
`Circuit.run_distributed` and `Circuit.to_device` inherit the gate because they
forward to it.

**Why refuse rather than implement:** premise 9 measures that no sharded path
carries feedback, that the distributed statevector executor never reads
`instruction.metadata`, and that the current outcome is a silently wrong state. A
silent wrong answer is worse than a refusal, and engineering decision principle 9
forbids the silent path. Implementing sharded feedback is not proposed here
because the def-use, ordering, and join rules it needs are exactly the `IR-007`
decision 6 contract, which has not landed; building the execution before the
contract would violate `ARCH_010` clause 3 and
`docs/development/MULTI_TEAM_DEVELOPMENT.md`.

**Refused:** any wording in this proposal, in the message, or in the evidence that
describes replicated per-rank execution as distributed scalability. The dynamic
path stays `single_process`; a sharded feedback path, if it is ever built, must
report `distribution_semantics="sharded_across_ranks"` with the full evidence set
before any capacity language is permitted.

### 7. `while` and repeat-until-success are refused; bounded loops are not added here either

**Refused.** This proposal adds no `while` operation, no repeat-until-success
operation, no data-dependent loop bound, and no measurement-dependent
termination, to the public dynamic surface or to any internal level. The reasons
are premise 10 (no construct, no producer, no consumer exists anywhere in the
tree, and the only repetition producer refuses a data-dependent loop at capture)
and `IR-007` decision 2, which rejects unbounded loops from the scope that would
own them. A data-dependent loop also multiplies the feedback question in decision
6 rather than answering it: its trip count depends on a mid-circuit measurement,
so it cannot be planned before execution on any target, sharded or not.

**Refused, separately:** copying, wrapping, or re-exporting the private
`flagquantum/compiler/_hybrid/` package to give the dynamic surface a loop
construct. Premise 11 measures that package as private Compiler machinery that is
not exported from `flagquantum` or `flagquantum.compiler`
(`IMPLEMENTATION.md:5-6`); § 19.13 of
`MULTI_LEVEL_IR_ARCHITECTURE.md` places its replacement under `flagquantum/_compiler/`,
which does not exist. A second internal home would be the parallel scaffolding
that `ARCH_010` clause 2 prohibits.

### 8. A provider that cannot execute a conditional is refused by name, and the Core vocabulary is not extended yet

**Added.** The refusal path that already exists is made explicit and total, and its
blocker strings are named here as the contract:

| Blocker | Condition that produces it |
| --- | --- |
| `circuit_exceeds_backend_qubit_capacity` | `circuit.n_wires > backend.n_wires` (exists, `deployment.py:43`) |
| `backend_does_not_support_openqasm` | the target declares no OpenQASM emission (exists, `:45`) |
| `backend_does_not_declare_dynamic_circuit_support` | the target does not declare conditional execution (exists, `:47`) |
| `required_classical_bits_exceed_backend_limit` | declared classical width exceeds the target limit (exists, `:51`) |
| `dynamic_conditional_execution_unsupported` | the request names a sharded or multi-rank execution path (new) |

The new blocker is added because premise 9 shows the current refusal set has no
member that describes a multi-rank request, so a caller cannot distinguish "this
target has no conditionals" from "this target has conditionals but no sharded
feedback". Both must be refusals with different names.

**Refused, for now:** adding a conditional-execution fact to
`CAPABILITY_NAMES` (`flagquantum/core/target_capabilities.py:23`). The reason is
that a name in that frozenset is only meaningful with a runtime-protocol handler
that can produce or check it, and premise 8 measures that no handler, no
`EMISSION_PROFILES` entry, and no `TargetDescription` path can carry the fact. A
name added ahead of its handler would be a vocabulary entry that no producer can
populate and no consumer can check — the speculative layer that engineering
decision principle 2 forbids. The fact belongs in the `IR-007` `TargetModule`,
whose contract does not exist.

**Refused, separately:** deriving conditional support from the existing
`artifacts.profiles` fact by inventing a profile string, because the profile
registry is closed (premise 8) and an unregistered string states nothing to any
consumer while appearing to state something to a reader. Fail-closed means the
fact is absent until it can be expressed, not present as a string nobody reads.

### 9. The private hybrid package is consumed, not duplicated, and its measurement is reported rather than assumed

`flagquantum/compiler/_hybrid/` is the only implementation in the tree that
lowers a measurement-derived `if` to conditioned instructions, and its
`IMPLEMENTATION.md` records the ceilings it enforces: `max_condition_clauses`
defaulting to 64, a bounded unroll limit, and fail-closed handling for
measurement-dependent loop bounds, conditional measurement, and stochastic
gradients. This proposal **adds nothing** to that package and **removes nothing**
from it. It records that its existence is the reason decision 7 refuses a second
loop mechanism and the reason decision 4 can state one classical-control contract
instead of inventing a fourth representation. Whether the public dynamic path
should route through it, or through a future `flagquantum/_compiler/` level, is an
`IR-007` question and not an API question, so it is left open here and named as an
open item rather than decided.

### 10. `DynamicExecutionResult` remains outside the stable contract

**Refused.** This proposal does not promote `DynamicExecutionResult`, does not
change its fields, and does not change `to_execution_result`. 062 excludes it and
`contracts/dynamic-circuit-v1-candidate.json` lists it under
`excluded_from_stable_extension` with the reason that native final-state
availability and provider metadata differ by implementation. One measured
consequence of staying outside is recorded for the record: `DynamicExecutionResult`
already exposes `classical_register` as a property alias
(`flagquantum/runtime/dynamic/result.py:32`) over the `classical_bits` field, so a
future promotion must settle which of the two names is the contract before it can
freeze either.

## Public API

**No public name is added, removed, or renamed by this proposal, and no root
export changes.** Two aggregate contracts are explicitly unchanged:

- `docs/public_api_v1.json` **does not change.** It enumerates the 34 root
  `stable_exports`, which already include `CircuitIR`, `Instruction`, and
  `IR_VERSION`, and it contains zero occurrences of the strings `dynamic` or
  `Dynamic`. No root name is added, and `flagquantum.dynamic` is a candidate
  namespace rather than a root export, so there is nothing in that file for this
  proposal to edit. Changing it would be a promotion action governed by 062, which
  this proposal does not request.
- `contracts/public-api-v1-candidate.json` **does not change.** Its
  `stable_extensions` array (line 50) lists `flagquantum.runtime`,
  `flagquantum.simulation.mps`, `flagquantum.simulation.tensor_network`,
  `flagquantum.deployment`, and others, and contains zero occurrences of
  `flagquantum.dynamic`. The dynamic candidate extension is recorded in
  `contracts/dynamic-circuit-v1-candidate.json`, whose supersession by a stable
  contract is criterion 8 of 062 and therefore 062's action, not this document's.
  Adding an entry here now would be a second source of truth for the same
  extension, which decision 4 avoids on purpose.

What is added, with the exact names and the namespaces they live in:

| Name | Namespace and file | Kind | Signature |
| --- | --- | --- | --- |
| `CLASSICAL_CONTROL_KEYS` | `flagquantum.core.classical_control`, `flagquantum/core/classical_control.py` | internal constant | `tuple[str, ...]`, exactly the four keys in decision 4 |
| `is_classical_control_instruction` | same module | internal function | `(instruction: Instruction) -> bool` |
| `validate_classical_control` | same module | internal function | `(instructions: Iterable[Instruction], *, n_wires: int) -> int` |
| `dynamic_conditional_execution_unsupported` | string blocker in `flagquantum.runtime.dynamic.deployment`, `flagquantum/runtime/dynamic/deployment.py` | blocker code inside the existing `DynamicBackendCompatibility.blockers` tuple, and the message key of the `CapabilityError` raised by `run_distributed` | n/a |

Three existing signatures change and are listed here so the change is not
discovered from a stack trace:

| Name | File | Before | After |
| --- | --- | --- | --- |
| `DynamicCircuit._append_dynamic` | `flagquantum/runtime/dynamic/circuit.py:18` | private, validates wires only | private, additionally calls `validate_classical_control` |
| `run_dynamic` | `flagquantum/runtime/dynamic/execution.py:763` | public signature above | unchanged signature; two internal raise sites change class |
| `run_distributed` | `flagquantum/runtime/execution.py:231` | no dynamic guard | leading `CapabilityError` for a classical-control program |

`flagquantum/runtime/execution.py:run_distributed` is reached from `Circuit.run_distributed`
(`flagquantum/circuit.py:622`) and `Circuit.to_device` (`:629`), so both inherit the
refusal. No new parameter is added to either.

**Explicitly not public:** `flagquantum/core/classical_control.py` is not exported
from `flagquantum`, is not added to `flagquantum.core.__all__` if that module
declares one, and is not documented in `docs/reference/API.md`. It is an internal
contract between Compiler, Runtime, and the dynamic executor, in the same spirit
as `flagquantum/compiler/_hybrid/` being private Compiler machinery that is not
exported from `flagquantum.compiler` (`IMPLEMENTATION.md:5-6`). Naming it here is
the justification required by human-maintainability guardrail 4 — "Require
justification for every new contract type" — and is not a publication.

## Compatibility

**What breaks.** Three things, all inside the experimental dynamic surface or on a
path that currently produces a wrong answer.

1. `flagquantum/runtime/dynamic/execution.py:301` and `:534` change the exception
   class for a condition on an unmeasured classical bit from `RuntimeError` to
   `ValidationError`. A caller that catches `RuntimeError` there loses the catch.
   `ValidationError` derives from `ValueError` and `FlagQuantumError`, and is not a
   `RuntimeError`. The migration is to catch `ValidationError`, or to catch
   `FlagQuantumError` if the caller wants every stable error, or to validate before
   execution so the exception is never raised.
2. `flagquantum/runtime/execution.py:231` `run_distributed` refuses a program that
   carries `is_dynamic`, `classical_bit`, `conditions`, or `condition_clauses`.
   A caller that previously reached the distributed statevector executor with such
   a program either received a silently wrong state (premise 9) or a `KeyError`.
   Both become one `CapabilityError`. The migration is to execute the program with
   `fq.experimental.dynamic.run_dynamic(..., shots=...)`, which is the only
   supported execution path for a conditional, and to treat decision 6 as the
   boundary statement until a sharded feedback path exists.
3. `DynamicCircuit._append_dynamic` now refuses a condition on a classical bit that
   no preceding `measure` produced. A program that built such a conditional and
   then never executed it will now fail at build time. The migration is to measure
   the bit first.

**What does not break.** The five `DynamicCircuit` signatures in the
`public_signatures` block of `contracts/dynamic-circuit-v1-candidate.json` — the
constructor plus `measure`, `reset`, `conditional`, and `state` — are unchanged, so
`measure`, `reset`, `conditional`, and `state` accept and return exactly what they
accept and return today. The `flagquantum.dynamic` import path is unchanged. The
`flagquantum.experimental.dynamic` facade is unchanged and `run_dynamic` keeps its
name, its parameters, and its defaults. `CircuitIR` serialization is byte-for-byte
unchanged, `IR_VERSION` is unchanged, and `content_hash` is unchanged, because no
field and no key is added or removed. Static circuits do not touch any changed
line: the new refusal in `run_distributed` reads the four metadata keys and
returns immediately for every static program measured in premise 4, and
`validate_classical_control` is called only on the dynamic construction path.
`Circuit.run_distributed` and `Circuit.to_device` keep their signatures and return
types for every static program.

**Anything that refuses more than before, and its migration.** The three items in
"What breaks" are the complete list. Nothing else in the tree refuses more after
this proposal than before it. In particular this proposal does not add a refusal
to `fq.plan`, `fq.run`, `fq.compile`, `fq.samples`, or `fq.counts`, and it does not
change the planner guard measured in premise 6 — a program that `fq.plan` accepts
today is accepted after this proposal, and a dynamic program that `fq.plan`
refuses today is refused with the same `CapabilityError` and the same message.

**Renames.** None. No name in this proposal is a rename, and no name is removed.

**Deprecation.** None is required, because every changed behaviour is inside a
surface that `capability-maturity.toml:617` records at `level = "experimental"`
and that `contracts/dynamic-circuit-v1-candidate.json` lists under
`excluded_from_stable_extension`. The two protected aggregate contracts listed in
"Public API" are unchanged, so no serialized public schema and no released provider
contract is affected.

## Required evidence before this proposal can be accepted

Every item below is a measurement to be taken, with its literal command and its
literal output recorded in this file's successor or in the pull request that
implements a decision. Nothing in this list is a result, and nothing here has been
built. An item that cannot be measured is recorded as an owned gap rather than
marked done.

- [ ] Measure that `docs/public_api_v1.json` is byte-identical before and after the
      change, and that its 34-name `stable_exports` list and `IR_VERSION` are
      unchanged.
- [ ] Measure that `contracts/public-api-v1-candidate.json` and
      `contracts/dynamic-circuit-v1-candidate.json` are byte-identical before and
      after the change.
- [ ] Measure that `dataclasses.fields(CircuitIR)`, `dataclasses.fields(Instruction)`,
      `dataclasses.fields(MeasurementNode)`, and `dataclasses.fields(ObservableNode)`
      return the same lists recorded in premise 4.
- [ ] Measure the def-use refusal for a condition naming each of: a bit one past the
      highest measured bit, a bit measured after the condition, and a bit measured
      on the same wire but a different index. Record the exact error class and
      message for each.
- [ ] Measure the build order of every conditional program in
      `tests/hybrid_compiler/test_dynamic_session.py`,
      `tests/hybrid_compiler/test_schedule_legalization.py`,
      `tests/test_dynamic_circuit.py`, `tests/test_dynamic_feedback.py`,
      `tests/test_dynamic_noise.py`, `tests/test_dynamic_observability.py`, and
      `tests/test_braket_iqm_dynamic.py`, and list every test that decision 5 would
      newly fail, or record that the list is empty.
- [ ] Measure the exact error class and message for each of the four
      classical-control keys injected into `run_distributed`, and confirm that no
      input reaches `flagquantum/simulation/gate_matrix.py:118`, so the
      `KeyError: 'measure'` recorded in premise 9 is no longer reachable through
      that entry point.
- [ ] Measure and record that the two states in premise 9 are no longer equal, by
      re-running that probe and confirming the call raises in place of returning an
      unconditioned state.
- [ ] Measure that every test in `tests/test_dynamic_circuit.py`,
      `tests/test_dynamic_conformance.py`, `tests/test_dynamic_feedback.py`,
      `tests/test_dynamic_noise.py`, `tests/test_dynamic_observability.py`,
      `tests/test_braket_iqm_dynamic.py`, `tests/unit/test_dynamic_architecture.py`,
      and `tests/unit/test_dynamic_circuit_candidate.py` passes after decision 5.
- [ ] Measure the blocker set produced by `assess_dynamic_backend` for a target
      with `supports_dynamic_circuits=False`, for the IQM profile at
      `flagquantum/remote/qpu/braket.py:164`, and for a multi-rank request, and
      record the exact tuple for each.
- [ ] Measure the checker that enforces `contracts/dynamic-circuit-v1-candidate.json`
      against the live `DynamicCircuit` signatures, and record that it reports no
      drift after decision 4 and decision 5.
- [ ] Measure static batched execution against the recorded baseline before and
      after the change, and record the comparison. Criterion 7 of 062 is the
      authority for what baseline and what threshold; this item reports against it
      and does not set one.
- [ ] For every numeric tolerance that decision 5 or decision 6 would introduce for
      a reproduced-state or reproduced-sample comparison, measure the margin the
      accepted rule produces and the divergence a plausible wrong rule produces —
      dropping the condition, applying it unconditionally, and applying it to the
      wrong classical bit are the three candidate wrong rules — and record both
      numbers. **No tolerance may be adopted before that measurement exists, and
      this proposal states no tolerance number.**
- [ ] Measure that `python tools/check_repository_language.py` reports
      `0 files contain Han-script text` after the document and any implementation
      notes are added.
- [ ] Measure that the negative searches in premise 10 still return zero for
      `while`, repeat-until-success, and a loop opcode in `flagquantum/core/` and
      `flagquantum/runtime/`, and record the commands.
- [ ] Measure that `flagquantum/compiler/_hybrid/capture.py:203` still refuses
      `ast.While` with `while is unsupported`, and record the literal message.
- [ ] Measure whether `flagquantum/_compiler/` exists and whether `IR-007` has moved
      off `Status: Proposed`. If either is false, record that decision 3's refusal
      stands and that the classical-register item of the plan row remains an
      explicit, owned gap.
- [ ] Measure and record whether `CAPABILITY_NAMES` gained a conditional-execution
      name, and if it did, record the runtime-protocol handler and conformance test
      that can produce and check it. Without both, record that decision 8's refusal
      stands.
- [ ] Record the owner's resolution of the conflict between plan row `W5-10` and
      decision 3 of `API_CHANGE_PROPOSAL_062`, naming which document was amended
      and on what date.
- [ ] Record the API owner's approval of the `RuntimeError` to `ValidationError`
      change in decision 5, as
      `contracts/dynamic-circuit-v1-candidate.json`
      `rules.future_breaking_changes_require_api_owner_approval` requires, and
      record that the change moves the two raise sites onto the
      `error_contract.invalid_wire_or_condition` value the same contract already
      freezes.
- [ ] Measure that `DynamicCircuit.conditional` and `DynamicCircuit.measure` still
      reject an invalid wire with `ValidationError`, and that `DynamicCircuit.state`
      still raises `CapabilityError`, matching `error_contract` after the change.
- [ ] Measure that the internal contract of decision 4 is registered on the
      integration branch together with a contract fake and a conformance test, per
      `IR-007` decision 6 and `docs/development/MULTI_TEAM_DEVELOPMENT.md`, and
      record the paths.

## Non-goals

- Promoting `DynamicCircuit`, `run_dynamic`, providers, dialects, or native
  results. The label question belongs to
  [`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md),
  and this proposal states no maturity change.
- Adding any field to `CircuitIR`, `Instruction`, `MeasurementNode`, or
  `ObservableNode`, or changing `IR_VERSION`. Decision 3 refuses this, and
  `IR-007` rejects it independently.
- Public syntax for `while`, repeat-until-success, data-dependent loop bounds, or
  any construct not already reachable from `DynamicCircuit`. Decision 7 refuses
  this.
- A general-purpose classical language, recursion, unbounded loops, dynamic
  memory, exceptions, closures, or higher-order values. `IR-007` decision 2 rejects
  these from the scope that would own them, and this proposal adds nothing that
  would reintroduce them.
- Implementing the internal `ProgramModule`, `QuantumModule`, or `TargetModule`
  levels. That is `W5-04` under `IR-007`, and this proposal only consumes their
  result and records their absence.
- Duplicating, wrapping, or re-exporting `flagquantum/compiler/_hybrid/`.
  Decision 9 consumes it as measurement and adds nothing to it.
- Building a sharded feedback path, a per-rank conditional evaluator, or any
  collective across ranks. Decision 6 refuses the silent path and does not replace
  it with an implementation.
- Any distributed scalability, capacity, performance, or parity claim. The dynamic
  path is `single_process`, a per-rank replicated execution is not scalability, and
  `scalability_claim_allowed` stays false.
- Adding a conditional-execution fact to `CAPABILITY_NAMES` ahead of the
  runtime-protocol handler and conformance test that would give it meaning.
  Decision 8 refuses this and names the evidence that would lift the refusal.
- Raising or lowering any `capability-maturity.toml` level, including
  `capabilities.dynamic_circuits`.
- Changing `assess_dynamic_backend`'s return type, `DynamicBackendCompatibility`'s
  fields, `DynamicFeatureSet`'s fields, or `DynamicExecutionResult`'s fields.
- Noise, calibration, pulse-level control, mid-circuit measurement timing, or
  measurement-dependent readout correction on the feedback path.
- QIR adaptive-profile emission, OpenQASM 3 hybrid-profile emission as a new
  `EMISSION_PROFILES` entry, or any change to `flagquantum/compiler/target_emission.py`
  beyond the refusals it already performs.
