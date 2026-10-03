# API Change Proposal 067: Control and adjoint modifiers, and an operator registration table

## Status

**Proposed. No code is written by this proposal and no behaviour changes until
the repository owner approves it and the acceptance items below are measured.**

This proposal fixes the contract W5-11 names: an arbitrary control-wire count, an
adjoint modifier, an operator registration table, and the backend lowering
contract a registered operator degrades through. Two other plan rows read this
document before they start. W6-01 lists W5-11 as its dependency, and the Wave 5
routing diagram sends `W5-04 ─► W5-10/11 ─► W6-01/06`, so the operator
vocabulary fixed here is the vocabulary W6-01 extends rather than replaces.
`docs/generated/OPERATOR_CAPABILITIES.md` is the third deliverable W5-11 names;
section 10 of "Decision" states exactly what regenerating it does and does not
mean.

Eight premises below were measured on this checkout before the proposal was
written, because engineering decision principle 6 requires inspecting what
already exists and principle 10 forbids claiming a gap that has not been
observed. Each measured premise is followed by the literal command and the
literal output, and every negative claim is followed by the search that
establishes the absence. Everything in "Required evidence" is a requirement on
the implementation, not a result: nothing here has been built.

## Problem

### 1. The operator schema table has no field that can hold a control count or an
adjoint rule

`flagquantum/core/operator_schema.py` defines `OperatorSchema` at line 11 with
ten fields, all of them required, and `OPERATOR_SCHEMAS` at line 130 holds 35
entries behind a `MappingProxyType`.

```
$ PYTHONPATH=$PWD python /tmp/probe067.py
FIELDS: ['opcode', 'aliases', 'arity', 'parameters', 'dtype_policy', 'semantic_kind', 'adjoint', 'decomposition', 'differentiable', 'wire_convention']
WITH_DEFAULTS: []
N_SCHEMAS: 35
ADJOINT_VOCAB: ['matrix_adjoint', 'not_applicable', 'self_inverse']
SEMANTIC_KIND_VOCAB: ['channel', 'unitary']
ARITY_GT_TWO: [('ccx', 3), ('cswap', 3)]
FIELDS_CONTAINING_CONTROL: []
WIRE_CONVENTION_3Q: [('control_0', 'control_1_or_target_0', 'target_or_target_1')]
```

Two facts follow directly and neither is an interpretation. First, exactly two
opcodes have arity greater than two, `ccx` and `cswap`, and both of them encode
"two controls" in their *name* rather than in a field. Second, no field name
contains `control`, so the number of control wires is not representable in the
schema at all: it is recoverable today only by looking up a table of known
opcode names. `_unitary` in the same file (line 49) is the constructor behind
every unitary schema and it selects `wire_convention` from a dictionary with
exactly three keys:

```
$ PYTHONPATH=$PWD python -c "from flagquantum.core.operator_schema import _unitary; [print(a, _unitary('op', a).wire_convention) for a in (1, 2, 3)]"
1 ('target',)
2 ('control_or_left', 'target_or_right')
3 ('control_0', 'control_1_or_target_0', 'target_or_target_1')
$ PYTHONPATH=$PWD python -c "from flagquantum.core.operator_schema import _unitary; _unitary('op', 4)"
KeyError: 4
```

So the table that would have to describe a three-control `x` cannot describe a
four-wire operator at all, and the wire-naming convention for the two existing
three-wire operators is one string tuple, not a rule that generalises.

### 2. Three unrelated things in this repository are already called "adjoint", and
none of them is a modifier

The word is currently overloaded, and the overload is measurable.

```
$ grep -rn "\.adjoint\b" --include=*.py flagquantum/ tools/ tests/ | grep -v __pycache__
flagquantum/core/operator_schema.py:173:            "adjoint": schema.adjoint,
flagquantum/runtime/executors/statevector/reverse_observable.py:49:    sweep.adjoint = seed_adjoint(
flagquantum/runtime/executors/statevector/reverse_adjoint_rotation_segment.py:41:            sweep.adjoint,
... (54 further matches, all in flagquantum/runtime/executors/statevector/reverse_adjoint*.py,
     flagquantum/simulation/statevector/adjoint.py and flagquantum/simulation/native_cpu/*.py)
```

Every match outside `flagquantum/core/operator_schema.py` belongs to the gradient
machinery: `sweep.adjoint` is an adjoint *state vector*, and
`flagquantum/simulation/statevector/adjoint.py` is the reverse-mode sweeper. The
same word appears a second time as a user-facing option:

```
$ grep -n "differentiation" flagquantum/algorithms/core.py | head -3
191:    differentiation: Literal["autograd", "adjoint"] = "autograd"
194:        raise ValueError("differentiation must be 'autograd' or 'adjoint'")
```

That option selects a *gradient* method and is scope of the plan row W2-03 and
`API_CHANGE_PROPOSAL_068_SV_GRADIENT_CONTRACT.md`, not of this document. The
schema field is a third thing: a three-value metadata string whose only reader
in the whole repository is the manifest renderer.

```
$ grep -rn "adjoint" --include=*.py flagquantum/core/ flagquantum/operators.py | grep -v "_unitary("
flagquantum/core/operator_schema.py:18:    adjoint: str
flagquantum/core/operator_schema.py:55:    adjoint: str = "matrix_adjoint",
flagquantum/core/operator_schema.py:70:        adjoint=adjoint,
flagquantum/core/operator_schema.py:85:        adjoint="not_applicable",
flagquantum/core/operator_schema.py:173:            "adjoint": schema.adjoint,
```

Line 173 is the only consumer, and it copies the string into the manifest. There
is no modifier: `Circuit` has none of the names a caller would try, and the
negative search is exhaustive over the three plausible spellings.

```
$ PYTHONPATH=$PWD python /tmp/probe067b.py
HAS_MCX: False | HAS_CCCCX: False
IR_REFUSED: 'mcx' -> unknown opcode 'mcx'; custom operations require an explicit matrix
IR_REFUSED: 'cccx' -> unknown opcode 'cccx'; custom operations require an explicit matrix
IR_REFUSED: 'adjoint_h' -> unknown opcode 'adjoint_h'; custom operations require an explicit matrix
IR_REFUSED: 'h_dg' -> unknown opcode 'h_dg'; custom operations require an explicit matrix
ARITY_REFUSED: 'cx' 4 wires -> opcode 'cx' requires 2 wire(s), got 4
CIRCUIT_ATTR_adjoint: False
CIRCUIT_ATTR_inverse: False
CIRCUIT_ATTR_dagger: False
CIRCUIT_ATTR_control: False
CIRCUIT_ATTR_ctrl: False
CIRCUIT_ATTR_conjugate: False
CIRCUIT_ATTR_power: False
CIRCUIT_DUNDER_invert: False | __pow__: False
```

```
$ grep -rniE "def adjoint|def dagger|def inverse|def control\(|with_adjoint" --include=*.py flagquantum/ ; echo "exit=$?"
exit=1
```

The refusal comes from `flagquantum/core/ir.py:364`, which is a deliberate rule
rather than an accident: an unknown opcode is admitted only when the instruction
carries an explicit matrix. A control-count-by-name design would therefore be
refused by the IR before it reached any backend, and each name a caller invented
would be a permanent addition to a name space the IR already treats as closed.

### 3. Control wires are fixed per opcode, and the nearest way a caller can ask
for more is silently ignored

`flagquantum/circuit.py:708` hard-codes the public qubit keywords.

```python
def _public_qubit_keywords(opcode: str, arity: int) -> tuple[tuple[str, ...], ...]:
    """Return public keyword aliases without changing internal wire vocabulary."""

    if arity == 1:
        return (("qubit", "target"),)
    if arity == 2 and opcode in _CONTROLLED_TWO_QUBIT_GATES:
        return (("control",), ("target",))
    if arity == 2:
        return (("qubit1", "left"), ("qubit2", "right"))
    if opcode == "ccx":
        return (("control1",), ("control2",), ("target",))
    if opcode == "cswap":
        return (("control",), ("target1",), ("target2",))
    return tuple((f"qubit{index + 1}",) for index in range(arity))
```

The two multi-control opcodes are the two literal branches, and the fallback
names wires `qubit1..qubitN`. `_install_gate_method` (line 724) consumes this
table to pop keywords, then forwards every remaining keyword to
`Circuit.gate`, which stores it as a parameter. The consequence is not an error
message; it is silence:

```
$ PYTHONPATH=$PWD python /tmp/probe067b.py
KWARG_IDENTICAL_TO_PLAIN_X: True
KWARG_STORED_PARAMS: {'controls': [0, 1]}
KWARG_PROBABILITY: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
```

`circuit.gate("x", [2], controls=[0, 1])` produced a circuit bit-identical to the
one without the keyword, recorded `{'controls': [0, 1]}` as instruction
parameters, and placed the amplitude in `|111>`, which is the state of an
uncontrolled `x(2)` after `x(0); x(1)` and not the state a caller asking for two
controls would expect. `IRValidationError` never fires because the parameters of
a schema are checked for *presence*, not for *absence of extras*
(`flagquantum/core/ir.py:371`). So the nearest thing to "arbitrary controls"
that compiles today is a silent wrong answer, which is the worst possible
starting point and the reason this proposal must fix a representation rather
than merely add one.

The rewrite a modifier would perform is already exact on this checkout, and the
competing convention is not close. Building the controlled block by hand, as
`identity` with the bottom-right `2x2` block replaced by the target matrix, and
comparing against the named gates:

```
$ PYTHONPATH=$PWD python /tmp/probe067f.py
one control equals cx: True [0.0, 0.0, 0.0, 1.0]
two controls equal ccx: True [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
three controls P(1111): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
target-first wrong rule: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
margin between the two conventions: 2.0
```

One control reproduces `cx` and two controls reproduce `ccx` exactly, at
`complex128`, with no tolerance involved. Three controls have no named
counterpart in the table and are produced by the same rule. The wrong
convention, which puts the target wire first, puts the population in `|110>`
instead of `|111>`: the two conventions differ by `2.0` in total variation, so a
tolerance chosen for this fixture is evidence rather than decoration.

### 4. A user matrix reaches the IR and is refused by every target, and there is
no registration entry point

The local path for a user-defined operation exists and works; only the target
boundary is closed. Measured end to end, with a registry that has the new schema
and no capability:

```
$ PYTHONPATH=$PWD python /tmp/probe067c.py
TABLE_TYPE: mappingproxy | IS_MAPPINGPROXY: True
MODULE_TABLE_HAS_MY_OP: False
NEW_REGISTRY_HAS_MY_OP: True
DEFAULT_REGISTRY_HAS_MY_OP: False
DEFAULT_MANIFEST_QASM_SUPPORTED_HAS_MY_OP: False
LOWERING_WITHOUT_CAPABILITY[pytorch]: backend 'pytorch' cannot lower: my_op (not registered)
LOWERING_WITHOUT_CAPABILITY[qasm]: backend 'qasm' cannot lower: my_op (not registered)
LOWERING_WITH_CAPABILITY[pytorch]: accepted
LOWERING_WITH_CAPABILITY[qasm]: backend 'qasm' cannot lower: my_op (not registered)
SIMULATED_P0: 0.5000000000000001
JSON_ROUND_TRIP_OPCODE: my_op | MATRIX_TYPE: Tensor | JSON_LENGTH: 729
```

Three things are measured here at once. The instruction is accepted by the IR
when it carries a matrix, simulated correctly (`my_op` is `h`, and `P(0)` is
`0.5`), and survives the JSON boundary in both directions. Registering the
schema alone changes nothing for any backend, and a capability for `pytorch`
changes nothing for `qasm`. That is the degradation contract the gap row asks
for, already half-built and entirely unexposed.

`Instruction.metadata` is the one field on the instruction that is free-form,
and it already crosses the same boundary unchanged, so it is the only place a
modifier could be recorded without changing a serialized schema:

```
$ PYTHONPATH=$PWD python /tmp/probe067h.py
METADATA_ROUND_TRIP: {'controls': (1, 2)}
METADATA_KEYS_ACCEPTED: ['controls']
PLAIN_METADATA: {}
```

An instruction with no metadata serializes an empty mapping, so a payload
written before a reserved key existed still reads back exactly as it was
written.

What is missing is an entry point. `flagquantum/operators.py` is five lines and
exports two names:

```python
"""Stable operator schema and discovery interface."""

from .core.operator_schema import GateInfo, gate_info

__all__ = ("GateInfo", "gate_info")
```

```
$ grep -rn "register_operation\|register_custom\|register_operator\|OPERATOR_REGISTRY" --include=*.py flagquantum/ tools/ tests/ ; echo "exit=$?"
exit=1
$ grep -rn "def register" --include=*.py flagquantum/
flagquantum/runtime/noise_registry.py:61:def register_noise_executor(
flagquantum/runtime/backend_registry.py:238:def register_backend(
flagquantum/benchmarking/registry.py:184:def register(
```

The three `register` functions in the package register noise executors, backends
and benchmark kernels; none of them registers an operator, and
`flagquantum/operators.py` contains no `register` symbol at all.

### 5. Registration is already representable as an immutable value, and in-place
mutation is impossible

`OperatorLoweringRegistry` in `flagquantum/compiler/operator_lowering.py:40` is a
frozen dataclass whose two mappings are wrapped in `MappingProxyType` at
construction, and both `with_operator` (line 54) and `with_capability` (line 62)
return `dataclasses.replace` results.

```
$ PYTHONPATH=$PWD python /tmp/probe067c.py
TABLE_TYPE: mappingproxy | IS_MAPPINGPROXY: True
ITEM_ASSIGNMENT: 'mappingproxy' object does not support item assignment
MODULE_TABLE_HAS_MY_OP: False
NEW_REGISTRY_HAS_MY_OP: True
RE_REGISTER: operator schema 'h' is already registered
```

and the manifest of a registry that has the new schema shows the new opcode in
every backend's `unsupported` list, which is the fail-closed default rather than
an omission:

```
$ PYTHONPATH=$PWD python /tmp/probe067i.py
jax     | supported has my_op: False | unsupported has my_op: True | n_supported: 31 | n_unsupported: 5
mps     | supported has my_op: False | unsupported has my_op: True | n_supported: 31 | n_unsupported: 5
pytorch | supported has my_op: False | unsupported has my_op: True | n_supported: 35 | n_unsupported: 1
qcis    | supported has my_op: False | unsupported has my_op: True | n_supported: 26 | n_unsupported: 10
total capability entries: 280
```

`BACKENDS` at line 13 is the closed tuple `('pytorch', 'jax', 'mps',
'tensor_network', 'qasm', 'qcis', 'qir', 'provider')`, `35 * 8 = 280`, and a
capability for an unknown backend or an unknown operator is refused:

```
$ PYTHONPATH=$PWD python /tmp/probe067i.py
unknown backend: ValueError - unknown lowering backend 'bogus'
unknown opcode: ValueError - unknown operator schema 'nope'
```

So the representation this proposal needs already exists and is already the only
one that cannot corrupt the shared table. The gap is that it lives in the
compiler layer while the schema it registers against lives in Core, and
`flagquantum/core/AGENTS.md` states the direction of that dependency:

> Core owns backend-neutral semantics and versioned domain types. It must not
> import Runtime, Compiler implementations, Simulation, Provider SDKs, ecosystem
> frameworks, or gateways.

Measured, Core obeys it today, and Core importing the compiler would also be
visible in the lazy import budget that `tools/check_import_time.py` protects:

```
$ grep -rn "from ..compiler\|from flagquantum.compiler" --include=*.py flagquantum/core/ ; echo "exit=$?"
exit=1
$ PYTHONPATH=$PWD python tools/check_import_time.py
flagquantum import fastest=0.001335s median=0.002066s slowest=0.003231s over 5 runs
$ PYTHONPATH=$PWD python tools/check_architecture.py
architecture boundaries passed
```

`import flagquantum.operators` currently pulls in no compiler module at all
(11 modules loaded, zero of them under `flagquantum.compiler`), so a registration
entry point that reached into `flagquantum.compiler` would be the first thing to
put the compiler on that path.

### 6. Every text exit already fails closed, and the capability document is
derived from two tables

The three text emitters share one guard. `require_static_gate_program`
(`flagquantum/compiler/operator_lowering.py:244`) is called from `_validated_ir`
in `openqasm.py:150`, and the equivalent paths in `qcis.py` and `qir.py`, after
`validate_lowering`. A registered opcode with no capability is refused by name,
and an opcode whose capability is explicitly unsupported is refused with the
capability's own reason:

```
$ PYTHONPATH=$PWD python /tmp/probe067j.py
UnsupportedLoweringError - backend 'qcis' cannot lower: cswap (no QCIS decomposition)
capability: {'backend': 'qcis', 'opcode': 'cswap', 'strategy': 'decomposition', 'implementation': 'flagquantum.compiler.qcis._decompose', 'supported': False, 'reason': 'no QCIS decomposition'}
```

Across the eight backends the 280 capabilities use exactly five strategies, so a
registered operator has a fixed vocabulary to declare itself in:

```
$ PYTHONPATH=$PWD python /tmp/probe067j.py
Counter({'native': 124, 'serialization': 70, 'decomposition': 35, 'lowering': 35, 'kraus': 16})
```

The capability document is generated, and the chain is three files long.

```
$ cat docs/generated/OPERATOR_CAPABILITIES.md | head -6
# Generated Operator Capabilities

Do not edit. Source: `docs/operator_manifest.json`; `yes` means an executable registered lowering, not release or scalability evidence.

| Operator | jax | mps | provider | pytorch | qasm | qcis | qir | tensor_network |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

`tools/operator_manifest.py:17` builds `docs/operator_manifest.json` from
`operator_manifest()` and `DEFAULT_LOWERING_REGISTRY.manifest()["backends"]`, and
`tools/docs_source_of_truth.py:95` renders the markdown from that JSON. All the
guards are green on this checkout:

```
$ PYTHONPATH=$PWD python tools/operator_manifest.py --check       -> exit=0
$ PYTHONPATH=$PWD python tools/docs_source_of_truth.py --check    -> exit=0
$ PYTHONPATH=$PWD python tools/parity_matrix.py --check           -> exit=0 :: parity-matrix: contract valid and document current
$ PYTHONPATH=$PWD python tools/check_capability_maturity.py       -> exit=0 :: capability maturity matrix passed
```

The two gap rows this proposal answers are already in the parity contract, at
`contracts/cudaq-parity-matrix.toml:187` (`control_adjoint_modifiers`, status
`unsupported`, priority `now`) and line 213 (`custom_operation_registration`,
status `partial`, priority `now`), and both are rendered into
`docs/reference/CUDAQ_PARITY_MATRIX.md` at lines 220 and 223. The reason text at
line 192 is the honest description of the state measured above: "Only the named
gates ccx and cswap provide multi-controlled forms, and no adjoint modifier
exists. A unitary passed through Circuit.any is not a modifier."

### 7. The adjoint rewrite a modifier would have to implement is not "negate
every parameter"

Adjoints were measured per operator as `|<seed|second then first|seed>|` on the
state vector, at full precision rather than rounded, with `h(0)` as the seed so
that the fixture is sensitive to a relative phase.

```
$ PYTHONPATH=$PWD python /tmp/probe067l.py
u3(t,p,l) then u3(-t,-p,-l)  0.99952259975704150
u3(t,p,l) then u3(-t,-l,-p)  0.99999999999999978
u2(p,l) then u2(-p,-l)       0.19789601078199381
u2(p,l) then u2(-l,-p)       0.50553334120484683
rx(t) then rx(-t)            1.00000000000000000
s then s                     0.00000000000000000
s then sdg                   0.99999999999999978
t then t                     0.70710678118654746
t then tdg                   0.99999999999999978
```

Negating every parameter is correct for `rx` and wrong for `u3`, whose adjoint
requires the `phi` and `lbd` arguments to be exchanged as well as negated: the
naive rule closes to `0.99952259975704150`, which is twenty-one orders of
magnitude away from the declared rule's `0.99999999999999978`. `u2` is worse than
wrong: neither the identity-orientation swap nor the reversed-orientation guess
reproduces the adjoint (`0.19789601078199381` and `0.50553334120484683`), which
means a modifier that handled `u2` by rule would be guessing. `s` and `t` are not
their own inverses (`0.00000000000000000` and `0.70710678118654746`) and are
exact against the named `sdg` and `tdg`, which is why the rewrite table is keyed
by name rather than derived from an arithmetic rule.

The seed matters, and this was measured rather than assumed. Run again from a
computational-basis seed, `s` and `t` become indistinguishable from their
inverses:

```
$ PYTHONPATH=$PWD python /tmp/probe067k.py
SEED=superposition |s> then s   : 0.0
SEED=superposition |s> then sdg : 0.9999999999999998
SEED=superposition |t> then t   : 0.7071067811865475
SEED=superposition |t> then tdg : 0.9999999999999998
SEED=basis         |s> then s   : 1.0
SEED=basis         |s> then sdg : 1.0
```

A diagonal gate applied to a basis state only moves a phase that `|overlap|`
cannot see, so a fixture built from basis-state seeds would report `1.0` for a
wrong adjoint rule. Every adjoint acceptance item below therefore requires a
seed that is not a computational basis state, and this measurement is the reason.

Composed over a mixed circuit at both dtypes, where the seed is
`h(0); t(1); sdg(2); cx(0,1); ccx(0,1,2); rx(0, theta=0.37);
u3(1, theta=0.37, phi=0.21, lbd=-0.53); rz(2, theta=-0.53)`:

```
$ PYTHONPATH=$PWD python /tmp/probe067e.py
complex64  declared  P(000)=1.00000023841857910 residual=2.384e-07
complex128 declared  P(000)=0.99999999999999933 residual=6.661e-16
negate-all (no per-operator rule)  P(000)=0.98237978262481229 margin=1.762e-02
reverse but do not conjugate       P(000)=0.32180839390072108 margin=6.782e-01
conjugate but do not reverse       P(000)=0.23824580456649552 margin=7.618e-01
```

The declared per-operator rule closes to `6.661e-16` at `complex128` and
`2.384e-07` at `complex64`, while the three plausible wrong rules miss by
`1.762e-02`, `6.782e-01` and `7.618e-01`. A tolerance of `1e-12` at
`complex128` sits three orders of magnitude above the measured residual and ten
orders of magnitude below the nearest wrong rule, so it is evidence rather than
decoration. The same `1e-12` cannot be used at `complex64`, where the measured
residual is `2.384e-07`; the acceptance item below therefore sets `1e-6` there,
which is still four orders below the nearest wrong rule.
The naive rewrite that ignores the per-operator rule is the *closest* competitor
at `1.762e-02`, and it is the `u3` instruction in the fixture that separates it.
Removing that one instruction makes the two rules indistinguishable, which is
measured rather than argued:

```
$ PYTHONPATH=$PWD python /tmp/probe067m.py
with_u3=True  declared   P(000)=0.99999999999999933 margin=6.661e-16
with_u3=True  negate-all P(000)=0.98237978262481229 margin=1.762e-02
with_u3=False declared   P(000)=0.99999999999999933 margin=6.661e-16
with_u3=False negate-all P(000)=0.99999999999999933 margin=6.661e-16
```

So the acceptance item below names a fixture containing a parameterised
non-diagonal gate with a non-basis seed, and does not accept a fixture built only
from self-inverse and diagonal gates.

### 8. What CUDA-Q does, and what must not be inherited

The counterparts were read from the CUDA-Q sources rather than from its user
guide, because the guide pages are not retrievable and a claim about them cannot
be measured.

```
$ curl -s -o /dev/null -w "%{http_code}" https://nvidia.github.io/cuda-quantum/latest/using/cudaq/control.html
404
$ curl -s -o /dev/null -w "%{http_code}" https://nvidia.github.io/cuda-quantum/latest/using/cudaq/adjoint.html
404
$ curl -s -o /dev/null -w "%{http_code}" https://nvidia.github.io/cuda-quantum/latest/using/cudaq/platform.html
200
```

The `using/cudaq/` tree contains only `platform/`, so the control and adjoint
surface is established from `cudaq/test/Frontend/*.cpp`,
`runtime/cudaq/qis/{qubit_qis.h,modifiers.h}` and
`python/cudaq/kernel/register_op.py` instead, and the unretrieved pages are
recorded as unverified rather than asserted.

Three properties of the CUDA-Q surface are load-bearing here, and all three are
about scope rather than syntax. A modifier applies to a kernel call region, a
template argument, or a registered operation, never to a plain named gate: the
AST bridge rejects both built-in gates and globally registered operations for
`cudaq.control` and `cudaq.adjoint` with dedicated messages. Registration refuses
parameterised operations outright:

```
$ grep -n "not yet supported\|not provided\|unknown type of unitary\|invalid matrix size" /tmp/regop.py
36:        raise RuntimeError("custom operation name not provided.")
39:        raise RuntimeError("parameterized custom operations not yet supported.")
46:        raise RuntimeError("unknown type of unitary.")
54:            "invalid matrix size, required 2^N * 2^N for N-qubit operation.")
```

and it attaches the operation to the kernel class rather than to a global table.
Negative controls (`{q, !t}`) are a real part of that surface and expand to
`x`-bracketing around the controlled region.

What this proposal must not inherit is therefore concrete: the kernel-region
modifier semantics, because FlagQuantum's unit of composition is a `Circuit` and
a callable region has no representation in `CircuitIR`; the refusal of
parameterised registered operations, because FlagQuantum's schema table already
carries `parameters` and the IR already validates them; and negative controls,
which are a separate contract with a separate expansion rule. Those are listed
under "Non-goals".

## Decision

### 1. A modifier is recorded on the instruction, not in the opcode name

The two candidate representations are an instruction-level modifier and a family
of generated opcode names (`mcx`, `cccx`, `h_dg`, ...). The second is refused.
`flagquantum/core/ir.py:349` canonicalises every instruction name through
`canonical_opcode` and `flagquantum/core/ir.py:364` refuses an unknown name
unless the instruction carries an explicit matrix, so a name-per-modifier design
would have to add a permanent, unbounded set of names to a table the rest of the
system already treats as closed, and each added name would create a new
`OPERATOR_SCHEMAS` entry that every backend must then be given a capability for.
The first representation is added: an instruction may carry a modifier record in
its existing `metadata` mapping, under the reserved keys `modifiers` and
`controls`. `Instruction.metadata` already round-trips through `to_dict`,
`to_json` and `from_dict` unchanged, as measured in section 4 of "Problem", and
an instruction with no metadata still serializes an empty mapping.

The reserved keys are `metadata["modifiers"]`, a tuple drawn from `("adjoint",)`,
and `metadata["controls"]`, a non-negative integer count. Nothing else is
reserved, and an instruction with neither key behaves exactly as it does today.
This is a representation decision, not an implementation: no field is added to
`Instruction`, and `IR_VERSION` stays `1.0`.

### 2. Control wires are a prefix; the count is the wire tuple length minus the
schema arity

An instruction that carries `controls = k` must carry exactly `arity + k` wires,
and the first `k` of them are the controls in the order given. This is not a new
convention invented here: it is the convention the two existing multi-control
opcodes already use, and their `wire_convention` is measured above as
`('control_0', 'control_1_or_target_0', 'target_or_target_1')`, which is
controls-first. The competing convention, targets first, is refused rather than
guessed, and section 3 of "Problem" measures the two apart by `2.0` in total
variation on a fixture where the correct convention is exact.

A control count of `0` is refused rather than treated as a no-op, because the
unmodified instruction is the correct spelling for zero controls and two
spellings for one meaning is exactly the kind of ambiguity that produced the
silent no-op measured in section 3.

### 3. The control count is unbounded in the modifier and stays bounded in the
schema table

A modifier may carry any number of controls, because the count lives in the wire
tuple and not in the schema. The schema table is not widened: no field is added
to `OperatorSchema`, and the `wire_convention` dictionary in `_unitary` keeps its
three keys. A controlled instruction reuses the schema of its *base* opcode, and
`arity` continues to describe the base operator. This is why the representation
generalises without touching the 35 entries: `ccx` remains the named two-control
`x`, and it is not deprecated or redefined by this change.

Where a backend lowers a controlled instruction it must do so from the count, not
from a name. The local statevector path already has the arithmetic: `identity` of
dimension `2**(k+1)` with the bottom-right `2x2` block replaced by the base
matrix reproduces `cx` for `k = 1` and `ccx` for `k = 2` exactly, as measured in
section 3 of "Problem".

### 4. The adjoint is a per-operator declarative rewrite applied in reverse order

The adjoint of an instruction sequence is the adjoint of each instruction in
reverse order, and the adjoint of an instruction carrying `controls = k` is the
control-carrying adjoint of the base operator, with the control wires unchanged.
The per-operator part is a rewrite table keyed by the base opcode, derived from
`OperatorSchema.adjoint` and, where that field is `matrix_adjoint`, from a
per-opcode rule. Section 7 of "Problem" measures why a single arithmetic rule is
insufficient: `u3` needs `phi` and `lbd` exchanged as well as negated, and `u2`
has no rule that any of the natural guesses reproduces.

Adding the modifier does not make `OperatorSchema.adjoint` a consumer-facing
field, and it does not promote the three-value vocabulary into a guarantee. The
field keeps its single reader at `operator_schema.py:173`; the rewrite table is
the new artefact and it is total over the base opcodes in the sense that every
opcode either has a rule or is refused.

### 5. An operator whose adjoint rule is not measured is refused

The rewrite table is fail-closed by construction. Three cases are refused rather
than approximated:

- A base opcode whose `semantic_kind` is `channel` — measured as exactly
  `amplitude_damping`, `bit_flip`, `depolarizing` and `phase_flip`, all with
  `adjoint = not_applicable`. A channel's adjoint is not a channel in general,
  and this proposal does not introduce a second object for it.
- A base opcode marked `matrix_adjoint` for which no per-opcode rule has been
  measured and recorded. `u2` is in this class today, and it stays refused until
  a rule is measured. The measured guesses for it are not close: the
  identity-orientation negation closes to `0.19789601078199381` and the
  reversed-orientation negation to `0.50553334120484683`, against `1.0` for a
  correct adjoint, so admitting it by rule would be a guess rather than a
  derivation.
- An instruction whose adjoint would require reversing a sequence the modifier
  cannot see, which in practice means an instruction carrying a caller-supplied
  `matrix`: the adjoint of an opaque matrix is computable, but the *rewrite* is
  not expressible in the base vocabulary, so the modifier refuses and the caller
  supplies the adjoint explicitly if that is what they want.

Each refusal names the opcode it blames. `u2`'s absence from the table is a
recorded limitation, not a silent fallback.

### 6. Registration returns a new frozen table; `OPERATOR_SCHEMAS` is never
mutated

`OPERATOR_SCHEMAS` is a `MappingProxyType` and item assignment on it raises
`TypeError`, as measured. That is not an obstacle to be worked around; it is the
shape the registration contract adopts. A registration is a pure function from a
table to a new table, in the same way `OperatorLoweringRegistry.with_operator`
already is.

Two tables are involved and they live in two layers, because
`flagquantum/core/AGENTS.md` forbids Core importing Compiler implementations. The
schema-only table is Core-owned and new; the lowerings remain
`OperatorLoweringRegistry`, which is already the only place a lowering can be
declared and already refuses a duplicate opcode, an unknown backend and an
unknown opcode. Core does not import the compiler, and the compiler consumes the
Core table. This direction is forced by the boundary rule quoted in section 5 of
"Problem" and is checked by `tools/check_architecture.py`, which passes today.

Refused explicitly:

- Any in-place mutation of `OPERATOR_SCHEMAS`, including via
  `object.__setattr__`, `MappingProxyType` unwrapping, or a module-level
  `dict` that shadows it. Measured to be unnecessary: every registration the
  contract allows is expressible as a new frozen value.
- A process-global mutable registry, because it would make
  `tools/operator_manifest.py --check` depend on import order and on which tests
  ran first, and the manifest is a checked-in artefact.
- Redefining an existing opcode. `with_operator` already refuses this with
  `operator schema 'h' is already registered`, and the refusal is kept rather
  than relaxed to an override.

### 7. A registered operator enters the IR only with an explicit matrix

`flagquantum/core/ir.py:364` admits an unknown opcode only when the instruction
carries a matrix, and this proposal keeps that rule rather than exempting
registered operators from it. The reason is that the IR is the serialized
boundary: an `Instruction` whose opcode is known only to a runtime registry
cannot be validated by a reader that does not share that registry, and the
matrix-bearing form is already measured to round-trip through
`CircuitIR.to_json`/`from_json` with the matrix intact.

So registration adds a schema, a lowering vocabulary and an emission strategy. It
does not add an IR-level name that has to be resolved before the program can be
read. `IR_VERSION` stays `1.0`, and no new `metadata` key is required for a
registered operator.

### 8. Backend lowering for a registered operator requires a declared capability

A registered opcode is added to every backend's `unsupported` list with no
capability entry, measured for all eight backends. That is the degradation
contract, and the emitter message is already correct for it:
`backend '<b>' cannot lower: <op> (not registered)`. Registration therefore
changes nothing for a backend until that backend is given a capability with an
explicit `strategy` drawn from the five measured strategies, and a capability
declared `supported=False` keeps the operator out of the emitter with its own
`reason` recorded, exactly as `cswap` behaves towards QCIS today.

The following are refused rather than inferred:

- Inferring support from the presence of a matrix. The local path can execute a
  matrix for any wiring; a text target cannot express it, and inferring text
  support from "the simulation ran" is precisely the failure the shared
  `require_static_gate_program` guard exists to prevent.
- Emitting a registered operator to a text profile as an opaque comment or an
  unchecked native name. A text exit that cannot express the operation refuses
  it by name.
- Adding a ninth backend. `BACKENDS` is a closed tuple of eight, and a
  registration for anything else is rejected with `unknown lowering backend`.

### 9. Registration does not change the root namespace

No name added by this proposal is a root export. `docs/public_api_v1.json` has 34
`stable_exports`, none of whose names is `register_operator`, `OperatorRegistry`,
`control` or `adjoint`, and `flagquantum.__all__` has the same 34 names. The new
names land in `flagquantum.operators`, which is a namespace entry in
`contracts/public-api-v1-candidate.json`, not a root export. Section 7 of
"Compatibility" states exactly what changes in that file and the negative control
that establishes it.

The modifier itself is added as methods on `Circuit` rather than as root
functions. `Circuit` is already a Stable Core name, and its method set is not
part of the frozen root manifest, so adding a method widens a class rather than
adding a root name. `flagquantum/algorithms/core.py:191` already uses the word
`adjoint` for a *gradient* option; the modifier is therefore spelled with names
that cannot be confused with it, and this proposal does not touch that option.

### 10. The capability document is regenerated, never hand-edited

`docs/generated/OPERATOR_CAPABILITIES.md` is derived from
`docs/operator_manifest.json`, which is derived from `operator_manifest()` and
`DEFAULT_LOWERING_REGISTRY.manifest()["backends"]`. It says so on line 3, and
`tools/operator_manifest.py:37` refuses to let it go stale. This proposal
therefore regenerates it by running the two generators, and refuses to edit it
directly; a hand edit is a defect, not a shortcut.

The document's *content* changes only if a built-in table changes. Adding a
schema to a copy of the registry does not touch `DEFAULT_LOWERING_REGISTRY`, and
`with_operator` on the default registry returns a new object, so a user
registration cannot move a cell in the checked-in table. If the implementation
represents controls and adjoints as modifiers on existing opcodes, as this
proposal decides, then `ccx` and `cswap` remain the only arity-3 built-ins and
the regenerated file is byte-identical to the current one. The acceptance item
below therefore requires the measured diff and does not assume it is empty: if
the implementation does add a built-in, the diff must be exactly the rows that
addition implies, and if it does not, the two `--check` commands must still pass
with no diff. Claiming "regenerated" without measuring the diff would make the
plan row's third deliverable unverifiable.

## Public API

One new namespace member set, two new `Circuit` methods, and one new Core table
type. No Stable Core root export is added, removed or renamed, so
`docs/public_api_v1.json` is unchanged by this proposal: its 34 `stable_exports`
do not contain any name introduced here, and its `experimental_stability` stays
`"none"`.

```python
from flagquantum.operators import OperatorRegistry, register_operator

registry = register_operator(
    "my_op",
    arity=1,
    parameters=(),
    aliases=(),
    adjoint="self_inverse",
    wire_convention=("target",),
)
registry.schemas["my_op"].opcode  # 'my_op'; registry.schemas is immutable
```

```python
from flagquantum import Circuit

circuit = Circuit(4)
circuit.control("x", controls=(0, 1, 2), wires=(3,))   # three controls
circuit.adjoint("u3", wires=(0,), params={"theta": 0.37, "phi": 0.21, "lbd": -0.53})
```

`register_operator` returns a new `OperatorRegistry`; it never mutates the
argument. `OperatorRegistry.schemas` is an immutable mapping and
`OperatorRegistry.with_operator` mirrors the existing registry method of the same
name. The lowerings stay where they are:
`flagquantum.compiler.operator_lowering.OperatorLoweringRegistry.with_capability`
is the only way to declare that a backend can lower a registered operator, and
`DEFAULT_LOWERING_REGISTRY` is not reachable from `flagquantum.operators`.

`docs/public_api_v1.json` does not change. `contracts/public-api-v1-candidate.json`
does change, in exactly two places, and the measurement that establishes both is
in "Compatibility". The new public root name count is zero.

## Compatibility

- **`docs/public_api_v1.json` is unchanged.** Its `stable_exports` list has 34
  names and none of them is introduced here, and `tools/public_api_snapshot.py`
  refuses automatic regeneration of that file at all:

  ```
  $ PYTHONPATH=$PWD python tools/public_api_snapshot.py --check
  error: unrecognized arguments: --check
  ```

  `--write` is the only flag it accepts and it hard-errors with `automatic API
  contract regeneration is disabled`. The file can therefore only move through an
  approved additive contract, which this proposal is not.

- **`contracts/public-api-v1-candidate.json` changes in two places**, and both
  are required. `flagquantum.operators` is already a `stable_extensions` entry
  with `symbols = ["GateInfo", "gate_info"]`. The new namespace members must be
  added to that entry's `symbols` list *and* to
  `approved_namespace_additions`. Each edit alone fails a different test, which
  was measured as a negative control by patching the file, running the two test
  modules, and restoring it:

  ```
  $ PYTHONPATH=$PWD python -m pytest tests/unit/test_public_api_candidate.py tests/unit/test_api_namespace_convergence.py -q
  # symbols only:
  E   AssertionError: flagquantum.operators is missing candidate exports: ['register_operator']
  FAILED tests/unit/test_public_api_candidate.py::test_candidate_classifies_every_historical_stable_export_exactly_once
  FAILED tests/unit/test_api_namespace_convergence.py::test_every_stable_migration_destination_is_importable
  2 failed, 16 passed

  # approved_namespace_additions only:
  E     Extra items in the right set: 'register_operator'
  FAILED tests/unit/test_public_api_candidate.py::test_candidate_classifies_every_historical_stable_export_exactly_once
  1 failed, 17 passed

  # both:
  E   AssertionError: flagquantum.operators is missing candidate exports: ['register_operator']
  FAILED tests/unit/test_api_namespace_convergence.py::test_every_stable_migration_destination_is_importable
  1 failed, 17 passed
  ```

  The third case is the shape the implementation must reach: the only remaining
  failure is that the symbol is not yet importable, which is what the
  implementation supplies. So the candidate contract is not "unchanged" and this
  proposal does not claim it is.

- **The existing silent no-op changes behaviour, deliberately.** A caller who
  passes `controls=` as a keyword to `Circuit.gate` or to a generated gate method
  today gets a parameter that nothing reads. After this change that spelling is
  either wired to the modifier or refused. Either way it is a behaviour change
  and it is the one genuine migration in this proposal: the old spelling silently
  did nothing, so a caller who depended on the old behaviour depended on a bug.
  The migration is recorded here rather than smoothed over.

- **`metadata` is widened by convention, not by type.** `Instruction.metadata`
  is already a free-form mapping and already round-trips, so an instruction
  written before this change deserializes unchanged. The two reserved keys are a
  new interpretation of existing data; a payload that already used
  `metadata["controls"]` for something else would be reinterpreted, which is why
  the acceptance evidence requires a search for existing writers of that key.

- **A channel instruction refuses an adjoint modifier** where it previously had
  no modifier to refuse. Nothing that ran before stops running; something that
  could not be expressed before is now expressible and is refused when it is
  meaningless.

- **No Stable Core root name changes, so no deprecation window is owed**, and no
  maturity level is raised by this proposal. `tools/check_capability_maturity.py`
  passes today and must still pass.

## Required evidence before this proposal can be accepted

Each item names the measurement, not the intention. A green suite does not show
that a refusal is exercised, so each refusal is additionally proven by deleting
it and observing a test fail.

- [ ] The mixed fixture of "Problem" section 7 is reproduced through the new
      modifier in one call, at `dtype=torch.complex128` and
      `dtype=torch.complex64`, and `P(000)` is reported at both. At
      `complex128` the value is within `1e-12` of `1.0`: the measured residual of
      the declared rule is `6.661e-16`, and the nearest wrong rule misses by
      `1.762e-02`, so the tolerance sits three orders above the residual and ten
      below the nearest wrong rule. At `complex64` the tolerance is `1e-6`, not
      `1e-12`: the measured residual there is `2.384e-07`, which is above `1e-12`
      and still four orders below the nearest wrong rule. The two tolerances are
      different because the measured residuals are different, and a single
      `1e-12` tolerance claimed for both dtypes would contradict the measurement.
- [ ] All three wrong rules from the same table (`negate-all`,
      `reverse-but-do-not-conjugate`, `conjugate-but-do-not-reverse`) are shown to
      fail that tolerance on the same fixture, with the three measured values
      reported, so the tolerance is evidence rather than decoration. The fixture
      is shown to be discriminating by the control measurement in "Problem"
      section 7, where removing the single `u3` instruction collapses the
      `negate-all` margin from `1.762e-02` to the declared rule's own `6.661e-16`.
- [ ] The per-operator adjoint fixture of "Problem" section 7 is reproduced
      through the modifier, with a seed that is not a computational basis state,
      and every measured pair is reported. The measurement in that section shows
      a basis-state seed reports `1.0` for both `s then s` and `s then sdg`, so a
      basis-seed fixture cannot support the claim and is not accepted as evidence.
- [ ] The rewrite table is shown total over the base operators: enumerating the
      35 schemas, every `self_inverse` opcode and every `matrix_adjoint` opcode
      with a recorded rule rewrites to an instruction the IR accepts, and every
      remaining opcode is refused by name. The refusal list is printed, and it
      contains all four `channel` opcodes and `u2`.
- [ ] `u2`'s adjoint is either given a measured rule with the fixture that
      establishes it, or is listed in the refusal report with the two measured
      guesses that do not work (`0.19789601078199381` for `u2(-phi,-lbd)` and
      `0.50553334120484683` for `u2(-lbd,-phi)` on the section 7 fixture). The
      item is satisfied by the record, not by forcing a rule.
- [ ] A `k`-control instruction reproduces the named gate exactly for `k = 1`
      (`cx`) and `k = 2` (`ccx`) at `complex128`, compared gate-for-gate on the
      same state, with both probability vectors reported; and a `k = 3` case is
      compared against a dense reference built independently of the modifier,
      with the reference route stated.
- [ ] The control-prefix convention is shown to be the implemented one by
      measuring the target-first variant against it and reporting the margin; the
      measured margin on the fixture used in "Problem" section 3 is `2.0`, so a
      tolerance for this comparison is evidence rather than decoration.
- [ ] A controlled instruction with `controls = 0` is refused by name, and an
      instruction whose wire count is not `arity + controls` is refused by the
      existing arity message or by a message that names the count rule.
- [ ] The four `channel` opcodes refuse the adjoint modifier, each with its
      opcode in the message, and the refusal is proven by a mutation that deletes
      it and fails a test.
- [ ] A registered operator with no capability is refused by all eight backends
      with the message `backend '<b>' cannot lower: <op> (not registered)`, and
      after a capability is declared for exactly one backend, that backend
      accepts and the other seven still refuse. The eight outcomes are reported.
- [ ] A capability declared `supported=False` with a `reason` keeps the operator
      out of the emitter and the reason appears in the message, as `cswap`
      towards QCIS does today, with both messages reported side by side.
- [ ] After a registration, `OPERATOR_SCHEMAS` is still a `MappingProxyType`
      with 35 entries, item assignment still raises `TypeError`, and
      `len(DEFAULT_LOWERING_REGISTRY.schemas)` is still 35. The registration is
      also shown to be unreachable from `DEFAULT_LOWERING_REGISTRY.manifest()`,
      reported as the measured `supported`/`unsupported` counts per backend.
- [ ] `docs/generated/OPERATOR_CAPABILITIES.md`'s regenerated diff is measured
      and reported. If the diff is empty, the two `--check` commands exit `0`
      with no output; if it is not empty, the added or changed rows are listed
      and each is traceable to a change in `operator_manifest()` or in the
      default registry's capabilities.
- [ ] `docs/public_api_v1.json` is byte-identical before and after, verified by
      hash, and `flagquantum.__all__` still has 34 names.
- [ ] `contracts/public-api-v1-candidate.json` differs from the current file in
      exactly the two places described in "Compatibility", and
      `tests/unit/test_public_api_candidate.py` and
      `tests/unit/test_api_namespace_convergence.py` both pass with no failures,
      including the case that currently fails only because the symbol does not
      exist yet.
- [ ] `python tools/check_import_time.py` passes, and `import
      flagquantum.operators` still loads zero modules under `flagquantum.compiler`.
      Both are measured, and the import figures are reported.
- [ ] `python tools/check_architecture.py` passes, proving Core did not import
      Compiler to reach the lowering registry.
- [ ] `python tools/check_capability_maturity.py` passes unchanged, and no
      maturity level is raised by this change.
- [ ] The parity rows `control_adjoint_modifiers` and
      `custom_operation_registration` in `contracts/cudaq-parity-matrix.toml` are
      updated only as far as the measurements above support, their rendered
      entries in `docs/reference/CUDA_PARITY_MATRIX.md` are regenerated rather
      than hand-edited, and `python tools/parity_matrix.py --check` exits `0`.
      The status values written there are the ones the recorded evidence
      supports, and the document that records them is not the place to argue for
      a level.
- [ ] `python tools/check_repository_language.py` reports `0 files contain
      Han-script text`.
- [ ] Repository owner approves. Not yet requested; see `## Status`.

## Non-goals

- Kernel-region modifiers. CUDA-Q applies `cudaq.control` and `cudaq.adjoint` to
  a callable region, and the measured CUDA-Q surface rejects both built-in gates
  and globally registered operations for those two functions. FlagQuantum's unit
  of composition is a `Circuit` and it has no callable-region node in
  `CircuitIR`, so inheriting the region form would mean adding a new IR concept
  rather than extending the existing one.
- Negative controls. `{q, !t}` in CUDA-Q expands to `x`-bracketing around the
  controlled region, which is a second expansion rule with its own correctness
  argument. It is not part of this contract.
- Parameterised registered operations are admitted here, unlike CUDA-Q, which
  refuses them with `parameterized custom operations not yet supported.` The
  schema table already carries `parameters` and the IR already validates them, so
  refusing them would discard a capability that exists; the difference is
  recorded rather than copied.
- A `control`/`adjoint` power modifier, a general `pow` or `__pow__`, a
  `conjugate` modifier, or a symbolic modifier algebra that can be composed
  before it is applied. The measured `Circuit` surface has none of these and this
  proposal adds only the two that W5-11 names.
- Raising any maturity level, extending the stable root namespace, or adding a
  new backend to `BACKENDS`.
- Emitting a registered operator to a text profile that cannot express it. The
  refusal is the contract.
- A mutable global operator registry, a registration that can override an
  existing opcode, and any form of reloading the manifest at import time.
- The adjoint differentiation option in `flagquantum/algorithms/core.py:191` and
  the reverse-mode sweepers. Those are the plan row W2-03 and
  `API_CHANGE_PROPOSAL_068_SV_GRADIENT_CONTRACT.md`. This proposal reuses the
  word and does not touch the machinery.
- Fermion and boson operator algebra. That is W6-01, which reads this document as
  a dependency; it extends the vocabulary fixed here and is not this proposal's
  deliverable.
