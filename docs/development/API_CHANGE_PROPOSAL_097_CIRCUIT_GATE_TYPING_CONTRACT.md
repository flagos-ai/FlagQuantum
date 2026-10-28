# API Change 097: Circuit Gate Typing Contract

Status: approved and implemented on the proposal branch
Approval directive: `Systematically solve this for both Circuit and DynamicCircuit.`
Approved scope: static typing for the existing schema-installed gate surface; no
runtime gate, signature, execution, provider, or capability change

## User outcome

The normal fluent API now retains its concrete builder type under static checking:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

static = fq.Circuit(3).h(0).cx(0, 1).rx(2, 0.25)
dynamic = DynamicCircuit(3).h(0).cx(0, 1).measure(1)
```

Type checkers resolve `static` as `Circuit` and `dynamic` as `DynamicCircuit`.
They also recognize every built-in opcode, public alias, and uppercase spelling
that the existing `OperatorSchema` installer exposes at runtime.

## Problem

Built-in gate methods are installed after the `Circuit` class body is created.
Python users can call them, but source-based static analysis previously saw no
such attributes. The first repository source file that used
`DynamicCircuit.x(...)` under strict mypy therefore failed even though the same
call executed correctly.

Changing user code to `circuit.gate("x", qubit)` would hide the typing defect and
weaken the documented fluent API. Declaring only `DynamicCircuit.x` would move the
failure to the next gate and duplicate the parent surface.

## Contract

1. `OperatorSchema` remains the only runtime implementation source for built-in
   gates and aliases.
2. During static analysis, `Circuit` inherits a typing base whose companion stub
   declares exactly the spellings installed by the schema registry. At runtime
   that base is `object`, so the class hierarchy and method ownership do not change.
3. Every declared method uses a generic self type. A gate called on `Circuit`
   returns `Circuit`; the same inherited method called on `DynamicCircuit`
   returns `DynamicCircuit`.
4. The declarations retain the runtime installer's flexible `*args` and
   `**kwargs` boundary. This change does not pretend to narrow signatures that
   runtime still accepts.
5. A contract test derives the expected spelling set from `OPERATOR_SCHEMAS` and
   `OPERATOR_ALIASES`, including uppercase spellings, and fails if the stub drifts.
6. The wheel carries both the typing stub and the PEP 561 `py.typed` marker, so
   installed-package users receive the same contract as repository developers.
7. Dynamically registered third-party operators continue to use `gate(...)` for
   static code unless their own package supplies an additional typed facade.

## Non-goals

- changing gate semantics, validation, matrices, aliases, or parameter order;
- promoting a provider or QPU capability;
- making inline noise channels stable for dynamic execution;
- merging `Circuit` and `DynamicCircuit`;
- replacing runtime schema installation with handwritten gate implementations.

## Verification

- strict mypy source checks for both builder chains;
- runtime identity and instruction-order checks;
- exact schema-to-stub spelling coverage;
- regression checks proving `DynamicCircuit` still inherits runtime gate methods
  from `Circuit` rather than carrying a second implementation.
