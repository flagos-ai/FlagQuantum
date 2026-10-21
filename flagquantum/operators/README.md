# Operators

This package owns the stable operator schema and discovery interface
(`GateInfo`, `gate_info`), the registration table (`OperatorRegistry`,
`register_operator`), and the superoperator algebra (`SuperOperator`). A
superoperator is a sum of left and right multiplication actions on a matrix; the
algebra assembles, inspects, applies, and densifies such a sum. It does not
execute a numerical scheme, choose resources, or adapt providers, and it must not
import `flagquantum.simulation`.

## Registration

`register_operator` declares an operator that is not built in. It returns a new
`OperatorRegistry` and never writes into the built-in
`flagquantum.core.operator_schema.OPERATOR_SCHEMAS`, which is a read-only mapping:

```python
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.operators import register_operator

registry = register_operator(
    "cphase_quarter", arity=2, adjoint="self_inverse",
    qubit_convention=("control", "target"),
)
print(registry.schemas["cphase_quarter"].arity)  # 2
print("cphase_quarter" in OPERATOR_SCHEMAS)      # False
```

Two consequences are contracts rather than omissions, and both have a test:

- **A registration does not make an opcode known to the IR.** `Instruction`
  resolves names through the built-in table, so an instruction naming a
  registered opcode is still an unknown opcode and still carries its own matrix.
  That is deliberate: the IR is the serialized boundary, and an instruction whose
  name only a runtime table knows cannot be validated by a reader that does not
  hold that table. The registration is a declaration; the matrix is the operand.
- **A registration buys nothing at a backend until that backend is given a
  capability.** `register_operator` touches no `OperatorLoweringRegistry`, so a
  fresh opcode is reported unsupported by all eight backends with
  `backend 'qasm' cannot lower: cphase_quarter (not registered)`. Declaring
  `with_capability(...)` is the only way in, and `supported=False` keeps the
  operator out with its own `reason`. Inferring support from "the simulation ran"
  is forbidden: a text target cannot express a dense matrix, which is what
  `require_static_gate_program` exists to prevent.

The lowerings stay in `flagquantum.compiler.operator_lowering`. This package must
not import it: Core owns the schema table, Compiler consumes it, and
`tools/check_architecture.py` enforces the direction.

Run the focused tests with:

```bash
pytest tests/unit/test_operator_registry.py tests/unit/test_operators_superoperator.py
```

## Superoperators

Start with the Lindblad dissipator, which is one right-apply term and two
one-sided terms:

```python
import torch

from flagquantum.operators import SuperOperator

lowering = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
number = lowering.conj().T @ lowering
dissipator = SuperOperator.left_right_multiply(lowering, lowering.conj().T)
dissipator += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
dissipator += SuperOperator.right_multiply((-0.5 + 0.0j) * number)
print(dissipator.dense().shape)  # torch.Size([4, 4])
```

A factor is a matrix operator: it provides `dtype`, `dimension` (`shape[0]` for a
`torch.Tensor`), `__matmul__` and `__rmatmul__` with a square state, and a
`dense()` matrix. Add a new factor implementation only when it can satisfy that
whole contract, and only together with a test that applies it in a term. A
non-tensor factor additionally needs `adjoint()` if it is ever adjointed, and
only then: `PauliSum` supplies it so a matrix-free `H` can be reversed, and a
factor that supplies it is checked against `dense().conj().T`, not by reading its
own `adjoint` back.

`SuperOperator.adjoint()` is the Hilbert--Schmidt adjoint of the sum: every term
conjugates its coefficient and adjoints each factor on the side that factor
already occupied, because the adjoint of `rho -> A rho B` is
`sigma -> A^dag sigma B^dag`. The sides do not swap, and a one-sided term stays
on its own side. The result satisfies `<sigma, L rho> = <L^dag sigma, rho>` and
has `adjoint()` as its own dense conjugate transpose; it is the map that carries a
cost gradient backwards through a generator, which is why `Liouvillian` derives
its reverse map from `adjoint()` instead of assembling one beside the generator.

The dense form uses the repository-wide row-major vectorization
`vec(A rho B) = (A (x) B^T) vec(rho)`, so the right factor is transposed and never
conjugate-transposed. Keep that convention if you add a second dense view.
