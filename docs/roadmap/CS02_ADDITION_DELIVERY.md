# CS-02: shared addition kernel and conservative lowering

CS-01 landed in PR #952 at main `8efd0c02f063456c9d893f01f7f3370e52d143c4`.
This delivery follows revision six's private addition milestone. It adds no
public synth namespace, export, schema, dependency or execution API.

## Implementation and ownership

- Compiler owns `_arithmetic_kernels.ripple_add`, extracted from the existing
  Cuccaro implementation. It accepts explicit disjoint MSB-first wire maps and
  returns existing Core Instruction records. It imports no Circuit, Algorithms,
  Runtime or Simulation module.
- Algorithms retains `adder_circuit`, `adder_qubits`, AdderQubits and their exact
  signatures, wire layout, carry contract and exception behavior. Its wrapper
  materializes the shared instructions into Circuit. The old MAJ/UMA templates
  and Circuit-specific generator are removed. The wrapper migration is a
  separate Algorithms-owned commit after the Compiler implementation commit;
  both are reviewed together against CS-01's merged dependency contract.
- `_construction.ranges.unsigned_width` supplies the shared exact inclusive-bound
  width rule, including width one for zero. The existing verifier remains the
  authority for closed operation/type/SSA/call-graph checks and full-width bounds.
- `_construction.lowering.lower_addition` verifies the complete graph and port
  bindings, then handles constants, exact addition, direct acyclic calls and
  returns. Calls use fresh local SSA environments; repeated logical reads use
  separate computational-basis copies rather than duplicate gate controls.

For each addition, both operands are zero-extended into fresh workspace registers.
The shared ripple adder leaves the sum in its second register and carry-out.
All intermediates are retained. The final value is XOR-copied into the supplied
output; the exact reverse of every emitted X/CX/CCX then cleans the workspace.
Known high zero bits may be omitted from an intermediate's physical encoding;
they never narrow declared input/output ports or truncate a result.

For every basis input and target, extended linearly including entangled inputs:

```text
|x,y,z,0_work> -> |x,y,z XOR (x+y),0_work>
```

Fresh assignment is the zero-target case. External zero assertions remain caller
preconditions, not compiler proofs. Workspace exhaustion is reported for this
retain-all schedule; it is not proof that no other implementation fits.

Safety caps are 256 declared wires and 100,000 emitted operations. The operation
budget includes computation, the final XOR copy and inverse computation. Existing
graph budgets apply before emission. Failure returns no partial artifact.
Multiplication has an explicit unsupported-lowering diagnostic. In-place source
addition, truncating XOR, roles/effects, free/drop, source capture, lifetime
optimization and candidate selection remain later milestones. Three-bit 7+1
cannot silently become three-bit wraparound arithmetic.

## Executable private developer example

The structured entry is private. It demonstrates the current implementation,
without presenting a future Python frontend as an available API:

```python
import flagquantum as fq
from flagquantum.compiler._construction.lowering import lower_addition
from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._hybrid.model import (
    Block, HybridProgram, Operation, Region, Value, ValueId,
)

x = Value(ValueId("main", 0), uint_type(2))
y = Value(ValueId("main", 1), uint_type(2))
total = Value(ValueId("main", 2), uint_type(3))
main = HybridProgram("main", Region((Block((x, y), (
    Operation("qint.add", (x, y), (total,)),
    Operation("qint.return", (total,)),
)),)))
block = lower_addition(
    (main,), {x.id: (0, 1), y.id: (2, 3)}, (4, 5, 6),
    entry="main", n_wires=13, workspace=tuple(range(7, 13)),
    workspace_state="fresh_zero",
)
# Prepare x=3, y=1; output and workspace start at zero.
circuit = fq.Circuit(13).x(0).x(1).x(3).compose(block)
options = fq.ExecutionOptions(
    mode="statevector", backend="torch", device="cpu", precision="complex128",
    require_gradients=False, allow_approximate=False, allow_backend_fallback=False,
)
result = fq.run(circuit, options=options)
state = result.statevector().reshape(-1)
expected = (((3 << 2) | 1) << 3 | 4) << 6
assert abs(complex(state[expected]) - 1) < 1e-12
```

Existing users continue using `adder_circuit(n_bits)` without changing their code.
They retain the in-place sum/carry behavior; this private evaluator additionally
preserves both addends and supports a disjoint arbitrary XOR target.

## Acceptance evidence

The raw and default-compiled two-bit example both cost **13 declared qubits,
6 workspace wires, 45 operations (37 CX, 8 CCX), depth 37**, using the existing
Compiler estimator. These are static exact-Core costs, not optimality, capacity
or hardware timing claims.

- Real `fq.run` CPU execution checks all 16 addend pairs and all eight target
  bitstrings: 128 complete statevectors, including all workspace amplitudes.
- A seeded complex input entangled with an external reference checks full
  amplitudes, input/output wire permutations and cleanup through the existing
  CPU Circuit.state path. That path owns arbitrary initial amplitudes;
  gate-level `fq.run` does not consume Circuit's private initial-state buffer.
- Real CPU cases cover unequal widths, maximal carry, inverse composition,
  zero constants, repeated reads, a direct subroutine call and relative phase.
- Structural permutation tests exhaust unequal widths and over-wide targets,
  constants in both operand positions, call composition, clean workspace,
  operation/workspace failures and unsupported source semantics.
- The existing arithmetic suite checks every clean basis input at widths 1-4,
  the whole-space permutation, exact phases, wire layout, gate counts and
  Compiler-derived Clifford+T cost. A replacement probe swaps the wrapper's
  kernel implementation without changing its consumer. Independently captured
  pre/post-extraction IR JSON is identical at widths 1, 2, 3 and 5.

Runnable checks:

```bash
python -m pytest tests/team/compiler/test_construction_addition.py \
  tests/integration/test_construction_addition_execution.py \
  tests/unit/test_algorithms_arithmetic.py \
  tests/team/compiler/test_construction_profile.py \
  tests/team/compiler/test_construction_semantics.py \
  tests/team/compiler/test_construction_dependencies.py \
  tests/integration/test_construction_execution_boundary.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/public_api_snapshot.py
```

Local validation: 146 focused tests passed, followed by 36 existing hybrid,
public-API, execution-result and CPU vertical-path regressions. Architecture,
dependency, public API, Ruff, Black and strict mypy checks passed. Combined
statement/branch coverage of the shared kernel and construction package was 98%;
the shared kernel and range helper were each 100%. This is not package-wide
coverage. Final default-tier and remote CI status
are recorded in the pull request; focused success does not certify those tiers.
