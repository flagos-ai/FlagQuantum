# CS-03: exact multiplication and polynomial lowering

PR #953 landed at main `0f10dc875ca285c8c8b78e5d4cab66e63f90a297`.
This delivery implements revision six's private multiplication milestone,
including its phase-oracle and nonzero-gradient acceptance probes. Public Python
capture, Qmod parsing, resource search and global lifetime scheduling remain
later deliveries. No stable export, schema, dependency or Runtime API changes.

## One arithmetic path

`_construction.lowering.lower_arithmetic` is the sole private lowering entry for
the existing unsigned constant/add/multiply/call/return profile. It replaces the
addition-only private prototype; there is no compatibility alias or duplicated
frontend/IR. Full graph/type/port verification still precedes emission.

The Compiler-owned `_arithmetic_kernels` module retains the shared Cuccaro adder
and adds `multiply_accumulate`. Algorithms' existing public adder wrapper and its
API, gate sequence, carry semantics and resource accounting remain unchanged.
Neither arithmetic kernel imports Algorithms, Circuit, Runtime or Simulation.

The multiplication kernel expands binary partial products into controlled
increments of a product register. Its intrinsic operation is modular addition
to an accumulator, with explicit clean scratch. The higher-level lowering
allocates that accumulator at zero, proves a sufficient unsigned width, then
XOR-copies the result into the user's output and reverses the computation:

```text
|inputs,z,0_work> -> |inputs,z XOR encode(f(inputs)),0_work>
```

The kernel's modular accumulator is not the user's XOR target. Overflow is not
silently introduced into the full-precision expression contract. Constant zero,
unequal widths, square expressions and direct subroutine calls use the same
generator; no workload-name, path or expression-hash special cases exist.

## Cleanup and allocation proof

For a partial-product predicate c[0], an increment first computes
`c[k+1] = c[k] AND old_product[k]`, then flips each product bit by its carry.
Because `old_bit = new_bit XOR carry`, the saved carry can be erased using
`c[k] AND new_bit`, followed by `c[k]`. The initial predicate is then erased by
the same X/CX/CCX which prepared it. Every increment restores its local scratch
for arbitrary accumulator values, including coherent inputs.

Repeated controls, such as one diagonal term of x*x, collapse by Boolean
idempotence before constructing an instruction. Distinct operand bits and both
cross terms remain present. The generated program never contains repeated
physical controls in a gate and never copies an unknown state as an independent
quantum state; basis fan-out remains coherent entanglement.

Expression results are retained until the global inverse. Only the multiplier's
locally proven clean scratch is reused by later computations. The reverse sweep
clears those later allocations before the earlier multiplier inverse needs its
scratch. No arbitrary dirty resource, user drop/free assertion or last-use rule
grants reuse.

Addition now borrows its first quantum operand, which Cuccaro restores, rather
than copying it. The other operand is copied before the borrowed operand can
change, including repeated reads. Zero padding, its sum register and carry
allocations remain retained until the reverse sweep. This local kernel property
is not a global schedule optimizer.

The existing 256-qubit, 4,096-node, call-depth and 100,000-operation safety
budgets remain. Multiplication instructions are streamed and checked against
the complete compute/inverse budget during emission. The final XOR copy is
also counted. Insufficient workspace and budget failures return no partial IR;
they describe this construction, not global impossibility. External workspace
zero remains a caller precondition, separate from allocator obligations.

## Executable private example

This uses existing structured records and Circuit/runtime APIs. It is not a
public Python synthesis frontend:

```python
import flagquantum as fq
from flagquantum.compiler._construction.lowering import lower_arithmetic
from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._hybrid.model import (
    Block, HybridProgram, Operation, Region, Value, ValueId,
)

x = Value(ValueId("main", 0), uint_type(2))
y = Value(ValueId("main", 1), uint_type(2))
square = Value(ValueId("main", 2), uint_type(4))
answer = Value(ValueId("main", 3), uint_type(4))
main = HybridProgram("main", Region((Block((x, y), (
    Operation("qint.multiply", (x, x), (square,)),
    Operation("qint.add", (square, y), (answer,)),
    Operation("qint.return", (answer,)),
)),)))
block = lower_arithmetic(
    (main,), {x.id: (0, 1), y.id: (2, 3)}, (4, 5, 6, 7),
    entry="main", n_qubits=18, workspace=tuple(range(8, 18)),
    workspace_state="fresh_zero",
)
# x=3, y=2, target=0: exact answer 11; all ten workspace qubits return zero.
circuit = fq.Circuit(18).x(0).x(1).x(2).compose(block)
options = fq.ExecutionOptions(
    mode="statevector", backend="torch", device="cpu", precision="complex128",
    require_gradients=False, allow_approximate=False, allow_backend_fallback=False,
)
state = fq.run(circuit, options=options).statevector().reshape(-1)
expected = (((3 << 2) | 2) << 4 | 11) << 10
assert abs(complex(state[expected]) - 1) < 1e-12
```

## Resources and acceptance

The existing Compiler estimator supplies the following static counts. Both
entries use two-bit addends and a four-bit output; these are exact Core costs,
not optimality or hardware timing claims.

| Expression | Declared qubits | Workspace | Raw operations / depth | Default-compiled operations / depth |
| --- | --- | --- | --- | --- |
| x*x+y | 18 | 10 | 162 / 123 | 162 / 123 |
| x*y | 16 | 8 | 92 / 72 | 88 / 69 |

The polynomial has 106 CX and 56 CCX instructions before and after default
compilation. The product has 44 CX and 48 CCX before, and 42 CX and 46 CCX after.
CS-02's historical addition-only measurement was 45 operations; borrowing its
preserved operand now yields 41 for the same two-bit logical sum. The existing
public in-place adder still has its original 8*n+1 instructions.

Acceptance evidence:

- Real fq.run CPU execution covers all 16 polynomial input pairs with both zero
  and nonzero XOR targets, comparing complete statevectors including workspace.
  Structural permutation tests exhaust all 16 output target values per input.
- Independent x*y, unequal widths, maximal values and constant multiplication
  (including zero) use the same kernel. Tests exercise constants in both operand
  positions, nonzero accumulators in the intrinsic kernel, repeated controls,
  multiple calls and retained intermediates.
- A seeded complex state jointly superposes inputs, all 16 output targets and
  an external reference qubit; full amplitudes and clean workspace are compared.
  Existing CPU Circuit.state owns arbitrary initial amplitudes. fq.run inputs
  are prepared with gates, without assuming its IR captures that private buffer.
- P1: compute x*x+y, apply Z to output weight four, then invert the same block.
  All result/workspace qubits return zero; each coherent input gains exactly
  `(-1)**((f(x,y)>>2)&1)`, including prepared relative phases.
- P3: one- and two-bit inputs, y=0, with RY(theta) on x's lowest bit. Rebinding
  the same symbolic theta at 0.37, 1.1 and 2.0 verifies expectation cos(theta)
  within 1e-10 and finite nonzero autograd -sin(theta) within 1e-8, both before
  and after existing target compilation. Discrete trainable Tensor inputs are
  rejected rather than silently detached into constants.
- Existing adder, CS-01 semantics, alias/workspace preconditions, dependency
  direction and public API protection regressions remain mandatory.

```bash
python -m pytest tests/team/compiler/test_construction_multiplication.py \
  tests/integration/test_construction_multiplication_execution.py \
  tests/team/compiler/test_construction_addition.py \
  tests/integration/test_construction_addition_execution.py \
  tests/unit/test_algorithms_arithmetic.py -q
python tools/check_qubit_vocabulary.py
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/public_api_snapshot.py
```

The pull request records completed local test counts, coverage, and fresh CI
status. Historical macOS native-extension limitations remain separate from the
new arithmetic acceptance evidence. No multiplication result is inferred from
successful parsing, gate counts or a source-level declaration alone.


## Local validation record

On macOS/arm64, Python 3.12.14 / PyTorch 2.13.0 CPU:

- 239 arithmetic, semantic, dependency, API and CPU integration regressions
  passed in the final selection, plus all 173 vocabulary contract/census cases:
  **412 selected cases passed**.
- Combined statement/branch coverage of the shared arithmetic kernel and private
  construction modules was **98%** (313 statements, 174 branches). The shared
  arithmetic kernel was 100%, lowering 96%, bindings/range helpers 100%.
  This is not package-wide coverage.
- Architecture, dependency policy, public API baseline, team scope, Ruff, Black,
  strict mypy, repository language, documentation links/source-of-truth and legacy
  root API checks passed. Both the updated CS-02 and new CS-03 examples executed.
- The broader local smoke/unit run stopped at the same three existing QFT
  optimization-count failures recorded in PRs #952/#953: 4,145 passed,
  121 skipped, 4,221 deselected and three failed in 205.95 seconds, with
  `--maxfail=3`. This macOS environment lacks the native CPU `_C` extension.
  The two subsequently added boundary cases passed in the final selection above.
  This is not a green local default-tier result.

Fresh PR CI status is recorded in the PR; the green checks of #953 do not certify
this new implementation. Existing Simulation code and its assertions are outside
this change.
