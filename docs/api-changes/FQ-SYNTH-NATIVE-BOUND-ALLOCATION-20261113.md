# Native qfunc allocation with bound classical sizes

The user explicitly approved this scope on November 13, 2026. Approval is
recorded from the current conversation; the reviewed proposal SHA-256 is
`6fb90cc1a8a0ebb894f91c02c49b0dac1a65d96fdd3741470580c32406f6e107`. The original proposal and earlier pending
approval evidence are preserved unchanged.


Status: approved for scoped implementation; release is not authorized. Scope: unreleased `flagquantum.synth`, Compiler ownership,
`single_device_fast_path`. No release promotion.

## Concrete need and existing boundary

The original QMCI Pi Notebook varies its coordinate precision through 1, 2, 3, 4.
The native frontend currently rejects `allocate(bits, x)` even when `bits` is an
already-validated synthesis-time integer. Reproductions and the raw diagnostic
are in `flagquantum-work-environments/shor-notebook-delivery-20261113/`.

The existing admission explicitly says: "This does not add variable-width
allocation: `allocate` still requires its existing literal width."
Source: `docs/api-changes/FQ-SYNTH-NATIVE-INTEGER-CONTROL-20261109.md` in the
FlagQuantum worktree. This proposal extends that narrow constructor boundary.

## Proposed user workflow

```python
import flagquantum as fq
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc, synthesize


@qfunc
def uniform(bits: int, x: Output[QNum]) -> None:
    allocate(bits, x)
    hadamard_transform(x)


program = synthesize(
    uniform,
    classical_inputs={"bits": 3},
    max_qubits=3,
    max_workspace=0,
    memory_limit_bytes=1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x",))))
```

The result contains integers 0..7, each with probability 1/8. Changing the binding
to 4 changes logical width and planning resources to 4, without changing the qfunc
or designing gates, physical wire numbers or bitstring decoding.

## Semantic contract

- Admit a size known before quantum capture/lowering. Reuse the authoritative
  bounded static integer evaluator: built-in integer literals, bound `int`
  parameters, static module/closure integer constants, active bounded loop
  indices, supported classical integer-array reads, unary signs and `+`, `-`,
  `*`. Existing evaluator depth 32 and integer-use bound 4096 stay authoritative.
- Final allocation width is a built-in integer in 1..256. Reject bool, float,
  unbound names, zero, negative and oversized results. Reject quantum reads,
  runtime measurements, calls or dynamic object access as size expressions.
- Resolve size before register construction. Check agreement with any existing
  declared literal width, allocation ownership, lifetime, aliases and resource
  budgets through their existing authorities. No physical IR opcode or internal
  effect is added.
- Each synthesis binds its own inputs and resources. Programs already produced
  retain their widths when another synthesis uses a different binding. Helpers
  receive checked integer values through the existing classical binder.
- Preserve the single-argument allocation form and literal-size form, signatures,
  exports, readout semantics and serialized IR. Keep numeric type annotations at
  their currently admitted literal metadata; type aliases and bound names in
  annotations require a separate proposal if later needed.
- Existing source/AST, total effects, operation, width, workspace and memory
  limits apply to the specialized program. No host code is executed to discover
  a size, and no dense state is allocated merely to validate width256.

## Implementation sequence and reuse

1. Record an integration-owned admission and authorization document, with an
   example and an acceptance test contract; preserve Stable Core snapshots.
2. Compiler reuses `frontend_views.static_integer` and `_width_spec` in the
   existing allocation dispatch. Extract allocation capture only if it improves
   the current owner boundary; introduce no second evaluator or public factory.
3. Reuse ordinary qreg allocation, Program lowering, named readout, runtime and
   resource checks. Add a native helper and a Qmod/Circuit comparison if useful
   for the concrete lattice workflow.
4. Run original QMCI sizes 1..4. Reuse the existing example-local count-only
   bounded iterative estimator with epsilon 0.03/alpha 0.01; verify every visited
   Grover circuit independently. Preserve three-bit QPE and explicitly label the
   source's `4*sin(pi*phase)` reporting formula and finite-grid bias.

## Acceptance

- Full complex states match independent Circuits for allocations 1..4 and both
  zero/nonzero classical helper bindings. Uniform-state probabilities and
  coherent controlled uses have the expected named labels and bit order.
- Mixed repeated synthesis 1/4/2/3 and helper calls retain independent widths,
  physical resources and fingerprints. No stale width cache or escaped storage.
- Width 1 and 256 are validated without unsafe dense execution. Width 0/257,
  negatives, bool, float, unresolved names, quantum-dependent expressions,
  unsupported host calls and excessive integer/depth expressions fail before
  lowering with actionable domain diagnostics.
- Mismatched declared widths, duplicate allocation, invalid view/borrow use and
  actual width/workspace/operation/memory budget failures remain rejected.
- Literal-width and single-argument allocation regressions pass; exact Stable
  Core/API snapshot checks remain unchanged. Strict typing, formatting, lint,
  architecture, language and dependency checks pass.
- Original 216 + approved 6 replay retains every identity, stage and measured
  resource row. Freeze Python inputs during the default tier and compare every
  failing node with the previous accepted baseline; no new failing node.
- QMCI Notebook is credited only after all 13 source cells, actual counts,
  independent states, plots, CLI and complete regression gates are accepted.

Approval authorizes the integration admission and scoped Compiler implementation
above, followed by the QMCI workflow. It does not authorize new stable exports,
function signatures, numeric annotation syntax, hardware or release claims.

The existing static evaluator remains authoritative. Allocation first rejects
call and attribute nodes, including `len`, so this admission cannot introduce
quantum-derived sizes through static view-length evaluation. Existing evaluator
errors retain their domain context; final width errors use `program.width`.
