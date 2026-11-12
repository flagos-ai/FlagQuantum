# Native bounded predicate-gated polynomial phase

The user authorized completing the native example library and continuing each
verified batch. Source review identifies the two retained Knapsack programs as
direct consumers of overlapping read-only predicate/phase inputs. This proposal
extends unreleased captured source semantics within the existing `control` and
`phase` entry points; public exports, signatures and Stable Core contracts remain
unchanged. Classification is `single_device_fast_path`, with implementation
within that authorization and `release_authorized=false`.

## Contract

`control(condition, lambda: phase(expression, coefficient))` may gate a real
polynomial phase by a scalar bit or one integer comparison (`==`, `!=`, `<`,
`<=`, `>`, `>=`). Comparison operands use the existing bounded unsigned-register
integer arithmetic; negative intermediate integer values are exact. Quantum
division, fractional predicates, chained comparisons, host calls and implicit
truth tests of multi-bit words remain unsupported. Entire arrays are not scalar
conditions or numbers; checked element views may supply scalar bits.

The effect is `exp(i*coefficient*expression)` exactly on satisfying basis states.
Predicate and expression may share read-only physical bits. A phase never changes
stored basis values; this permission does not authorize writing a predicate input,
controlling a gate on itself, general nested control or uncontrolled Output escape.
Constant terms remain relative phases inside the satisfying subspace. Existing
accepted scalar control paths retain their representation; newly admitted
polynomial/view/shared-read phase cases use the existing Compiler expansion.

Writable helpers may compose this phase-only control. They retain checked view
bindings, acyclic calls and disjoint writable arguments. They may not publish
Outputs, allocate borrowed ports, escape local views, acquire arbitrary coherent
control of mutable gates, or bypass the existing Const-only adjoint boundary.
No Python callback or comparison is executed on a quantum value.

Reuse `_phase_polynomial.expand_phase(condition=...)`, `qreg.control_phase`,
`qreg.phase`, ownership verification and existing lowering. No program IR,
execution kernel, numerical dependency or public entry point is added. Reuse
24 total phase input bits, eight predicate input bits, degree eight, 256 monomials,
20,000 intermediate products and 128 native expression depth. Every encountered
read has an initialization/ownership witness even after algebraic cancellation.
All coefficients and final sparse angles are finite within ±2**53; exceeding
semantic or resource budgets fails before execution.

## User workflow

```python
import flagquantum as fq
from flagquantum.synth import (
    Output, QNum, allocate, control, hadamard_transform, phase, qfunc, synthesize,
)


@qfunc
def cost(gamma: float, a: QNum, b: QNum) -> None:
    control(2*a + 3*b <= 12, lambda: phase(-3*a - 5*b, gamma))


@qfunc
def main(gamma: float, a: Output[QNum], b: Output[QNum]) -> None:
    allocate(3, a)
    allocate(2, b)
    hadamard_transform(a)
    hadamard_transform(b)
    cost(gamma, a, b)


program = synthesize(main, classical_inputs={"gamma": -0.37}, max_workspace=0)
print(dict(program.probabilities(fq.run(program), registers=("a", "b"))))
```

## Acceptance

- Migrate complete `qaoa.qmod` and `qaoa_knapsack.qmod` quantum programs, preserving
  three layers, all six parameters, the inclusive weight 12 boundary and full
  five-bit storage. Explicit native ports replace source struct fields; source
  field names remain in test-only comparisons. Gamma values precede beta values.
- For both objectives and two nonzero parameter vectors, compare original/native
  raw complex128 states against independent predicate cost tables and Circuit
  rotations, without global-phase alignment. Require zero workspace and report
  actual original/native resources. Notebook optimizers remain separate work.
- Cover negative integer differences, all six comparisons, scalar-bit views,
  shared reads, conditional constant phase with interference, cancelled reads,
  mutable-helper scope restoration and unchanged alias/Output-escape refusals.
- Reject unsupported predicates, uninitialized values, host calls and all relevant
  expansion/resource overflows. Keep existing scalar-control behavior and
  mutable-helper adjoint/control refusals. Recheck the existing replaceable shared
  boundary without modifying its consumers.
- Replay 216 original identities/outcomes/resource rows and six separately approved
  revisions; update native coverage and delivery evidence. No remote CI or PR.

## Reuse applications and measured limits

The same phase infrastructure also migrates the complete eighteen-edge patch
vertex-cover program (eight bits, three layers), the integer/slack optimization
program (eight bits, one layer), and the fixed four-bit tutorial interference
program. The first two use two nonzero parameter vectors and both objectives;
the fixed tutorial has no configurable classical parameters. Independent Circuit
references retain the exact source costs, angle conventions and global phase.

Static IR resources are recorded separately from frozen source baselines:

| Program | Native qubits / instructions / depth | Source qubits / instructions / depth |
| --- | --- | --- |
| Each Knapsack program | 5 / 848 / 721 | 5 / 856 / 721 |
| Patch vertex cover | 8 / 103 / 31 | 8 / 103 / 31 |
| Integer/slack learning | 8 / 45 / 16 | 8 / 45 / 15 |
| Four-bit interference | 4 / 33 / 27 | 5 / 9 / 5 |

Native workspace is zero in all five; source interference uses one clean bit.
Maximum instruction width is two except source interference, where it is three.
Local evolution fusion and layout can alter instruction statistics. Neither the
counts nor source compatibility certify hardware efficiency or optimality.
Complete related Notebook optimizer workflows remain pending.
