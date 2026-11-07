# Const/fresh-output register call delivery

Date: 2026-11-01. Sixth-plan Compiler dependency after #986, merged at
819b813e61dd7b433d8014cd7d9d907766483f09 with 57 successful checks and
three configured skips. See the
[semantic contract](../api-changes/FQ-SYNTH-CONST-CALLS-20261101.md).
This is private role/effect evidence, with independent trials 0/2 and
release_authorized=false. Native const syntax is not yet admitted.

## Existing public workflow

Users continue to describe pure high-level arithmetic without gate design:

```python
import flagquantum as fq
from flagquantum import synth


def energy(x: int, y: int) -> int:
    return x*x+y


program = synth.synthesize(
    energy,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    max_qubits=24,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

All sixteen `(x,y,x*x+y)` tuples have probability approximately 1/16. Widths,
preparation, quantum effects and budgets remain user declarations; gates,
scratch, cleanup and physical decoding remain Compiler/Program responsibilities.
This existing public example is not a claim that general quantum function
roles are now available in Python or Qmod.

## New contributor flow

Run `python -m examples.synthesis.tutorials.composition.const_calls.main`. Its owned-IR producer constructs
a value callee with two Const references and a fresh energy output, a Const-only
phase callee, and a combined callee invoking both. The root prepares two
two-bit registers, with an S on x's least-significant bit, calls the combined
callee, then flips x's least-significant bit after return. It uses the existing
private Services bridge to obtain the same Program/run/named-reader product.

For original x,y, the complete state has amplitude
`i**(x&1) * (-1)**(x<y) / 4` at `(x XOR 1, y, x*x+y)`, and zero elsewhere,
including all work slots. Uniform probabilities alone do not certify this
phase. Tests also entangle an external reference with x, compare independent
direct-effect and call producers, repeat Const aliases, retain global signs,
and reuse nonzero scratch across repeated calls under both schedules.

| Responsibility | Implementation |
| --- | --- |
| Const entry and mixed call roles | Extend existing Compiler register verifier |
| Bounded acyclic reference substitution | New private Compiler register_expansion module |
| Arithmetic, marker, exact inverse and work reuse | Existing register/arithmetic lowering |
| Reachable phase target checks | Existing guard applied to expanded root |
| Selection, source identities, Program, memory, execution and reader | Reuse existing boundaries |

Reuse HybridProgram/Block/Value/Operation and quantum.register/effect. The
existing function sequence carries arithmetic and register definitions, with
no new public schema, type, dependency, export or executor. Only Compiler
production code changes. Original semantic identities include every callee;
the existing report and checked plan stay authoritative. The contributor
example is separate from the public high-level example above.

## Role and failure boundaries

Const is basis-value preservation, not classical const or unchanged reduced
state. Reads require initialized references, even for an empty callee that
does not use a parameter. Repeated Const actuals can share one encoding;
fresh outputs must be disjoint from borrows and each other. Widths match
exactly, all outputs return initialized and all references return in order.
Gate targets, reinitialization, compute destinations and nested Output roles
cannot refer to Const parameters. Even cancelling X/X is conservatively refused.
No external state is inspected or certified.

Validate every definition, reject cycles and cap call nesting at 32 independent
of definition order. Count expansion visits, including empty calls, to stop
an exponential no-effect call tree; check final expanded nodes in addition to
source nodes. Existing caps are 4096. Final SSA/effects validate again after
substitution. Reachable nested phases retain exact target restrictions;
an unused phase callee does not restrict a value-only root. Budgets fail before
planning where knowable; memory retains the existing Runtime planning guard.

The native Output-only adapter remains unchanged and frozen for replacement
through this Compiler boundary in the next Ecosystem role-admission change.
Do not add a parallel source-role verifier. Input consumption, native Const,
mutable borrowing, locals, free/drop, within/apply and arbitrary unitary effects
remain separate contracts. Original corpus stays 3/3 with unchanged source and
license hashes; no all-216, hardware, gradient or release admission is claimed.

## Measured resources and acceptance

Native CPU complex128, CPython 3.12.14, PyTorch 2.14.1: the complete contributor
example uses 18 total/used qubits, 10 workspace, 212 operations and depth 163
under either objective, including preparation, both callees and caller mutation.
The two objectives select identical costs here; this is not a claim of improved
scheduling or optimality. Full complex-state and zero-workspace checks use
1e-12 tolerance.

All 71 new cases pass (36 verifier and 35 execution cases). With 71 existing
register/phase cases, the coverage run passes 142 tests. Register verification,
expansion and lowering cover 301/301 statements and 162/162 branch exits;
selection's unchanged broader paths have 64% coverage in this focused run.
This is not a whole-Compiler coverage claim. Broad regression, default-tier
and static checks are recorded in the PR on completion, separating the known
local QFT count failures from this role contract. Run the focused pair with:

```bash
python -m pytest tests/team/compiler/test_construction_register_calls.py tests/integration/test_synthesis_const_calls.py -q
```
