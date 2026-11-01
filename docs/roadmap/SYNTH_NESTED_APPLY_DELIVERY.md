# Bounded nested apply scopes

This branch follows #993, merged at
f95491d635549234750bf72c198f0e3cf250b433 after its restored CI complete gate
succeeded on the second workflow attempt. See the
[contract](../api-changes/FQ-SYNTH-NESTED-APPLY-20261101.md).

## User workflow

Users may keep dependent temporary computations together without extracting an
artificial helper solely to express nested cleanup:

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    energy:qnum; flag:qbit;
    within {energy=x+y;} apply {
        within {flag=energy<4;} apply {phase(flag,pi);}
    }
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

All sixteen pairs have probability 1/16, with full amplitudes
(-1)**int(x+y<4)/4. Only x/y are returned; energy/flag and arithmetic work are
zero after execution. Run `python -m examples.synthesis_qmod_nested`.
Probabilities alone do not prove phases, so acceptance checks full amplitudes.

## Ownership and bounded semantics

Only two existing Ecosystem production modules change. The parser distinguishes
preparation from apply and bounds scope depth to 32; expression depth and
existing operation/node/width/memory caps remain independent. A redundant parser
nesting Boolean is removed: the scope-depth count also determines whether local
declarations are legal. Source words remain contextual identifiers.

The adapter saves and restores its parent scope list. It cleans only newly
materialized inner locals, in reverse order, leaving parent bindings and cleanup
proof intact. Compiler still owns initialization, Const/alias/dependency checks,
selected exact inverses, local pooling, resource selection and targets. An inner
preparation cannot reinitialize an active parent. Inner references cannot escape;
siblings may reuse released names, retaining their widths with fresh identities.

Preparation nesting, declarations inside blocks, literal/allocation/helper/gate
preparation, mutable apply, arbitrary angles/control/inversion and Input/free/drop
remain refused. No new public API, type, canonical IR, Compiler op, executor,
registry, dependency or duplicate ownership validator is added. Existing helper
composition remains a replacement producer behind the same Program/run/reader.

## Executed evidence

The example on CPU complex128 with Torch 2.13.0 uses 14 allocated and 12 used
qubits, ten hidden work bits, 97 operations (4 H, 68 CX, 20 CCX, 4 X, 1 Z), depth
82. Both width/depth objectives produce these counts. Reserved physical width
still counts toward memory/budgets even when two allocated bits have no gates.
This is an implementation cost, not a global optimum or hardware claim.

There are 44 new acceptance cases: 42 execution/refusal/replacement cases and two
depth-boundary cases. They cover all sixteen two-bit inputs under both objectives,
complex reference entanglement, dependent inner computations, sibling reuse,
continued parent use, phase cancellation, clean work, helper replacement, scope
escape, parent reinitialization, forbidden writes/preparation, depth 32/33 and a
workspace constraint. The former blanket apply-nesting negative now preserves
the live-parent initialization refusal.

The focused 363-case run passes, covering all 548 statements and 238 branch exits
in both production modules. Architecture and public API snapshot checks pass.
An initial documentation command resolved the older installed package because
PYTHONPATH was omitted; repeating it against this checkout passes without changing
the API baseline. The local environment lacks NumPy and emits a Torch warning;
these statevector tests execute through Torch and are not NumPy backend evidence.
The broader native-source regression plus public API candidate suite passes
624 cases in 28.56 seconds. These sets overlap; counts are not added.

Original source/license bytes and the four-original cohort remain unchanged;
this synthetic addition does not certify another original. Developer trials remain
0/2 and release_authorized=false. General Classiq within compatibility, arrays,
consuming roles, SDK/cloud and hardware are not admitted by these local results.
