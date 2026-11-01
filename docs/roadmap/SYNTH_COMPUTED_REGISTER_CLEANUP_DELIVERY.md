# Computed storage cleanup and reuse

Date: 2026-11-01. Sixth-plan lifecycle dependency after #990, merged at
74231c735863034d0b404e195c86c86627436cc2 with 57 successful checks and three
configured skips. See the [contract](../api-changes/FQ-SYNTH-COMPUTED-CLEANUP-20261101.md).
Independent trials remain 0/2 and release_authorized=false.

## User value and current entry

Users need a temporary condition or arithmetic value to remain available across
multiple quantum actions, then be cleaned safely for storage reuse. A locally
named value cannot simply be forgotten: its correlations and its input
dependencies determine whether inverse computation remains valid. This Compiler
slice provides that proof and exact cleanup before introducing native local or
within syntax. It does not expose a manual cleanup requirement to ordinary users.

The existing public high-level phase workflow still needs no gates, flag design,
wire numbers, work scheduling or bit decoding. Save in a Python source file:

```python
import flagquantum as fq
from flagquantum import synth


def marked(x: int, y: int) -> bool:
    return x <= y


program = synth.synthesize(
    marked,
    inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=24,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

Each input pair has probability 1/16, with sign -1 when x<=y. Probabilities alone
do not prove this phase. This is an existing supported user example, not a new
public cleanup API. The new contributor path is
`python -m examples.synthesis_register_cleanup`, whose owned records compute
less/equal flags successively into the same physical slot, phase each flag,
clean each and reinitialize the retained flag to zero. It also retains a copied
basis reference and an S-prepared complex input. Expected amplitudes are
i**(x&1)*(-1)**int(x<=y)/4 at (x,y,0,x).

## Architecture and semantic boundary

Only three existing Compiler production modules change. register_profile.py
extends its current initialization/effect verifier with direct-compute cleanup
authority; register_lowering.py retains the actual selected forward candidate;
lowering.py derives its exact inverse using existing gates/events/candidate
records. Its self-adjoint gate predicate is shared with existing phase cleanup.
Services, Runtime, Ecosystem, public types/signatures/exports/report schemas and
dependencies are unchanged. No role manager, canonical IR, executor, alternative
state verifier or generic scope framework is introduced.

qreg.uncompute(effect,destination) has no attributes/regions. It requires an
unchanged direct qreg.compute, including all direct input encodings. Writes,
even diagonal or cancelling gates, invalidate this conservative proof. Checked
compute/phase/Const borrows preserve it. A dependent output may remain live when
its source is cleaned, but this closed profile then refuses its cleanup. We do
not infer arbitrary helper Output inverses.
Within a helper, directly computed results can clean and reinitialize before
return; every helper and the final expanded root are independently verified.

Cleanup returns the result to fresh proven-zero storage and keeps its physical
allocation/reference. Fresh reads/returns, double cleanup, Const cleanup and
cleanup of allocated/prepared/asserted-zero state fail. A later initialize or
compute reuses the same slot. All outputs still return initialized. This is not
Input consumption, physical freeing, local scope, arbitrary state erasure or
general controlled/inverse block semantics.

The official [within-apply contract](https://docs.classiq.io/qmod-reference/language-reference/statements/within-apply/)
requires const-context use of within-created dependencies and the exact inverse
of the selected preparation, and separately describes qubit reclamation. This
slice establishes a narrower direct-compute proof and selected inverse while
retaining storage; native blocks, local escape rules, resource reclamation and
general control require later contracts. No Classiq within compatibility is
claimed here.

The lowerer reverses the same selected X/CX/CCX instructions, including previous
cleanups/rematerialization and current bindings. It refuses other gates or
payloads and unbalanced value-allocation events. Inverse events exchange
compute/uncompute actions and use the recorded physical allocations; inverse
rematerialization is counted. Both segments enter actual operation budgets and
resource estimates. The same zero work tail is reused; no new inverse candidate
search or silent target relaxation is performed.

## Acceptance and costs

86 new cases pass: 33 verifier/schedule and 53 real workflow cases. They include
all sixteen two-bit inputs under both schedules, absolute and relative signs,
complex external entanglement, interference, repeated physical-flag reuse,
literal-derived zero/fixed values, widened results, aliases, source mutation
after cleanup, nested helpers, dependent live outputs, memory/target/no-execution
guards and the continued refusal of native uncompute syntax. Unknown cleanup authority,
uninitialized reads/returns, double cleanup, Const writes, stale effects,
attributes/regions, mutation and wrong dependency order fail closed.

Existing directly borrowed phase and explicit flag-cleanup producers share the
same unchanged Program/run/named-reader consumer. Full-state and dirty-work
tolerances are 1e-12; there is no simulator-based zero assertion during synthesis.
675 Compiler-focused cases pass. The register verifier covers 211/211 statements
and 138/138 branch exits; register lowering covers 66/66 statements and 20/20
branch exits. Across the three production modules, coverage is 483/499 statements
and 235/248 branch exits, or 96% combined. The new inverse helper and new cleanup
branches are covered; existing unrelated arithmetic paths are not claimed as
fully covered by this invocation. Broad/default results are recorded in the PR; the latter
continues to report existing local QFT failures separately when present.

CPU complex128 measurements use CPython 3.12.14 and PyTorch 2.14.1. Both
objectives choose the same complete contributor circuit here:

| Producer | Allocated / used qubits | Internal work | Operations | Depth |
| --- | --- | --- | --- | --- |
| Explicit compute/phase/cleanup with retained flag | 12 / 12 | 5 | 58 | 44 |
| Existing directly borrowed phase, same zero flag output | 12 / 11 | 5 | 54 | 41 |

The explicit circuit has 1 U3, 3 H, 14 X, 24 CX, 14 CCX and 2 Z, including all
preparation and cleanup. The zero flag remains allocated; no physical-release
or width improvement is claimed. It costs four more CX than the direct producer.
The latter remains the simpler implementation for a single closed predicate.

The pressure graph t=x*y; a=t*x; u=a*y; return t+u at one-bit x/y exercises actual
rematerialization. Raw forward-plus-inverse costs are:

| Schedule | Internal work | Operations | Forward / inverse rematerializations |
| --- | --- | --- | --- |
| retain | 6 | 80 | 0 / 0 |
| recompute | 5 | 116 | 3 / 3 |

These raw schedules exclude input preparation and target transforms. Complete
complex-state execution checks both physical histories. These are measured
candidate costs, not an optimality, hardware, speed or release claim.
Original Qmod sources/licenses/hashes and the four-original cohort are unchanged.

```bash
python -m pytest tests/team/compiler/test_construction_register_cleanup.py tests/integration/test_synthesis_register_cleanup.py -q
python -m examples.synthesis_register_cleanup
```
