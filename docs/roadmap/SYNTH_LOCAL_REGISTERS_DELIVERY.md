# Local ownership, cleanup and physical reuse

Date: 2026-11-01. Continue sixth-plan resource scopes after #991, merged at
e0993b6a9a63d5eedd425cf70e956dc353675684 with 57 successful checks and three
configured skips. See the [contract](../api-changes/FQ-SYNTH-LOCAL-REGISTERS-20261101.md).
Independent developer trials remain 0/2 and release_authorized=false.

## User value and current workflow

Temporary conditions should not become extra user results merely because the
compiler needs storage for them. This slice hides owned local registers from
the named output map, proves they are zero before release and reuses their
physical slots for subsequent locals. It establishes the Compiler dependency
for a future bounded source within/apply entry. Public syntax is unchanged.

Ordinary users keep writing high-level logic in a retained Python source file:

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

This existing public entry needs no decorator, gates, wire numbers, local-pool
design or decoding. Every input pair has probability 1/16; x<=y has negative
amplitude. Probabilities alone do not prove phase. No manual local/release API
is added to this user flow.

Run `python -m examples.synthesis_local_registers` for the new **contributor-only
owned-IR** flow. Const helpers compute less/equal conditions into private locals,
phase them, use their exact selected inverses and release them. Only x/y/reference
are returned; the temporary does not need a fake named zero output. With the
S-prepared input and reference=x, full amplitudes are
i**(x&1)*(-1)**int(x<=y)/4 at (x,y,x). The same Program/run/named reader consumes
inline locals, nested helpers, overlapping locals or existing direct phase calls.

## Semantics and architecture

Only three existing Compiler production modules change. register_profile.py
extends existing storage/effect verification and binding with local ownership;
register_expansion.py clones helper-local identities/labels; register_lowering.py
includes the local band in physical widths and workspace. Reuse existing
quantum.register, HybridProgram/Value/Operation, #991 exact cleanup candidates,
resource selection, target checks, Program, Runtime and reader. No public type,
signature, export, report field, dependency, allocator record, role manager or
new executor is introduced. The shared test producer gains a local declaration
option; it remains test-only.

qreg.local owns fresh storage outside the output list. qreg.release accepts
only a live owned local in fresh proven-zero state and invalidates that reference.
A directly computed local must uncompute first. Initialized storage, including
an allocated or literal zero, cannot be released by assertion. Const/output
release, double release, subsequent access and local escape fail. Every local
must be released before return. Only initialized Const reads and fresh Output
roles cross existing calls; helper locals never do. All definitions verify even
when unused, then the expanded root verifies again.

Locals use a deterministic lowest-available zero pool after all named outputs.
Noncontiguous ports preserve numeric significance. Arithmetic scratch remains
above the complete high-water local band throughout the program. This is
necessary for exact inverse safety when another local is created after a saved
forward computation. Released local slots can serve later locals, but are not
mixed into that scratch tail. Whole workspace equals local-band peak plus
maximum segment scratch; all hidden storage participates in qubit/workspace,
target and memory constraints. Reserved future outputs also count. The simulator
still allocates the complete final-width state, not dynamically resized tensors.

The [official within-apply contract](https://docs.classiq.io/qmod-reference/language-reference/statements/within-apply/)
combines exact inverses, dependency-preserving use and resource reclamation.
This private subset now proves local zero-slot reuse, but does not admit native
blocks, general unitary inversion/control, initialized external free assertions,
arbitrary-state drop, consuming Input, views or full Classiq within compatibility.
No all-216 source claim follows; original sources/licenses/digests remain intact.

## Acceptance and actual resources

119 new cases pass: 30 contract/binding and 89 real workflow cases. They cover
all sixteen two-bit inputs under both schedules and both overlapping/sequential
lifetimes; complex states, absolute/relative phases, external reference
entanglement, interference, aliased Const actuals, nested/repeated helper-local
isolation, noncontiguous numeric storage, named output exclusion and actual
resource/target/memory guards. Raw source circuits with nontrivial middle phases
and checked targets both execute, preventing adjacent inverse cancellation from
masking incorrect scratch placement. Full-state/dirty-work tolerance is 1e-12.

Refusal coverage includes initialized/unknown ownership, stale/double release,
Const/output release, future reserved outputs, malformed operations, local escape
and unused invalid helpers. A 257-local sequential probe fits two physical bits
with one returned bit and one reused internal slot. Model/visit/call budgets
remain bounded. Native local/release/free/drop syntax remains explicitly refused.

741 Compiler-focused cases pass. All three production modules cover 373/373
statements and 202/202 branch exits. Broader/default and static results are
recorded in the PR, including local disk-related recovery or existing QFT
failures; coverage/default cases overlap and are not added to distinct totals.

Measured on CPython 3.12.14, PyTorch 2.14.1, CPU complex128, including preparation,
phases, inverse cleanup and basis reference. Both width/depth objectives select
the same costs here:

| Producer / lifetime | Local band | Whole internal work | Allocated / used qubits | Operations | Depth |
| --- | --- | --- | --- | --- | --- |
| Inline sequential locals | 1 | 6 | 12 / 12 | 58 | 44 |
| Const helpers with sequential locals | 1 | 6 | 12 / 12 | 58 | 44 |
| Overlapping locals | 2 | 7 | 13 / 13 | 68 | 53 |
| Existing direct phase producer | 0 | 5 | 11 / 11 | 54 | 41 |

Sequential/helper costs are 1 U3, 3 H, 14 X, 24 CX, 14 CCX, 2 Z. Overlap costs
are 1 U3, 3 H, 14 X, 26 CX, 22 CCX, 2 Z. Under max_qubits=12 and max_workspace=6,
sequential reuse passes and overlap is rejected with complete resource evidence.
These are distinct effect orders; the selector does not automatically rewrite
overlap into a sequential order. The direct producer remains cheaper for this
closed predicate. No global optimum, speed, hardware or release claim is made.

```bash
python -m pytest tests/team/compiler/test_construction_local_registers.py tests/integration/test_synthesis_local_registers.py -q
python -m examples.synthesis_local_registers
```
