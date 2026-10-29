# Independent synthesis developer trials

Status: ready-to-run instructions, **no completed human records**. Two algorithm
developers who did not implement this feature must perform the tasks themselves.
This document and agent-run tests are not their evidence. The initial target is
15 minutes of active work per core task, without the implementer writing code.
Record installation/setup separately; retain unsuccessful attempts and help.

## Freeze the code and environment

Start from merged #964: `7bb7e39c330fd6ab19ab2d598c19f1d3f3b5f1aa`, or record the
actual later review commit you use. Follow the [quick start](../../README.md#install-and-train-your-first-quantum-model)
in a clean environment. Use Python source files in the checkout and run with
`PYTHONPATH=.`. Record `git rev-parse HEAD`, Python/Torch versions, OS, CPU,
dtype and installed native extension status. Do not overwrite another user's
checkout. No Classiq SDK, GPU, credentials or paid service is needed.

Run `python -m examples.algorithms.synthesis_trial` as a warm-up. It should
print basis (3,2,11) with probability 1, 16 uniform joint entries with probability
1/16, a four-entry line example, and real resource statistics. If setup fails,
record the actual failure before requesting help; do not count it as a quantum
semantic failure or silently switch the execution backend.

## Semantics users need to declare

Use `from flagquantum import synth`. An ordinary source-retaining Python
function is enough; `@synth.qfunc` is optional. Supported arithmetic is
nonnegative integer constants, +, *, literal **2, single local assignments,
explicitly registered direct helper calls and one final return. Globals,
closures, mutation, branches, loops, negative/signed/fractional arithmetic and
arbitrary Python execution are outside the initial profile.

Declare quantum input names and UInt widths, optional explicit classical
constants, basis or uniform input preparation, and any resource ceilings.
UInt(2) spans 0..3, not 0..2. Uniform means a coherent superposition over all
basis values, not a single randomly chosen classical input. Reading probabilities
uses an exact CPU statevector in the current profile; it does not mean a QPU
returns every answer in one measurement. Probability dictionaries are display
values, not differentiable training objectives or proof of relative phase.

The main path is `program=synth.synthesize(...)`, `result=fq.run(program)`, then
`program.probabilities(result, registers=(...))`. No gate design, physical wire
numbers, workspace maps, candidate choice or bitstring decoding are required.
A checked Program rejects execution overrides; resynthesize for changed bounds.
See the [current supported entry and full example](../../flagquantum/synth/README.md).

## Core tasks, performed independently

1. Write a new supported expression that is not copied from the warm-up, using
   one- or two-bit inputs. Run a nonzero basis case and check named results
   against an independently calculated integer reference. Then prepare uniform
   input, check all expected named keys/probabilities, and explain how this
   differs from classical sampling. Record the actual code and output.
2. Change an input width within one to two bits and change a resource ceiling.
   Run again and inspect program.report. Explain whether the request succeeds
   and how resource use changes. Stay within available CPU statevector memory;
   do not raise safety caps to force a large simulation. A refused request is
   a legitimate observation, not a reason to alter reference values.
3. Diagnose and repair an error. Use an unsupported syntax construct, or use
   the fixed small pressure workload below with max_qubits=8, max_operations=39,
   max_depth=34 and max_workspace=4. Preserve the original error, explain it,
   and explicitly retry an applicable `error.repairs` suggestion by passing its
   keyword mapping to a new synthesize call. Verify the retry's actual result.
   A construction-budget failure may have no recovery suggestion; do not report
   that as a proved impossibility for every implementation.

The fixed recovery workload is:

```python
def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u
```

Use x/y UInt(1), uniform input. Valid joint results are (0,0,0), (0,1,0),
(1,0,0), (1,1,2), each 1/4. Catch `synth.SynthesisError` to inspect the diagnostic
and its repairs. Suggestions change the request only when you explicitly retry.
Constraints are final resource ceilings; ConstructionBudget bounds the finite
search work. They are different concepts.

## Routed and advanced follow-up

Repeat the one-bit pressure workload with max_qubits=9 and topology="line".
Use the same named result check; do not decode a physical bitstring. This is the
supported ideal line profile, not a calibrated device claim. Inspect resource
selection reasons once, without making that knowledge a core-task prerequisite.

As an optional advanced task, compose an emitted circuit with an existing
Circuit using that API's explicit mappings. Expert composition may use gates
and mappings, but it changes the program and requires fresh planning/validation.
Do not use the original Program's checked report or reader to certify the
modified circuit. The [workspace example](../../examples/synthesis_workspace.py)
shows existing low-level composition; it is separate from the normal user path.

## Blank record, completed only by an actual tester

Copy this section into an issue/PR attachment after doing the work. Leave
unperformed fields explicitly pending. Do not insert this blank template into
the machine admission contract as though it were a completed trial.

- Tester and algorithm background:
- Independence from implementation; prior knowledge of tasks:
- Exact commit, command, Python/Torch/OS/CPU/native-extension/dtype:
- Setup elapsed time and failures:
- Task 1 code, real output, independent reference, active time, result:
- Task 2 old/new widths and bounds, real reports/output, active time, result:
- Task 3 original error, reasoning, explicit retry code/output, active time:
- Routed follow-up actual named results and resources:
- Concepts/docs consulted; each human/AI help request and what it supplied:
- Gates, physical mappings or bit decoding required during core tasks:
- Explanation of input state, result meaning, precision and failure cause:
- Blocking issues, ownership, reproduction and unseen-task retest evidence:
- Reviewer conclusion and remaining release blockers:

A task taking over 15 minutes, implementer-written code, mandatory low-level
work, misunderstood state/precision or an unexplainable failure is a UX finding.
Record and address it; do not average it away. Two actual records and review are
required before changing release/trial status. Passing automated tests alone
cannot satisfy that requirement.
