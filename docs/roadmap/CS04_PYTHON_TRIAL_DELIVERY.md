# CS-04 Python workflow trial

This revision-six slice lets users write supported high-level integer logic
without designing gates, assigning physical qubits or decoding output bitstrings.
It is a private local CPU trial, not a released `flagquantum.synth` namespace.

## Complete user path

Run `PYTHONPATH=. python examples/algorithms/synthesis_trial.py` from the checkout.
The [example](../../examples/algorithms/synthesis_trial.py) contains both basis and
coherent cases. The shortest supported pattern is:

```python
import flagquantum as fq
from flagquantum.services import _synthesis as synth

def polynomial(x: int, y: int) -> int:
    return x * x + y

options = fq.ExecutionOptions(
    mode="statevector", backend="torch", device="cpu",
    precision="complex128", require_gradients=False,
    allow_approximate=False, allow_backend_fallback=False,
)
program = synth.synthesize(
    polynomial, inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
    input_state=synth.InputState.basis(x=3, y=2), max_qubits=18,
)
result = fq.run(program.circuit, options=options)
observed = program.read_distribution(result, registers=("x", "y", "result"))
assert set(observed) == {(3, 2, 11)}
```

Replace `InputState.basis(...)` with `InputState.uniform()` to prepare all sixteen
input pairs coherently. Named joint output contains `(x, y, x*x+y)` with probability
1/16 each. Omitting `registers` returns the result marginal, combining collisions.
The reader reduces the real execution tensor; it does not predict classical
answers from the input or rerun the circuit. Omitting input_state prepares zero
and explicitly records `basis_default_zero` in the report.

Users declare unsigned input widths, initial state and resource ceilings. Output
width, temporary allocation, reversible arithmetic, cleanup, input preparation
and named decoding are supplied by the implementation. Inputs are immutable
basis borrows; arithmetic computes a full-width coherent XOR into fresh zero
output. Widths and constants are structural, not trainable quantum amplitudes.

## Executable syntax and declaration comparison

| Supported | Boundary |
| --- | --- |
| Ordinary `def`, positional parameters, optional `int` annotations | Source must be inspectable; no defaults, variadics, keyword-only or async definitions |
| Single local assignments, aliases, final return | Single assignment; no mutation, loops, branches, effects or arbitrary Python execution |
| Names, nonnegative plain integer literals, `+`, `*`, `**2` | Exact unsigned full precision; no subtraction, division, shifts, comparisons, truncation or floating point |
| Explicit `constants={"scale": 3}` | Every parameter is declared once as quantum input or classical constant; no implicit closure/global capture |
| Calls to `functions=(helper,)` | Declared function names, positional arguments, acyclic bounded expansion; no implicit library lookup |
| Optional `@synth.qfunc` | Same capture, verifier and lowering as ordinary functions; no general Classiq qfunc effects |

The runnable plain/decorated comparison produces identical CircuitIR and resource
estimates. Ordinary functions remain callable for host unit tests; source is
captured when synthesizing. `qfunc` snapshots source at declaration time and
rejects direct host calls. It can report unsupported definition structure earlier.
The trial defaults to ordinary functions for pure integer logic; the decorator
is optional. This is an executable comparison, not a human usability study.

Capture never calls the user body, evals expressions or reads helper globals.
Source, AST size, expression depth, helper depth, expanded operation count,
integer widths and circuit operations have explicit caps. Unsupported syntax
includes filename/line/column diagnostics and a supported alternative where
available. Unknown helpers can be repaired through explicit registration;
workspace/operation failures identify the relevant budget instead of silently
choosing another algorithm. There is one conservative candidate and no depth
objective, optimality guarantee or candidate search in CS-04.

Real `.py` execution and IPython 9.17.1 `InteractiveShell.run_cell` were exercised
with both forms: each produced sixteen uniform joint keys. This supports
source-retaining cells only. Missing inspectable source fails with a specific
instruction to use a source file/source-retaining cell; arbitrary Notebook,
lambda, dynamically generated source and wrapped callable support is not claimed.
CI tests use stdlib linecache to exercise the same retained/missing-source boundary;
IPython is a local validation tool, not a production dependency.

## Existing Circuit and result integration

`program.circuit` goes through the existing `fq.run` path. Synthesis performs no
execution and named reading performs no execution, compilation or replanning.
The resource report is produced by the existing Compiler estimator. For UInt(2)
`x*x+y`, the uniform prepared circuit measured 18 qubits, 166 logical operations
(4 H, 106 CX, 56 CCX), depth 124. The arithmetic block alone has 162 operations,
depth 123. These include preparation where indicated and precede hardware gate
decomposition/routing; they are not device latency or optimality claims.

Advanced users may pass an input-only static unitary preparation CircuitIR and
an explicit injective map covering every named input. Measurements, reset,
opaque matrices, symbolic unbound angles and conflicting preparation declarations
are rejected. Alternatively compose `program.arithmetic` into an existing
parameterized Circuit with its explicit ports and fresh zero output/workspace.
The regression checks an existing `ry(theta)` angle gradient against `-sin(theta)`.
The normal basis/uniform examples require none of these advanced physical maps.

Named reading checks original circuit content and actual Runtime plan identities.
It refuses unrelated same-width circuits, modified sources, routing, noise,
unrecognized transformed executions, density matrices and nonmaterialized or
batched states. Only normalized exact local CPU torch statevectors are supported.
A content-identical CircuitIR JSON round trip is accepted. This is layout and
plan association, not authentication of manually edited ExecutionResult tensors.

Defaults cap selected bits at 16, possible keys at 65536 and scanned amplitudes
at 4194304. The dense read cost is O(2^n) state inspection and O(2^k) marginal/key
materialization, even for sparse output. Budgets are checked before tensor scans
and marginal allocation. Diagnostics report elapsed time, scanned amplitudes,
possible/materialized keys, original probability mass and omitted probability.
Probabilities at or below 1e-14 are filtered without renormalization. In one local
macOS CPU complex128 probe after execution, this 18-qubit joint read scanned
262144 amplitudes, considered 256 possible keys and retained 16. With tracemalloc
and RSS sampling enabled it took 0.054 seconds; Python heap peak was 296601 bytes.
The process RSS baseline and sampled peak were both 237256704 bytes (13 samples,
1 ms requested period). These are separate measurements, not a total tensor
allocator peak or a performance/scalability guarantee. CS-06 must address broader
backend/layout support before relaxing these checks.

## Acceptance and ownership

Compiler tests cover source locations, constants, helpers, aliases, bounded
expansion, unavailable source, source snapshots and rejection without body
execution. Workflow tests cover all sixteen basis inputs; coherent joint and
marginal probabilities; plain/decorated equivalence; error recovery; identity
round trips and rejection; filtering, budgets and no hidden execution; and
advanced Circuit/gradient composition. CS-03 regressions retain nonzero target
XOR, phase-sensitive coherent correctness and auxiliary cleanup coverage.

The [boundary record](../api-changes/FQ-SYNTH-CS04-20261029.md) assigns capture to
Compiler and composition to Services. Existing Hybrid IR, arithmetic kernels,
CoreIR, Compiler, Runtime, Simulation and autograd are reused. New behavior is
restricted Python capture, named preparation composition, a private workflow
artifact and bounded named views. There are no new production dependencies,
stable exports, result fields or serialized contracts.

Local verification: 90 new frontend/workflow cases passed; the combined
construction, arithmetic, execution and API regression run passed 329 cases;
Services and experimental namespace regressions passed another 75. Branch-aware
coverage over the three new production modules was 95% rounded (Compiler capture
91%, Services composition/readout 98% each), not a whole-repository claim.
Ruff, Black, strict mypy for those three modules, architecture/dependency checks,
public API baseline, documentation links, language and vocabulary gates passed.
The executable example and real IPython source-retaining cell probe both passed.

The local default smoke/unit run stopped at the three existing macOS native QFT
fusion-count failures, with 4193 passed and 121 skipped at that point. The local
environment lacks the native CPU extension; these failures also occurred before
CS-04. This is not a green default-suite claim. Fresh Linux PR CI is required;
the preceding PR's results do not validate this branch.
