# Native execution of the classical/tensor notebook

The locked Qualtran v0.7.0 `xcheck_classical_quimb.ipynb` contains 13 code cells.
They prepare X, CNOT, Toffoli and Add(5), test an incorrect X output, draw four
closed diagrams and one expanded Add diagram, then contract Add(32) at a=5555,
b=6666. This batch executes all original code cells without changing their
bytes or assertions, replacing their constructor/drawing/network boundaries
with explicit native test adapters.

## Ownership and interfaces

`prepare_classical_crosscheck` in the existing cross-check example returns the
same closed ProgramRecord consumed by the numerical and drawing paths. Native
bit ordering remains row-major/MSB-first. State/effect preparation and physical
clean-release ownership are unchanged. The original small all-measurement-branch
and append tests remain in the regression.

`classical_tensor_networks.py` composes the existing Compiler tensor selection,
Simulation tensor wiring and explicit cotengra slicing planner. It returns the
actual nodes and existing plan, refusing open ports. The budget is 1,024 peak
tensor elements with one greedy optimizer repeat. This is explicit example
policy, not a hidden simulator fallback. Add remains a factorized network; no
whole-Add matrix is constructed. Logical AND and adjoint suppliers stop tensor
selection before physical measurement flattening. Repeated selection is
idempotent for these examples.

`classical_tensor_scores.py` owns the existing example-specific style and port
symbols, extracted unchanged from the drawing test. Both function ASTs match
their prior definitions. The existing geometry/glyph comparisons now consume
this same owner. Drawer retains layout and rendering; it does not contract or
transform programs. Numerical and drawing paths have separate functions and
reuse the same immutable ProgramRecord.

The foreign API-shaped objects exist only in the notebook test. There is no
production Bloq/Quimb compatibility layer, foreign constructor interception or
global provider registration. The adapter's `flatten` selects native declared
logical leaves; it does not assert identical physical flattening behavior for
every upstream object.

```bash
python examples/algorithms/classical_tensor_networks.py
```

## Measured network and resource accounts

These are actual native node/plan measurements, not process resident RAM or
promised runtime memory bounds. The original notebook's wording about RAM is
retained only while executing its unchanged code; native documentation reports
peak tensor elements explicitly. No equal optimizer-width claim is made.

| Original input | Logical gate bits | Nodes | Largest leaf / planned peak elements | Slices | Complex amplitude |
| --- | --- | --- | --- | --- | --- |
| Add(5), 2+3 | 10 | 69 | 32 / 32 | 1 | 1+0j |
| Add(32), 5555+6666 | 64 | 501 | 32 / 32 | 1 | 1+0j |

At complex128, 32 elements occupy 512 bytes for that single tensor; this excludes
other simultaneously live tensors, input storage, planner and interpreter
memory. Correct and deliberately wrong effects for both widths contract to 1
and 0 respectively, before and after logical selection.

| Original closed input | Actual physical instructions | Source closed QEC model | Physical pool / source peak |
| --- | --- | --- | --- |
| Add(5), 2+3 | X6, H12, T8, Tdg8, CX45, S4, measure4, conditional CZ4, reset4 | AND4, Clifford29, measurement4 | 14 / 14 |
| Add(32), 5555+6666 | X32, H93, T62, Tdg62, CX369, S31, measure31, conditional CZ31, reset31 | AND31, Clifford218, measurement31 | 95 / 95 |

Physical instructions are counted after native binding; source QEC values come
from unchanged upstream models. These are different accounting layers. No
32-bit statevector or exhaustive set of 31-bit measurement histories is claimed.

## Remaining scope

All 13 cells under this explicit boundary replacement and all five native
renders are verified. This closes the concrete notebook workflow, not general
source-wide provider discovery, all Bloq/Quimb methods, optimizer parity, or the
entire source-file inventory. The original notebook file remains byte-identical.
The existing default-suite timeout remains open and is not repeated merely for
this bounded example composition.

## Complete notebook and test-file audit

The user-facing native notebook is
[`examples/notebooks/classical_tensor_crosscheck.ipynb`](../../examples/notebooks/classical_tensor_crosscheck.ipynb).
It runs directly without reference boundary adapters, preserves the original
13-cell progression and concrete inputs, and renders five diagrams. Its test
refuses reference `Bloq.tensor_contract` calls. Markdown teaching text, source
attribution, Python kernel metadata and empty saved outputs are retained or
adapted explicitly. The source notebook has nine Markdown cells and no saved
outputs; there are no unported output-image assets. The native text distinguishes
complex overlap from probability and planned tensor elements from resident RAM.

| Original code-cell index | Native responsibility |
| --- | --- |
| 1 | Native preparation, network and drawing imports; typed display helper |
| 3, 9, 12, 15 | Explicit supported operation choice and width |
| 5, 10, 13, 16 | Prepare/draw X, CNOT, Toffoli and Add(5) |
| 7 | Explicit incorrect X effect and zero-overlap assertion |
| 18 | Select Add(5) logical tensor leaves and draw the expanded graph |
| 19 | Contract both Add(5) representations |
| 21 | Add(32), a=5555/b=6666, native nodes/plan/result and measured peak report |

All five functions of the original `xcheck_classical_quimb_test.py` are exercised
unchanged under explicit boundary replacement: the four original small examples
use the prior executable cross-check composition, while `test_notebook` calls a
replacement notebook executor that compiles the original 13 code-cell strings.
The executor rejects any other notebook name. Imports are resolved before local
constructor replacement so cached module aliases cannot leak into later tests.
A same-process run of the notebook tests followed by all original example tests
passes. The initial isolation failure and its corrected regression are retained
as evidence rather than suppressing the four original assertions.

These two source assets are audited at their own concrete scope. Generic
provider discovery, additional foreign methods, open-boundary label differences
and general planner performance belong to their still-open implementation
parents; they are not methods of this notebook or its five-function test file.

## Local acceptance evidence

Eight new cases and the unchanged related scenarios give 1,531 concentrated
passes in 68.39 seconds (18 retained upstream numerical warnings), plus 84
CPU/tensor-network integration passes in 5.27 seconds (two existing warnings).
Strict typing covers five changed Python files and the concatenated native
notebook code. Black, Ruff, team scope, architecture, multi-level IR and protected
API checks pass. The three example modules have combined 98% statement/branch
coverage (177 statements, 50 branches); direct module-entry guards are also
executed by standalone example runs. Both moved drawing functions have identical
ASTs. Seven relevant source blobs match the frozen inventory, including the
notebook, original tests, tensor traversal, flattening and drawing sources.

Evidence files in the reference workspace use `g2-notebook-pipeline-`, including
`evidence.json`, `regression.log`, `integration.log`, `strict.log` and the retained
`isolation-before-fix.log`. The final regression includes the original tests in
the same process after notebook adaptation, verifying import isolation. The
concrete Q-0897 notebook and Q-0899 test-file audits are complete; general
implementation parents and the default-suite timeout remain open.
