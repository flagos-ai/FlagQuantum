# Source workflow and semantic boundaries

| Original code cell | Native counterpart |
| --- | --- |
| 2: timing and high-level imports | Public FlagQuantum qfuncs and monotonic local duration measurements |
| 3: size 6, degree 3, four phases, maximum width 11, CX objective | Exact source values, fixed logical types, synthesis width budget, actual target CX counts |
| 5: block encoding, phase projectors and QSVT composition | Modular helpers on composed views, reflection, checked phase polynomial, adjoint and function composition |
| 6: CX/U custom transpilation preference | Existing basis converter plus caller-owned immutable exact CX/U3 rules |
| 8: compilation and width/depth/CX/time | Native qfunc compilation, exact target conversion, actual resource estimator and measured local durations |

Bit views are logical and least-significant-first. `[data, block[1]]` is a
seven-bit modular word with data in the low six bits and deletion in the high
bit. `block[0]` selects the shift; `block[2]` selects the reflection. The source
reflection is +1 on zero and -1 elsewhere. Its sign is observable under control.

For an auxiliary basis value `a`, the controlled-X/RZ/controlled-X phase is
`angle * (a - 1/2) + angle * (1 - 2*a) * [block == 0]`. The native polynomial
expresses this same diagonal operator without an extra predicate register. It
retains the original scalar phase. It is an explicit mathematical identity, not
a specialized result for a chosen input or a new compiler optimization.

Acceptance compares native and converted complete complex states to independent
Circuit block matrices for zero input, nonzero block/data/auxiliary boundary
inputs and complex coherent input. A second nonzero phase sequence is checked
independently. All compiler work must return to zero, while logical block and
auxiliary outputs remain retained. Tests also check both RZ columns and coherent
input at nine angles, exact target counts, source-empty Notebook execution,
invalid phases and width/work/operation/depth/memory/conversion refusals.

The independent Circuit reuses the existing source permutation/reflection
reference and remaps wires to named Notebook order. Its largest complex128
matrix is 256 by 256 (1 MiB), explicitly for local correctness inspection. Native
compilation does not use this dense operator. Tests do not claim exhaustive
coverage of all 1024 full-program input columns.

The source has no sampling or plotting stage. No foreign SDK, cloud execution,
private optimizer/custom-pass equivalence, global optimum, hardware or speedup
claim is made. This workflow is `single_device_fast_path`. The other three paper
QSVT reference Notebooks require their own source-semantic acceptance; in
particular their control conventions, wire declarations and angle units cannot
be silently identified with this source.

Native synthesis has a 100,000-instruction budget. Target conversion has a
separate limit of 200,000 added instructions and rechecks physical width eleven.
Reported target counts describe the actual converted IR, rather than implying
that the native instruction limit also caps a different gate basis. The 64 MiB
setting is the synthesis execution-memory budget, not a process-RSS cap. Observed
local durations are single runs without a cross-framework speed comparison.
