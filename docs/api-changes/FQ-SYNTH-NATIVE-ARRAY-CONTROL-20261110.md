# Native all-ones coherent control

Standing authorization covers native counterparts of every pinned source,
with independent local acceptance before one integration PR. Array control is
the common missing native operation in the 7-, 15- and 50-control X programs.
Only the first two gain complete public native execution in this batch. The
50-control program still requires a separately admitted public native exact-MPS
planning path; compiler capture alone does not count as a completed example.

Broaden the pre-freeze `control` condition annotation to `QNum | QArray[QBit]`.
`QBit` remains a subtype of `QNum`. Existing scalar, predicate phase, helper and
controlled Fourier behavior remains checked through the same capture path.
The added direct spelling is `control(controls, lambda: X(target))`, with
1..256 distinct, initialized control locations and one distinct writable target.
All control bits must be one, including numeric registers; this is not a
nonzero test. Static array elements and nonempty slices retain logical bit order.
Mutable static helpers may use the same operation with disjoint formal views.
The verifier enforces initialization and Const target protection. Output
allocation inside control, nested dynamic callbacks and general array-controlled
unitaries remain outside this admission. Comparison-controlled direct X remains
outside this slice; it is not silently interpreted as an array condition.

Emit the existing indexed `qreg.control_x` and reuse its verifier, scratch
schedule, lowering and resource budgets. No IR opcode, executor, importer,
measurement or reset is introduced. Separate control capture from the larger
program dispatcher by moving the existing scalar implementation intact.

Acceptance covers the two exact original widths against independent Circuits,
every basis input at small widths, complex superpositions and nonzero targets,
active/near-active controls, disjoint slices and helper calls, workspace zero,
absolute phase, Const/alias/uninitialized/view/callback refusals and actual
qubit/workspace/operation/depth/memory ceilings. The original 50-control source
and its existing exact basis-MPS evidence remain preserved and separately
classified. Implementation is authorized locally; release is not authorized.
