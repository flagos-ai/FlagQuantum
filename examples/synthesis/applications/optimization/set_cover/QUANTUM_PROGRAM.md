# Complete constraint quantum program

Complete retained 23-bit 3-layer quantum program, two nonzero parameter vectors and raw full-state source/native/independent Circuit comparison. Both width/depth instruction sequences and actual budgets checked. Complete Notebook optimizer workflow remains pending.

Native explicit one-dimensional quantum fields replace qstruct storage; keep every original field width, coefficient, constraint, layer and gamma-first/beta-last parameter convention. Preserve all two-bit slack fields with weights1,1 rather than binary1,2. Existing phase arithmetic, disjoint mutable helpers, bounded static loops and RX are reused. No production Compiler, API, budget, dependency or executor changes. Independent affine-square reference emits only one/two-bit truth tables, preserving the existing six-bit enumerated-reference bound. Field-placement mapping explicitly handles declaration order and low-bit indices within each field. No Qmod source/importer in user examples.

Independent references first check all small affine-residual inputs against
one/two-bit factor tables, including nonzero offsets and negative coefficients.
A separate asymmetric-field four-bit dense Circuit validates placement and
mixer order. Full 23-bit complex128 source and native states agree with the
Circuit within1e-12, without phase alignment or probability thresholding.
Compiler workspace is zero, maximum instruction width is two, and width/depth
objectives retain identical complete instruction sequences.

The first independent full-state run exposed a reference layout error: globally
low-bit-first factor numbers had been mistaken for declaration-ordered fields
with local low-bit indexing. The explicit field-placement map and asymmetric
conformance test corrected it before acceptance. The source and native program
semantics were preserved.

Static IR resources for the first binding are native 23qubits,
329operations,depth84 versus frozen original
23qubits,325operations,depth49. Instruction
scheduling differs; this is semantic coverage, not resource optimality,
hardware-basis cost or a guaranteed efficiency improvement.

[README](README.md) contains runnable native and Circuit commands. Complete
Notebook optimization/training remains pending separately. No PR, remote CI or
release is authorized by this local result.
