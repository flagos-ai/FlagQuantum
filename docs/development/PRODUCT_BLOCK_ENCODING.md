# Product block encoding

G2-BE04 composes ordered blocks on the same system. Source tuple (A,B,C)
executes C then B then A; its adjoint executes A-adjoint, B-adjoint, C-adjoint.
Packed ancilla contains n-1 success flags first and max(child ancilla widths)
shared ancilla second. Resource storage is shared at its maximum declared width,
under the original assumption that each child restores its resource to zero.
Child factories own directional binding and clean private work. Flags and signal
ancilla are logical outputs, not scratch claimed to be always zero.

Reuse shared metadata, compiler partitioning, existing multi_control_x_ir,
ComponentCost and the existing lazy Circuit append capture. Capture and bound-IR
composition move to a private helper now used by TensorProduct and Product.
No child arithmetic, decoder, runtime or quantum IR is duplicated.

Preserve two upstream distinctions. The direct Product call graph uses all-one
controls for flag costing, while its actual decomposition uses all-zero controls.
Resource models must name that route and retain both QEC profiles. Different
nonzero child resource widths trigger the original resource partition bug;
metadata can be inspected, but numerical binding and resource queries refuse
that unsupported combination rather than silently patching the reference.

The direct source flag model resolves HasLength(n) to an all-one pattern; the
actual Product decomposition checks zeros then flips the flag. The native
resource_model option flag_model selects "call_graph" (default) or
"decomposition", independently of legacy_shims. Caller-supplied child costs
must use the same profile and direction. No resource model is inferred from an
emitted gate count. Both model routes preserve the source distinction even when
they produce equal numbers for a particular width or QEC profile.

Forty original Unitary-child configurations cover single and triple products,
noncommuting T/H/X, zero/mixed ancilla widths, common resource widths, both
directions and first-only/last-only nonzero ancilla boundaries. Six further
source cases use explicitly defined unitary blocks that leak into signal
ancilla, with encoded T/sqrt(2) and H/sqrt(2). Their projected product is
T*H/2 (or its adjoint), not the unflagged product of full unitaries. All complete
source matrices and small measurement branches match; flag outputs are retained
while private work clears. Both source cost routes and both QEC profiles match
for these cases, including the source logical peak.

Two additional measured-Add cases share one clean pool between both child
carries and the larger success predicate. Compare the original component call
tree before exporting each child's unchanged primitive decomposition: flattening
all scopes first can reorder independent bookkeeping and change temporary
physical allocation lifetimes. The existing source-graph reader now has a shared
leaf for the original Partition mapping. It changes no source algorithm. The
scoped gate flow, complete original source matrix, all 16 measurement histories
and both cost routes/profiles agree; logical peak is 10. Across the 48 executable
source configurations there are 192 route/profile model comparisons.

Three source-only cases reproduce the unequal-nonzero-resource partition
failure in decomposition, QEC and width queries. Native metadata retains the
same maximum resource width, while make_on/resource_model refuse before a
factory is invoked. This is an explicit unsupported source boundary, not whole
file acceptance. Signal preparation must be the canonical identity as in source.

Run `python -m examples.algorithms.product_block_encoding`. It uses
circuit.append(op) with nonzero failure branches, arbitrary complex system
input, two measurement erasures and a full exact inverse:

| Quantity | Source call-graph model | Source decomposition model | Emitted IR |
|---|---:|---:|---:|
| T equivalents / T instructions | 9 | 9 | 9 |
| Clifford categories | 9 | 21 | Printed by opcode |
| Measurements | 2 | 2 | 2 |
| Logical peak / reserved qubits | 7 | 7 | 7 |

The zero-signal probability is 1/4. Flags and signal ancilla are meaningful
failure outputs; only private scratch is required to clear after the forward
block. The exact inverse restores all input, flag and work amplitudes. Static
emitted conditional instructions are not reported as measured runtime-active
counts. No synthesis, memory-peak or physical-hardware claim is inferred.

There are 68 new tests and 303 concentrated passes without skips, including
TensorProduct, Phase, Unitary, PREPARE, SELECT and LCU consumers after shared
binding extraction. Strict mypy, Black/Ruff, scope and architecture checks pass.
Source blob 0c806922e9d734f7a2d823c960369c1a443ebb4a matches the pinned tree and
installed reference; neither is modified. Symbolic register binding, unequal
resource-width execution and whole-file acceptance remain open G2 items.
