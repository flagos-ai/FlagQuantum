# Reversible amplitude amplification with quantum functions

Prepare A once, then repeat `-A R0 A† Z_indicator` k times. Native qfuncs
compose state loading, reversible predicate XOR, named phase operations and
within/apply regions. `adjoint` supplies the inverse loading. The compiler
chooses gates and physical storage; named indicator probabilities avoid manual
wire numbers and bitstring decoding. `k=0` still applies A.

Predicate XOR uses H/phase kickback and preserves a nonzero marker. The scalar
minus sign is retained: when every prepared state is marked it changes the
absolute state phase even though the output probabilities stay the same.
The independent Circuit builds the full loading/marking unitary and zero
reflection, then uses dense matrix powers. Complete complex-state comparisons
never align away a global phase or discard dirty dropped data. Clean compiler
workspace is checked separately from that retained environment.

All four programs test k=0/1/2 against their unchanged source and independent
Circuit, named marginals, two nonzero marker/data columns, clean workspace and
actual operation caps. The build helper accepts integer k in 0..16; compiler
construction, operation, depth and memory bounds remain authoritative and may
reject a costly instance. All native forms need zero extra clean work. The
current phase truth-table decomposition can use substantially more gates than
the source path; resource reports are separate and imply no optimization claim.
`drop` only hides a name; it neither resets nor reuses the underlying data.

The two counting originals share the same quantum semantics and native
preparation/reflection functions. The circle source keeps its one-bit x/y and
x*x+y*y<4, which marks all four source inputs. VaR keeps all 128decimal
probabilities, a seven-bit unsigned asset, asset<32 and the complete
prefix-conditioned loading unitary. Source mathematical setups are preserved.

Quantum-program acceptance does not certify complete Notebook workflows,
iterative estimators, confidence intervals, data analysis or finance results.
Those are tracked separately. No additional public amplification API or
production lowering was needed for this migration.
