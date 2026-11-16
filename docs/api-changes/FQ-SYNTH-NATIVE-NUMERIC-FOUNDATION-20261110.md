# Shared numeric expression foundation

The next native migration group contains nine independent arithmetic originals:
signed bitwise OR/XOR and equality, two maximum examples using the same
`max(3.5, a)` source expression, exact binary multiplication, negation, and
integer/fractional subtraction. All remain pending native acceptance.

Before extending user syntax, move the existing source adapter's exact word
inference and qint emission into Compiler. Source syntax, source diagnostics and
source ownership remain in Ecosystem. Compiler accepts source-independent node
accessors and emits the existing HybridProgram/qint representation. It does not
parse Qmod, import Ecosystem, execute Python, select a backend or introduce a new
numeric IR. Source identities, inferred formats, diagnostics and operation
resources must remain unchanged. This replaces one implementation; it does not
add a second arithmetic engine.

The foundation preserves existing two's-complement interpretation, signed
extension, range-derived result widths, exact binary decimal scaling,
subtraction, signed equality and bitwise rules. It preserves current refusals
for unsupported signed operators, non-binary decimals, invalid modulo and width
overflow. Do not silently adopt the distinct eight-fraction-bit quantized
program profile for these exact arithmetic cases.

Acceptance uses the existing source adapter and a separate syntax consumer of
the same Compiler inference, independent signed/rational expected values,
existing verifier/lowering, source-located refusals and all216 frozen source
identities/resources. This internal extraction changes no public signature,
annotation admission, allocation form, result schema or native coverage count.

The subsequent native slice must carry sign and binary scale through capture,
initialization, helper validation and named output interpretation. Negative and
extreme values, nonzero output XOR, coherent absolute amplitudes, clean scratch,
precision/overflow rejection and all nine unchanged originals are required before
counting native completion. Public numeric admission is a separate reviewed
contract; current native unsigned-only behavior remains in force meanwhile.
